"""Agent mode: a multi-turn tool-calling loop built as a Coder subclass.

The base single-shot pipeline (parse edit blocks -> apply -> commit) is left
untouched. AgentCoder overrides send_message() with its own loop: call the
model, execute the tool calls it returns, feed results back, repeat until the
model calls `done` (or a budget trips). Edits happen through the tool registry
(reusing editblock's pure matching), so exactly one commit is made per turn.
"""

import json

from aider.agent import protocol
from aider.agent.permissions import PermissionPolicy
from aider.agent.tools import ToolError, ToolRegistry, format_tool_docs
from aider.reasoning_tags import remove_reasoning_content
from aider.sendchat import ensure_alternating_roles

from .agent_prompts import AgentPrompts, build_system_prompt
from .base_coder import Coder

# Keep a few recent tool results verbatim; stub older ones when context is tight.
KEEP_RECENT_RESULTS = 6
CONTEXT_HEADROOM = 2048
MAX_MALFORMED_STREAK = 3
STALL_WINDOW = 4


class AgentCoder(Coder):
    edit_format = "agent"
    gpt_prompts = AgentPrompts()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_protocol = protocol.resolve_protocol(
            self.agent_tool_protocol, self.main_model
        )
        self.tool_policy = PermissionPolicy(
            self.io,
            auto_approve_edits=self.agent_auto_approve_edits,
            allow_shell=self.agent_allow_shell,
        )
        self.tools = ToolRegistry(
            root=self.root,
            io=self.io,
            repo=self.repo,
            policy=self.tool_policy,
            dry_run=self.dry_run,
        )

    # The agent system prompt is fully built (shell status + protocol section),
    # so bypass the base placeholder .format() which expects edit-format keys.
    def fmt_system_prompt(self, prompt):
        return build_system_prompt(
            self.agent_allow_shell, self.agent_protocol, format_tool_docs()
        )

    # ------------------------------------------------------------------ loop

    def send_message(self, inp):
        self.event("message_send_starting")
        self.io.llm_started()

        self.cur_messages += [dict(role="user", content=inp)]

        chunks = self.format_messages()
        base_messages = list(chunks.all_messages())
        if not self.check_tokens(base_messages):
            return

        # Per-turn tool state (read set gates write_file; edited set drives commit).
        self.tools.read_files = set()
        self.tools.edited_files = set()

        transcript = []  # assistant/tool messages appended during the loop
        native = self.agent_protocol == protocol.NATIVE
        tools_param = self.tools.openai_tools() if native else None

        final_text = None
        malformed_streak = 0
        recent_calls = []
        stop_reason = "max iterations reached"

        for iteration in range(self.agent_max_iterations):
            transcript = self.enforce_budget(base_messages, transcript)
            messages = base_messages + transcript

            try:
                message_obj, text, ok = self.model_turn(messages, tools_param)
            except KeyboardInterrupt:
                stop_reason = "interrupted"
                break
            if ok == "switch":
                # Model rejected native tools; retry this iteration as text.
                native = False
                tools_param = None
                continue
            if not ok:
                stop_reason = "model call failed"
                break

            if native:
                parsed = protocol.parse_native_message(message_obj)
            else:
                text = remove_reasoning_content(text or "", self.reasoning_tag_name)
                parsed = protocol.parse_text_turn(text)

            if parsed.text:
                self.io.assistant_output(parsed.text)

            if parsed.kind == "final":
                final_text = parsed.text or "Done."
                stop_reason = "done"
                break

            if parsed.kind == "malformed":
                malformed_streak += 1
                if malformed_streak >= MAX_MALFORMED_STREAK:
                    stop_reason = f"stopped after {malformed_streak} malformed replies"
                    break
                self.io.tool_warning(f"Agent reply not understood: {parsed.error}")
                transcript.append(self.assistant_message(native, parsed, message_obj))
                transcript.append(
                    dict(role="user", content=protocol.format_retry_prompt(parsed.error))
                )
                continue

            malformed_streak = 0
            transcript.append(self.assistant_message(native, parsed, message_obj))

            call_sig = self.call_signature(parsed.tool_calls)
            recent_calls.append(call_sig)

            for i, call in enumerate(parsed.tool_calls):
                result, ok = self.execute_tool(call)
                self.render_tool_event(call, result, ok)
                transcript.append(self.tool_result_message(native, call, result, ok, i))

            if self.is_stalled(recent_calls):
                self.io.tool_warning("Agent appears stuck repeating the same action; stopping.")
                stop_reason = "stalled"
                break
        else:
            self.io.tool_warning(
                f"Agent reached the {self.agent_max_iterations}-iteration limit; stopping."
            )

        self.finish_turn(inp, transcript, final_text, stop_reason)

        # send_message must be a generator (run_one lists it, run_stream yields
        # from it); we render directly to io, so nothing is yielded to the TUI.
        return
        yield  # pragma: no cover

    # ------------------------------------------------------------------ model

    def model_turn(self, messages, tools_param):
        """One non-streaming model call with the base retry/backoff policy.

        Tool turns are non-streaming in this MVP: Ollama/litellm streaming does
        not reliably surface tool_call ids. Returns (message_obj, text, ok).
        """
        import time

        from aider.exceptions import LiteLLMExceptions

        litellm_ex = LiteLLMExceptions()
        retry_delay = 0.125

        send_messages = messages
        if self.main_model.is_deepseek_r1():
            send_messages = ensure_alternating_roles(messages)

        while True:
            try:
                hash_object, completion = self.main_model.send_completion(
                    send_messages,
                    functions=None,
                    stream=False,
                    temperature=self.temperature,
                    tools=tools_param,
                    tool_choice="auto" if tools_param else None,
                )
                self.chat_completion_call_hashes.append(hash_object.hexdigest())
                message_obj = completion.choices[0].message
                text = getattr(message_obj, "content", None) or ""
                try:
                    self.calculate_and_show_tokens_and_cost(send_messages, completion)
                except Exception:
                    pass
                return message_obj, text, True
            except litellm_ex.exceptions_tuple() as err:
                ex_info = litellm_ex.get_ex_info(err)
                if ex_info.name == "ContextWindowExceededError":
                    self.io.tool_error("Context window exceeded during agent turn.")
                    return None, "", False
                if protocol.is_no_tool_support_error(err) and tools_param is not None:
                    self.io.tool_warning(
                        "Model rejected the native tools API; switching to the text protocol."
                    )
                    self.agent_protocol = protocol.TEXT
                    return None, "", "switch"
                should_retry = ex_info.retry
                if should_retry:
                    retry_delay *= 2
                    if retry_delay > 60:
                        should_retry = False
                if not should_retry:
                    self.check_and_open_urls(err, ex_info.description)
                    return None, "", False
                self.io.tool_warning(str(err))
                self.io.tool_output(f"Retrying in {retry_delay:.1f} seconds...")
                time.sleep(retry_delay)
                continue
            except KeyboardInterrupt:
                raise

    # ------------------------------------------------------------------ tools

    def execute_tool(self, call):
        try:
            allowed, reason = self.tools.check_permission(call.name, call.args)
        except ToolError as err:
            return str(err), False
        if not allowed:
            return f"permission denied: {reason}", False
        try:
            return self.tools.dispatch(call.name, call.args), True
        except ToolError as err:
            return str(err), False
        except Exception as err:  # defensive: tool bugs shouldn't kill the loop
            return f"tool raised an unexpected error: {err}", False

    def render_tool_event(self, call, result, ok):
        summary = self.tools.describe_action(call.name, call.args)[0]
        self.io.tool_output(f"→ {summary}", bold=True)
        preview = result.strip().splitlines()
        if preview:
            head = "\n".join(preview[:8])
            if len(preview) > 8:
                head += f"\n… ({len(preview) - 8} more lines)"
            if ok:
                self.io.tool_output(head)
            else:
                self.io.tool_error(head)

    # ------------------------------------------------------------------ messages

    def assistant_message(self, native, parsed, message_obj):
        if native and parsed.tool_calls:
            return dict(
                role="assistant",
                content=parsed.text or None,
                tool_calls=[
                    dict(
                        id=call.call_id or f"call_{i}",
                        type="function",
                        function=dict(name=call.name, arguments=json.dumps(call.args)),
                    )
                    for i, call in enumerate(parsed.tool_calls)
                ],
            )
        return dict(role="assistant", content=parsed.text or "")

    def tool_result_message(self, native, call, result, ok, index):
        body = protocol.format_tool_result(call.name, result, ok=ok)
        if native:
            return dict(
                role="tool",
                tool_call_id=call.call_id or f"call_{index}",
                name=call.name,
                content=result,
            )
        return dict(role="user", content=body)

    # ------------------------------------------------------------------ budget

    def effective_context(self):
        params = self.main_model.extra_params or {}
        num_ctx = params.get("num_ctx")
        if num_ctx:
            return int(num_ctx)
        max_input = (self.main_model.info or {}).get("max_input_tokens")
        return int(max_input) if max_input else 8192

    def count_tokens(self, messages):
        try:
            count = self.main_model.token_count(self.stringify(messages))
        except Exception:
            return None
        return count or None

    def stringify(self, messages):
        out = []
        for msg in messages:
            content = msg.get("content")
            if content is None:
                content = ""
            if msg.get("tool_calls"):
                content += " " + json.dumps(msg["tool_calls"])
            out.append(dict(role="user", content=content))
        return out

    def enforce_budget(self, base_messages, transcript):
        limit = self.effective_context() - CONTEXT_HEADROOM
        count = self.count_tokens(base_messages + transcript)
        # A failed/zero count is treated as over-budget (safer than assuming room).
        if count is not None and count <= limit:
            return transcript

        # Evict oldest tool results (role tool, or user-role text tool results),
        # keeping the most recent ones and all assistant turns.
        kept = []
        tool_idxs = [
            i
            for i, m in enumerate(transcript)
            if m.get("role") == "tool"
            or (m.get("role") == "user" and str(m.get("content", "")).startswith("[tool "))
        ]
        evict = set(tool_idxs[:-KEEP_RECENT_RESULTS]) if len(tool_idxs) > KEEP_RECENT_RESULTS else set()
        if not evict:
            return transcript
        for i, msg in enumerate(transcript):
            if i in evict:
                stub = dict(msg)
                stub["content"] = "[older tool result elided to save context]"
                kept.append(stub)
            else:
                kept.append(msg)
        self.io.tool_warning("Context is tight; older tool results were summarized.")
        return kept

    # ------------------------------------------------------------------ stall + finish

    def call_signature(self, tool_calls):
        return tuple(sorted((c.name, json.dumps(c.args, sort_keys=True)) for c in tool_calls))

    def is_stalled(self, recent_calls):
        window = recent_calls[-STALL_WINDOW:]
        return len(window) == STALL_WINDOW and len(set(window)) == 1

    def finish_turn(self, inp, transcript, final_text, stop_reason):
        edited = {self.get_rel_fname(p) for p in self.tools.edited_files}

        if edited and self.repo and self.auto_commits and not self.dry_run:
            context = f"aider (agent): {inp.strip()[:200]}"
            res = self.auto_commit(sorted(self.tools.edited_files), context=context)
            if res:
                self.io.tool_output(res)

        # Scrub the tool transcript into plain-text history so the summarizer,
        # commit-message context, and sanity checks never see role=tool / None.
        summary = self.plain_transcript(transcript, final_text, stop_reason, edited)
        self.cur_messages += [dict(role="assistant", content=summary)]
        self.partial_response_content = summary
        self.move_back_cur_messages(None)

        if edited:
            self.io.tool_output(f"Agent edited: {', '.join(sorted(edited))}")
        self.io.tool_output(f"Agent finished ({stop_reason}).")

    def plain_transcript(self, transcript, final_text, stop_reason, edited):
        lines = []
        for msg in transcript:
            role = msg.get("role")
            if role == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    lines.append(f"[called {fn.get('name')} {fn.get('arguments', '')}]")
                if msg.get("content"):
                    lines.append(str(msg["content"]))
            elif role == "assistant":
                if msg.get("content"):
                    lines.append(str(msg["content"]))
            # tool results are intentionally omitted from the durable summary
        if final_text:
            lines.append(final_text)
        if edited:
            lines.append(f"Files edited: {', '.join(sorted(edited))}.")
        lines.append(f"(agent session ended: {stop_reason})")
        return "\n".join(lines) if lines else "(agent produced no output)"
