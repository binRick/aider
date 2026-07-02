"""Tool-call protocol adapter for agent mode.

Two transports:

- "native": the provider's tools API (OpenAI-style tool_calls). Used for
  tool-trained models.
- "text": a prompt-based protocol for models without a tools API (notably
  Ollama's gemma3, which rejects the tools parameter outright). The model
  emits one fenced block per turn:

      ```tool
      {"tool": "edit_file", "args": {"path": "x.py", "search": "a", "replace": "b"}}
      ```

  Large raw payloads (file bodies) travel outside the JSON, between sentinel
  lines, so code never has to survive JSON string escaping:

      <<<PAYLOAD
      ...raw text, fences and all...
      PAYLOAD>>>

  referenced from the JSON as the arg value "<<PAYLOAD>>". A turn that calls
  no tool must call `done`; anything else is malformed, so a fumbled block is
  distinguishable from a final answer.
"""

import json
import re
from dataclasses import dataclass, field

NATIVE = "native"
TEXT = "text"
AUTO = "auto"

PAYLOAD_REF = "<<PAYLOAD>>"
PAYLOAD_START = "<<<PAYLOAD"
PAYLOAD_END = "PAYLOAD>>>"

# ```tool ... ``` with any fence of 3+ backticks; also accept tool_code/json
# tags since small models drift between them.
TOOL_BLOCK_RE = re.compile(
    r"^(?P<fence>`{3,})(?P<tag>tool|tool_code|json)\s*\n(?P<body>.*?)^(?P=fence)\s*$",
    re.MULTILINE | re.DOTALL | re.IGNORECASE,
)

# Ollama's deterministic rejection for models without tool support.
NO_TOOL_SUPPORT_RE = re.compile(r"does not support tools", re.IGNORECASE)


@dataclass
class ToolCall:
    name: str
    args: dict
    call_id: str = ""


@dataclass
class ParsedTurn:
    kind: str  # "tool" | "final" | "malformed"
    tool_calls: list = field(default_factory=list)
    text: str = ""
    error: str = ""


def resolve_protocol(setting, model):
    """Map --agent-tool-protocol / ModelSettings.tool_protocol / auto to a mode."""
    if setting in (NATIVE, TEXT):
        return setting
    model_pref = getattr(model, "tool_protocol", None)
    if model_pref in (NATIVE, TEXT):
        return model_pref
    if (model.info or {}).get("supports_function_calling"):
        return NATIVE
    if model.is_ollama():
        # Local models default to the text protocol; Ollama hard-rejects
        # tools= for models without the capability (e.g. gemma3).
        return TEXT
    return NATIVE


def is_no_tool_support_error(err):
    return bool(NO_TOOL_SUPPORT_RE.search(str(err)))


def parse_text_turn(content):
    """Parse a text-protocol response into a ParsedTurn."""
    if content is None:
        content = ""
    matches = list(TOOL_BLOCK_RE.finditer(content))

    header = None
    for match in matches:
        body = match.group("body").strip()
        try:
            candidate = json.loads(body, strict=False)
        except (ValueError, TypeError):
            candidate = try_repair_json(body)
        if isinstance(candidate, dict) and "tool" in candidate:
            header = (match, candidate)
            break

    if header is None:
        if matches:
            return ParsedTurn(
                kind="malformed",
                text=content,
                error=(
                    "a fenced block was found but it is not valid JSON of the form"
                    ' {"tool": "<name>", "args": {...}}'
                ),
            )
        return ParsedTurn(
            kind="malformed",
            text=content,
            error=(
                "no tool block found. Call a tool in a ```tool fenced block, or call the"
                " `done` tool if the task is finished."
            ),
        )

    match, parsed = header
    name = parsed.get("tool")
    args = parsed.get("args", {})
    if not isinstance(name, str) or not isinstance(args, dict):
        return ParsedTurn(
            kind="malformed",
            text=content,
            error='the tool block must look like {"tool": "<name>", "args": {...}}',
        )

    payload, payload_err = extract_payload(content, match.end())
    if payload_err:
        return ParsedTurn(kind="malformed", text=content, error=payload_err)
    if payload is not None:
        args = substitute_payload(args, payload)

    leading_text = content[: match.start()].strip()
    if name == "done":
        summary = args.get("summary") or leading_text
        return ParsedTurn(kind="final", text=summary or "done")

    return ParsedTurn(kind="tool", tool_calls=[ToolCall(name=name, args=args)], text=leading_text)


def try_repair_json(body):
    """One cheap repair pass for common small-model JSON slips."""
    fixed = body.replace("“", '"').replace("”", '"').replace("‘", "'")
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)  # trailing commas
    try:
        return json.loads(fixed, strict=False)
    except (ValueError, TypeError):
        return None


def extract_payload(content, search_from):
    """Find the sentinel-delimited payload after the tool block, if any."""
    tail = content[search_from:]
    lines = tail.splitlines()
    start = end = None
    for i, line in enumerate(lines):
        if line.strip() == PAYLOAD_START and start is None:
            start = i
        elif line.strip() == PAYLOAD_END and start is not None:
            end = i
            break
    if start is None:
        return None, None
    if end is None:
        return None, f"found {PAYLOAD_START} but no closing {PAYLOAD_END} line"
    return "\n".join(lines[start + 1 : end]), None


def substitute_payload(args, payload):
    substituted = False
    result = {}
    for key, value in args.items():
        if value == PAYLOAD_REF:
            result[key] = payload
            substituted = True
        else:
            result[key] = value
    if not substituted:
        # The model sent a payload but forgot the reference; attach it to the
        # conventional content-carrying arg if one is empty or missing.
        for key in ("content", "replace"):
            if key in result and result[key] in ("", None):
                result[key] = payload
                substituted = True
                break
        if not substituted and "content" not in result and "replace" not in result:
            result["content"] = payload
    return result


def parse_native_message(message):
    """Parse a non-streaming provider message (OpenAI shape) into a ParsedTurn."""
    tool_calls = getattr(message, "tool_calls", None) or []
    content = getattr(message, "content", None) or ""

    if not tool_calls:
        return ParsedTurn(kind="final", text=content)

    calls = []
    for tc in tool_calls:
        func = getattr(tc, "function", None)
        name = getattr(func, "name", None) if func else None
        raw_args = getattr(func, "arguments", None) if func else None
        if not name:
            continue
        try:
            args = json.loads(raw_args or "{}", strict=False)
        except (ValueError, TypeError):
            return ParsedTurn(
                kind="malformed",
                text=content,
                error=f"tool call {name} had unparseable JSON arguments",
            )
        if not isinstance(args, dict):
            args = {}
        calls.append(ToolCall(name=name, args=args, call_id=getattr(tc, "id", "") or ""))

    if not calls:
        return ParsedTurn(kind="final", text=content)
    return ParsedTurn(kind="tool", tool_calls=calls, text=content)


def format_tool_result(name, result, ok=True):
    status = "result" if ok else "error"
    return f"[tool {status}: {name}]\n{result}"


def format_retry_prompt(error):
    return (
        f"Your last reply could not be executed: {error}\n"
        "Reply again following the protocol exactly: one ```tool fenced block containing"
        ' {"tool": "<name>", "args": {...}}, with any raw file content in a'
        f" {PAYLOAD_START} / {PAYLOAD_END} payload. Call the `done` tool when finished."
    )
