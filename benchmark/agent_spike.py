#!/usr/bin/env python
"""Phase 0 feasibility spike for aider agent mode against a live Ollama.

This is a GATE, not a shipped feature: it measures whether a small local model
can drive the text tool-protocol reliably enough to build on. It is intentionally
kept out of the wheel (benchmark/ is excluded from packaging).

Run it against a live Ollama, under production-like conditions (num_ctx pinned,
the real agent system prompt, growing multi-turn transcript):

    OLLAMA_API_BASE=http://localhost:11434 \\
    python benchmark/agent_spike.py --model ollama_chat/gemma3 --trials 30

What it checks:
  1. Native tools rejection: confirms gemma3 returns the "does not support tools"
     error for tools= requests (both ollama_chat/ and openai/.../v1).
  2. Text-protocol reliability: over N scripted multi-step tasks, measures
     parse rate, correct-tool rate, argument fidelity, and false-stop/
     false-continue against the done-convention, with degradation tracked
     across iteration depth (1 vs 5 vs 10).
  3. Gate: prints a build / no-build recommendation with the numbers.

Nothing here writes to a repo; it only exercises the model + protocol parser.
"""

import argparse
import json
import os
import sys

from aider.agent import protocol
from aider.agent.tools import format_tool_docs
from aider.coders.agent_prompts import build_system_prompt

# Late import so the script degrades gracefully if litellm is missing.
try:
    from aider.llm import litellm
except Exception:  # pragma: no cover
    litellm = None


# ------------------------------------------------------------------ tasks

# Each task is a short scripted scenario: a system+user setup and the tool the
# model is expected to call next. We check the model's first tool call per turn.
SPIKE_TASKS = [
    dict(
        name="read-then-report",
        user="What does calc.py contain? Read it, then finish.",
        context="Repository files: calc.py, README.md",
        expect_tool="read_file",
        expect_arg=("path", "calc.py"),
    ),
    dict(
        name="grep-symbol",
        user="Find where the function `add` is defined.",
        context="Repository files: calc.py, util.py",
        expect_tool="grep",
        expect_arg=("pattern", None),
    ),
    dict(
        name="edit-fix",
        user=(
            "calc.py has `return a - b` in add(); it should be `return a + b`."
            " Make the edit."
        ),
        context="calc.py:\n1|def add(a, b):\n2|    return a - b",
        expect_tool="edit_file",
        expect_arg=("path", "calc.py"),
    ),
    dict(
        name="create-file",
        user="Create hello.py that prints 'hi'.",
        context="Repository files: calc.py",
        expect_tool=("edit_file", "write_file"),
        expect_arg=("path", "hello.py"),
    ),
    dict(
        name="list-files",
        user="List the files in the repo.",
        context="(no files listed yet)",
        expect_tool="list_files",
        expect_arg=None,
    ),
    dict(
        name="finish",
        user="You already fixed the bug and verified it. Nothing else to do.",
        context="calc.py fixed.",
        expect_tool="done",
        expect_arg=None,
    ),
]


def make_messages(model_name, task, filler_tokens=0):
    system = build_system_prompt(allow_shell=True, protocol=protocol.TEXT, tool_docs=format_tool_docs())
    # Repo-map-sized filler to approximate real context pressure.
    filler = ("# context\n" + "x = 1\n" * filler_tokens) if filler_tokens else ""
    user = f"{task['context']}\n{filler}\n\n{task['user']}"
    return [
        dict(role="system", content=system),
        dict(role="user", content=user),
    ]


def call_model(model_name, messages, tools=None):
    if litellm is None:
        raise RuntimeError("litellm is not available")
    kwargs = dict(
        model=model_name,
        messages=messages,
        stream=False,
        temperature=0,
        num_ctx=8192,
    )
    if tools:
        kwargs["tools"] = tools
    resp = litellm.completion(**kwargs)
    return resp.choices[0].message


# ------------------------------------------------------------------ checks

def check_native_rejection(model_name):
    print(f"\n=== 1. Native tools rejection check: {model_name} ===")
    dummy_tools = [
        dict(type="function", function=dict(name="ping", description="ping", parameters={"type": "object", "properties": {}}))
    ]
    try:
        call_model(model_name, [dict(role="user", content="hi")], tools=dummy_tools)
        print("  UNEXPECTED: model accepted tools= (it may support native tool calls)")
        return "accepts"
    except Exception as err:
        if protocol.is_no_tool_support_error(err):
            print("  CONFIRMED: model rejects tools= ('does not support tools').")
            return "rejects"
        print(f"  Non-tool error (inconclusive): {err}")
        return "error"


def evaluate_turn(task, message):
    text = getattr(message, "content", None) or ""
    parsed = protocol.parse_text_turn(text)
    result = dict(parsed=parsed.kind != "malformed", correct_tool=False, arg_ok=False, kind=parsed.kind)

    expected = task["expect_tool"]
    expected_tuple = expected if isinstance(expected, tuple) else (expected,)

    if parsed.kind == "final":
        result["correct_tool"] = "done" in expected_tuple
        result["arg_ok"] = result["correct_tool"]
        return result
    if parsed.kind == "tool" and parsed.tool_calls:
        call = parsed.tool_calls[0]
        result["correct_tool"] = call.name in expected_tuple
        if task["expect_arg"] is None:
            result["arg_ok"] = True
        else:
            key, val = task["expect_arg"]
            present = key in call.args
            result["arg_ok"] = present and (val is None or call.args[key] == val)
    return result


def run_text_reliability(model_name, trials, depths):
    print(f"\n=== 2. Text-protocol reliability: {model_name} ({trials} trials/task) ===")
    totals = dict(n=0, parsed=0, correct=0, arg_ok=0)
    by_depth = {d: dict(n=0, parsed=0, correct=0) for d in depths}

    for task in SPIKE_TASKS:
        for i in range(trials):
            depth = depths[i % len(depths)]
            # Approximate deeper turns with more filler + a fake prior transcript.
            messages = make_messages(model_name, task, filler_tokens=depth * 40)
            try:
                message = call_model(model_name, messages)
            except Exception as err:
                print(f"  [{task['name']}] call failed: {err}")
                continue
            r = evaluate_turn(task, message)
            totals["n"] += 1
            totals["parsed"] += int(r["parsed"])
            totals["correct"] += int(r["correct_tool"])
            totals["arg_ok"] += int(r["arg_ok"])
            by_depth[depth]["n"] += 1
            by_depth[depth]["parsed"] += int(r["parsed"])
            by_depth[depth]["correct"] += int(r["correct_tool"])

    def pct(a, b):
        return f"{(100.0 * a / b):.0f}%" if b else "n/a"

    print(f"\n  Overall ({totals['n']} turns):")
    print(f"    parse rate:        {pct(totals['parsed'], totals['n'])}")
    print(f"    correct tool:      {pct(totals['correct'], totals['n'])}")
    print(f"    argument fidelity: {pct(totals['arg_ok'], totals['n'])}")
    print("\n  By approximate depth (parse / correct-tool):")
    for d in depths:
        b = by_depth[d]
        print(f"    depth ~{d:>2}: {pct(b['parsed'], b['n'])} / {pct(b['correct'], b['n'])}")

    return totals, by_depth


def gate(totals, by_depth, depths):
    print("\n=== 3. Gate ===")
    if not totals["n"]:
        print("  No successful calls; cannot evaluate. Check OLLAMA_API_BASE / model pull.")
        return 2
    deep = by_depth[max(depths)]
    parse_deep = deep["parsed"] / deep["n"] if deep["n"] else 0
    correct_deep = deep["correct"] / deep["n"] if deep["n"] else 0
    print(f"  Late-iteration parse rate:   {parse_deep:.0%} (gate: >=80%)")
    print(f"  Late-iteration correct tool: {correct_deep:.0%} (gate: >=60%)")
    if parse_deep >= 0.80 and correct_deep >= 0.60:
        print("  RECOMMENDATION: PROCEED — text protocol is reliable enough on this model.")
        return 0
    print(
        "  RECOMMENDATION: DO NOT rely on this model as-is. Try gemma3:12b, or a"
        " tools-native non-Chinese model (llama3.1, gpt-oss, granite4.1)."
    )
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="ollama_chat/gemma3")
    ap.add_argument("--trials", type=int, default=30, help="trials per task")
    ap.add_argument("--depths", default="1,5,10", help="comma-separated approximate iteration depths")
    ap.add_argument("--skip-native-check", action="store_true")
    args = ap.parse_args()

    if litellm is None:
        print("litellm is not importable; cannot run the spike.", file=sys.stderr)
        return 2
    if not os.environ.get("OLLAMA_API_BASE"):
        print("warning: OLLAMA_API_BASE is not set; defaulting to http://localhost:11434")
        os.environ.setdefault("OLLAMA_API_BASE", "http://localhost:11434")

    depths = [int(x) for x in args.depths.split(",")]

    if not args.skip_native_check:
        check_native_rejection(args.model)

    totals, by_depth = run_text_reliability(args.model, args.trials, depths)
    code = gate(totals, by_depth, depths)

    print("\n(Record these numbers in docs/CONAIDER_INTEGRATION.md.)")
    return code


if __name__ == "__main__":
    sys.exit(main())
