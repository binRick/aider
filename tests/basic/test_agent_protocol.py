from types import SimpleNamespace

from aider.agent import protocol


def test_parse_simple_tool_block():
    content = """Let me read it.
```tool
{"tool": "read_file", "args": {"path": "x.py"}}
```"""
    turn = protocol.parse_text_turn(content)
    assert turn.kind == "tool"
    assert len(turn.tool_calls) == 1
    call = turn.tool_calls[0]
    assert call.name == "read_file"
    assert call.args == {"path": "x.py"}


def test_parse_done_marker_is_final():
    content = """```tool
{"tool": "done", "args": {"summary": "all set"}}
```"""
    turn = protocol.parse_text_turn(content)
    assert turn.kind == "final"
    assert "all set" in turn.text


def test_no_block_is_malformed_not_final():
    # A plain-text reply with no tool block must be malformed so the loop can
    # nudge, rather than silently ending the session.
    turn = protocol.parse_text_turn("I think the bug is in add().")
    assert turn.kind == "malformed"
    assert "done" in turn.error


def test_invalid_json_block_is_malformed():
    content = """```tool
{tool: read_file}
```"""
    turn = protocol.parse_text_turn(content)
    assert turn.kind == "malformed"


def test_json_repair_trailing_comma_and_smart_quotes():
    content = """```tool
{“tool”: “list_files”, “args”: {},}
```"""
    turn = protocol.parse_text_turn(content)
    assert turn.kind == "tool"
    assert turn.tool_calls[0].name == "list_files"


def test_payload_substitution_with_nested_fences():
    content = """```tool
{"tool": "write_file", "args": {"path": "r.md", "content": "<<PAYLOAD>>"}}
```
<<<PAYLOAD
# Title
```python
print("hi")
```
PAYLOAD>>>"""
    turn = protocol.parse_text_turn(content)
    assert turn.kind == "tool"
    payload = turn.tool_calls[0].args["content"]
    assert "```python" in payload
    assert 'print("hi")' in payload


def test_payload_missing_close_is_malformed():
    content = """```tool
{"tool": "write_file", "args": {"path": "r.md", "content": "<<PAYLOAD>>"}}
```
<<<PAYLOAD
no closing sentinel"""
    turn = protocol.parse_text_turn(content)
    assert turn.kind == "malformed"


def test_native_message_with_tool_calls():
    func = SimpleNamespace(name="grep", arguments='{"pattern": "def"}')
    tc = SimpleNamespace(function=func, id="call_1")
    msg = SimpleNamespace(content="searching", tool_calls=[tc])
    turn = protocol.parse_native_message(msg)
    assert turn.kind == "tool"
    assert turn.tool_calls[0].name == "grep"
    assert turn.tool_calls[0].call_id == "call_1"


def test_native_message_no_tool_calls_is_final():
    msg = SimpleNamespace(content="I'm done", tool_calls=None)
    turn = protocol.parse_native_message(msg)
    assert turn.kind == "final"
    assert turn.text == "I'm done"


def test_native_parallel_tool_calls_preserved():
    calls = [
        SimpleNamespace(function=SimpleNamespace(name="read_file", arguments='{"path":"a"}'), id="1"),
        SimpleNamespace(function=SimpleNamespace(name="read_file", arguments='{"path":"b"}'), id="2"),
    ]
    msg = SimpleNamespace(content="", tool_calls=calls)
    turn = protocol.parse_native_message(msg)
    assert turn.kind == "tool"
    assert [c.call_id for c in turn.tool_calls] == ["1", "2"]


def test_resolve_protocol_text_for_ollama_gemma():
    from aider.models import Model

    model = Model("ollama_chat/gemma3")
    assert protocol.resolve_protocol("auto", model) == protocol.TEXT
    # Explicit override wins.
    assert protocol.resolve_protocol("native", model) == protocol.NATIVE


def test_no_tool_support_error_detection():
    assert protocol.is_no_tool_support_error(
        Exception("registry.ollama.ai/library/gemma3 does not support tools")
    )
    assert not protocol.is_no_tool_support_error(Exception("some other error"))
