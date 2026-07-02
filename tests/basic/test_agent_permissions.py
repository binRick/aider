from aider.agent import permissions as perm
from aider.io import InputOutput


def test_read_always_allowed():
    policy = perm.PermissionPolicy(InputOutput(yes=True), allow_shell=False)
    ok, _ = policy.check(perm.READ, "read a file")
    assert ok


def test_edit_auto_approved_when_flag_set():
    policy = perm.PermissionPolicy(InputOutput(yes=False), auto_approve_edits=True)
    ok, _ = policy.check(perm.EDIT, "edit x.py")
    assert ok


def test_edit_prompts_and_respects_yes():
    policy = perm.PermissionPolicy(InputOutput(yes=True), auto_approve_edits=False)
    ok, _ = policy.check(perm.EDIT, "edit x.py")
    assert ok


def test_shell_disabled_by_default():
    policy = perm.PermissionPolicy(InputOutput(yes=True), allow_shell=False)
    ok, reason = policy.check(perm.SHELL, "run a command")
    assert not ok
    assert "disabled" in reason


def test_shell_requires_explicit_yes_even_under_yes_always():
    # --yes-always maps to InputOutput(yes=True); it must NOT auto-approve shell
    # because the prompt is explicit_yes_required (mirrors aider's shell policy).
    io = InputOutput(yes=True)
    policy = perm.PermissionPolicy(io, allow_shell=True)
    ok, _ = policy.check(perm.SHELL, "run a command", subject="rm -rf /")
    assert not ok


def test_shell_allowed_when_user_confirms():
    io = InputOutput(yes=False)
    io.confirm_ask = lambda *a, **k: True
    policy = perm.PermissionPolicy(io, allow_shell=True)
    ok, _ = policy.check(perm.SHELL, "run a command", subject="ls")
    assert ok
