from aider.agent.tools import ToolError, ToolRegistry
from aider.io import InputOutput
from aider.utils import ChdirTemporaryDirectory


def make_registry(root):
    return ToolRegistry(root=root, io=InputOutput(yes=True, pretty=False), dry_run=False)


def test_read_and_edit_file():
    with ChdirTemporaryDirectory() as root:
        with open("calc.py", "w") as f:
            f.write("def add(a, b):\n    return a - b\n")
        reg = make_registry(root)

        out = reg.dispatch("read_file", {"path": "calc.py"})
        assert "return a - b" in out
        assert any(p.endswith("calc.py") for p in reg.read_files)

        reg.dispatch(
            "edit_file", {"path": "calc.py", "search": "return a - b", "replace": "return a + b"}
        )
        assert open("calc.py").read() == "def add(a, b):\n    return a + b\n"
        assert any(p.endswith("calc.py") for p in reg.edited_files)


def test_edit_file_no_match_reports_helpfully():
    with ChdirTemporaryDirectory() as root:
        with open("calc.py", "w") as f:
            f.write("def add(a, b):\n    return a + b\n")
        reg = make_registry(root)
        try:
            reg.dispatch("edit_file", {"path": "calc.py", "search": "return a - b", "replace": "x"})
            assert False, "expected ToolError"
        except ToolError as err:
            assert "did not match" in str(err)


def test_create_new_file_with_empty_search():
    with ChdirTemporaryDirectory() as root:
        reg = make_registry(root)
        reg.dispatch("edit_file", {"path": "new.py", "search": "", "replace": "print(1)\n"})
        assert open("new.py").read() == "print(1)\n"


def test_write_file_requires_prior_read():
    with ChdirTemporaryDirectory() as root:
        with open("data.txt", "w") as f:
            f.write("original\n")
        reg = make_registry(root)
        try:
            reg.dispatch("write_file", {"path": "data.txt", "content": "new"})
            assert False, "expected ToolError"
        except ToolError as err:
            assert "read it first" in str(err)
        # After reading, the rewrite is allowed.
        reg.dispatch("read_file", {"path": "data.txt"})
        reg.dispatch("write_file", {"path": "data.txt", "content": "new\n"})
        assert open("data.txt").read() == "new\n"


def test_path_escape_is_rejected():
    with ChdirTemporaryDirectory() as root:
        reg = make_registry(root)
        try:
            reg.dispatch("read_file", {"path": "../../../etc/passwd"})
            assert False, "expected ToolError"
        except ToolError as err:
            assert "outside the repository" in str(err)


def test_grep_finds_matches():
    with ChdirTemporaryDirectory() as root:
        with open("a.py", "w") as f:
            f.write("def foo():\n    pass\n")
        with open("b.py", "w") as f:
            f.write("x = 1\n")
        reg = make_registry(root)
        out = reg.dispatch("grep", {"pattern": r"def \w+"})
        assert "a.py:1:def foo():" in out
        assert "b.py" not in out


def test_bash_runs_and_captures():
    with ChdirTemporaryDirectory() as root:
        reg = make_registry(root)
        out = reg.dispatch("bash", {"command": "echo hello"})
        assert "hello" in out
        assert "exit status 0" in out


def test_git_rejects_non_readonly_subcommand():
    with ChdirTemporaryDirectory() as root:
        reg = make_registry(root)
        try:
            reg.dispatch("git", {"args": "push origin main"})
            assert False, "expected ToolError"
        except ToolError as err:
            assert "not allowed" in str(err)


def test_validate_args_missing_required():
    with ChdirTemporaryDirectory() as root:
        reg = make_registry(root)
        try:
            reg.dispatch("read_file", {})
            assert False, "expected ToolError"
        except ToolError as err:
            assert "missing required" in str(err)


def test_unknown_tool():
    with ChdirTemporaryDirectory() as root:
        reg = make_registry(root)
        try:
            reg.dispatch("frobnicate", {})
            assert False, "expected ToolError"
        except ToolError as err:
            assert "unknown tool" in str(err)
