from types import SimpleNamespace

from aider.coders import Coder
from aider.dump import dump  # noqa: F401
from aider.io import InputOutput
from aider.models import Model
from aider.utils import GitTemporaryDirectory


def scripted_model(replies, commit_msg="test: agent edit"):
    """Return a send_completion replacement that yields text-protocol replies."""
    it = iter(replies)

    def fake_send_completion(
        self, messages, functions, stream, temperature=None, tools=None, tool_choice=None
    ):
        try:
            content = next(it)
        except StopIteration:
            content = commit_msg  # commit-message model call
        msg = SimpleNamespace(content=content, tool_calls=None)
        completion = SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)
        return SimpleNamespace(hexdigest=lambda: "abc123"), completion

    return fake_send_completion


def make_agent(io):
    model = Model("ollama_chat/gemma3")
    return Coder.create(
        main_model=model,
        edit_format="agent",
        io=io,
        fnames=[],
        agent_tool_protocol="text",
        agent_auto_approve_edits=True,
        stream=False,
    )


def test_agent_reads_edits_and_commits_once():
    with GitTemporaryDirectory() as root:
        import git

        repo = git.Repo(root)
        with open("calc.py", "w") as f:
            f.write("def add(a, b):\n    return a - b\n")
        repo.git.add("calc.py")
        repo.git.commit("-m", "init")
        head_before = repo.head.commit.hexsha

        replies = [
            '```tool\n{"tool": "read_file", "args": {"path": "calc.py"}}\n```',
            '```tool\n{"tool": "edit_file", "args": {"path": "calc.py",'
            ' "search": "return a - b", "replace": "return a + b"}}\n```',
            '```tool\n{"tool": "done", "args": {"summary": "fixed add"}}\n```',
        ]

        io = InputOutput(yes=True, pretty=False)
        Model.send_completion = scripted_model(replies)
        coder = make_agent(io)
        coder.run(with_message="fix add", preproc=False)

        assert open(f"{root}/calc.py").read() == "def add(a, b):\n    return a + b\n"
        commits = list(repo.iter_commits())
        # Exactly one new commit for the agent turn.
        assert len(commits) == 2
        assert repo.head.commit.hexsha != head_before


def test_two_edits_same_file_one_commit():
    with GitTemporaryDirectory() as root:
        import git

        repo = git.Repo(root)
        with open("m.py", "w") as f:
            f.write("a = 1\nb = 2\n")
        repo.git.add("m.py")
        repo.git.commit("-m", "init")

        replies = [
            '```tool\n{"tool": "read_file", "args": {"path": "m.py"}}\n```',
            '```tool\n{"tool": "edit_file", "args": {"path": "m.py", "search": "a = 1",'
            ' "replace": "a = 10"}}\n```',
            '```tool\n{"tool": "edit_file", "args": {"path": "m.py", "search": "b = 2",'
            ' "replace": "b = 20"}}\n```',
            '```tool\n{"tool": "done", "args": {"summary": "bumped"}}\n```',
        ]

        io = InputOutput(yes=True, pretty=False)
        Model.send_completion = scripted_model(replies)
        coder = make_agent(io)
        coder.run(with_message="bump", preproc=False)

        assert open(f"{root}/m.py").read() == "a = 10\nb = 20\n"
        # Two edits to the same file must still yield a single commit for the turn.
        assert len(list(repo.iter_commits())) == 2


def test_malformed_streak_stops_gracefully():
    with GitTemporaryDirectory():
        replies = ["no tool here", "still no tool", "nope"]
        io = InputOutput(yes=True, pretty=False)
        Model.send_completion = scripted_model(replies)
        coder = make_agent(io)
        # Should stop after the malformed streak without raising.
        coder.run(with_message="do something", preproc=False)


def test_history_scrubbed_of_tool_messages():
    with GitTemporaryDirectory() as root:
        import git

        repo = git.Repo(root)
        with open("f.py", "w") as f:
            f.write("x = 0\n")
        repo.git.add("f.py")
        repo.git.commit("-m", "init")

        replies = [
            '```tool\n{"tool": "read_file", "args": {"path": "f.py"}}\n```',
            '```tool\n{"tool": "done", "args": {"summary": "looked"}}\n```',
        ]
        io = InputOutput(yes=True, pretty=False)
        Model.send_completion = scripted_model(replies)
        coder = make_agent(io)
        coder.run(with_message="inspect", preproc=False)

        # No tool-role or None-content messages leak into durable history.
        for msg in coder.done_messages:
            assert msg["role"] in ("user", "assistant")
            assert msg["content"] is not None
