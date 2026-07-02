from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aider import gitlab as gl
from aider.gitlab import GitLabError


def test_parse_remote_oauth2_url():
    base, token, project = gl.parse_remote_url(
        "http://oauth2:glpat-secret@gitlab/mygroup/myproj.git"
    )
    assert base == "http://gitlab"
    assert token == "glpat-secret"
    assert project == "mygroup/myproj"


def test_parse_remote_ssh_url():
    base, token, project = gl.parse_remote_url("git@gitlab.example.com:group/proj.git")
    assert base == "https://gitlab.example.com"
    assert token is None
    assert project == "group/proj"


def test_parse_remote_plain_https():
    base, token, project = gl.parse_remote_url("https://gitlab.example.com/g/p")
    assert base == "https://gitlab.example.com"
    assert project == "g/p"


def test_resolve_config_from_env(monkeypatch):
    monkeypatch.setenv("GITLAB_URL", "https://gl.local")
    monkeypatch.setenv("GITLAB_TOKEN", "tok")
    monkeypatch.setenv("AIDER_GITLAB_PROJECT", "grp/proj")
    cfg = gl.resolve_config(cwd="/nonexistent")
    assert cfg.url == "https://gl.local"
    assert cfg.token == "tok"
    assert cfg.project == "grp/proj"


def test_resolve_config_missing_raises(monkeypatch):
    for var in (
        "GITLAB_URL",
        "GITLAB_HOST",
        "CI_SERVER_URL",
        "AIDER_GITLAB_URL",
        "GITLAB_TOKEN",
        "GITLAB_PRIVATE_TOKEN",
        "CI_JOB_TOKEN",
        "AIDER_GITLAB_TOKEN",
        "AIDER_GITLAB_PROJECT",
        "CI_PROJECT_PATH",
        "CI_PROJECT_ID",
    ):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(GitLabError):
        gl.resolve_config(cwd="/nonexistent")


def test_slugify():
    assert gl.slugify("Fix the Bug! #42") == "fix-the-bug-42"
    assert gl.slugify("") == "issue"


def test_build_mr_description_close_keyword():
    issue = SimpleNamespace(iid=7, title="Broken thing")
    desc = gl.build_mr_description(issue, close_keyword="Fixes", summary="did it")
    assert "did it" in desc
    assert "Fixes #7" in desc


def test_build_task_prompt_includes_labels():
    issue = SimpleNamespace(iid=3, title="T", description="body", labels=["bug", "p1"])
    prompt = gl.build_task_prompt(issue)
    assert "issue #3" in prompt
    assert "body" in prompt
    assert "bug, p1" in prompt


def test_run_issue_to_mr_happy_path(monkeypatch):
    # Fake repo
    repo = MagicMock()
    repo.root = "/repo"
    repo.get_current_branch.return_value = "main"
    repo.get_head_commit_sha.side_effect = ["sha_before", "sha_after"]
    repo.create_branch.return_value = "3-fix-it"

    # Fake coder that "makes a commit" (head sha changes via side_effect above)
    coder = MagicMock()
    coder.repo = repo
    coder.io = MagicMock()
    coder.last_aider_commit_message = "fix: it"

    issue = SimpleNamespace(iid=3, title="Fix it", description="d", labels=[])
    mr = SimpleNamespace(iid=11, web_url="http://gitlab/mr/11")

    fake_client = MagicMock()
    fake_client.get_issue.return_value = issue
    fake_client.branch_name_for_issue.return_value = "3-fix-it"
    fake_client.create_merge_request.return_value = mr

    monkeypatch.setattr(gl, "resolve_config", lambda **kw: gl.GitLabConfig("u", "t", "p"))
    monkeypatch.setattr(gl, "GitLabClient", lambda cfg: fake_client)
    monkeypatch.setattr(gl, "_default_branch", lambda repo: "main")

    result = gl.run_issue_to_mr(coder, 3)

    assert result is mr
    # A new branch was created (current == default == main), coder ran, branch pushed.
    repo.create_branch.assert_called_once_with("3-fix-it")
    coder.run.assert_called_once()
    repo.push.assert_called_once()
    # MR description carries the close keyword.
    _, kwargs = fake_client.create_merge_request.call_args
    assert "Closes #3" in kwargs["description"]
    assert kwargs["target_branch"] == "main"


def test_run_issue_to_mr_no_commits_skips_mr(monkeypatch):
    repo = MagicMock()
    repo.root = "/repo"
    repo.get_current_branch.return_value = "feature"
    repo.get_head_commit_sha.return_value = "same"  # unchanged before/after
    coder = MagicMock()
    coder.repo = repo
    coder.io = MagicMock()

    issue = SimpleNamespace(iid=5, title="T", description="", labels=[])
    fake_client = MagicMock()
    fake_client.get_issue.return_value = issue

    monkeypatch.setattr(gl, "resolve_config", lambda **kw: gl.GitLabConfig("u", "t", "p"))
    monkeypatch.setattr(gl, "GitLabClient", lambda cfg: fake_client)
    monkeypatch.setattr(gl, "_default_branch", lambda repo: "main")

    result = gl.run_issue_to_mr(coder, 5)
    assert result is None
    fake_client.create_merge_request.assert_not_called()


def test_run_issue_to_mr_reuses_feature_branch(monkeypatch):
    # Simulates conaider: session already on a feature branch.
    repo = MagicMock()
    repo.root = "/repo"
    repo.get_current_branch.return_value = "conaider/20260701"
    repo.get_head_commit_sha.side_effect = ["before", "after"]
    coder = MagicMock()
    coder.repo = repo
    coder.io = MagicMock()
    coder.last_aider_commit_message = "msg"

    issue = SimpleNamespace(iid=9, title="X", description="", labels=[])
    mr = SimpleNamespace(iid=1, web_url="u")
    fake_client = MagicMock()
    fake_client.get_issue.return_value = issue
    fake_client.create_merge_request.return_value = mr

    monkeypatch.setattr(gl, "resolve_config", lambda **kw: gl.GitLabConfig("u", "t", "p"))
    monkeypatch.setattr(gl, "GitLabClient", lambda cfg: fake_client)
    monkeypatch.setattr(gl, "_default_branch", lambda repo: "main")

    gl.run_issue_to_mr(coder, 9)
    # No new branch created; the existing session branch is the MR source.
    repo.create_branch.assert_not_called()
    _, kwargs = fake_client.create_merge_request.call_args
    assert kwargs["source_branch"] == "conaider/20260701"
