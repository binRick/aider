"""GitLab integration for aider: issue-to-MR and MR discussion helpers.

Config resolution follows common conventions and adds a zero-config fallback
for sandboxes (like conaider) that clone with an ``oauth2:<token>@host`` remote:
the token and base URL are recovered from ``git remote get-url origin`` when no
flag or env var is set.

python-gitlab is an optional dependency; import failures surface an
offline-aware message rather than trying to pip-install at runtime.
"""

import os
import re
import subprocess
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

DEFAULT_CLOSE_KEYWORD = "Closes"


class GitLabError(Exception):
    pass


@dataclass
class GitLabConfig:
    url: str
    token: str
    project: str  # numeric id or "group/project" path
    ssl_verify: bool = True


def _run_git(args, cwd=None):
    try:
        result = subprocess.run(
            ["git"] + args, cwd=cwd, capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def parse_remote_url(remote_url):
    """Return (base_url, token_or_None, project_path) parsed from a git remote.

    Handles https://oauth2:TOKEN@host/group/proj.git and plain
    https://host/group/proj.git and git@host:group/proj.git.
    """
    if not remote_url:
        return None, None, None

    token = None
    project = None
    base = None

    ssh_match = re.match(r"^git@([^:]+):(.+?)(?:\.git)?$", remote_url)
    if ssh_match:
        host, project = ssh_match.group(1), ssh_match.group(2)
        return f"https://{host}", None, project

    split = urlsplit(remote_url)
    if not split.scheme:
        return None, None, None

    host = split.hostname or ""
    if split.username == "oauth2" and split.password:
        token = split.password
    elif split.username and not split.password:
        # e.g. https://TOKEN@host/...
        token = split.username

    port = f":{split.port}" if split.port else ""
    base = urlunsplit((split.scheme, f"{host}{port}", "", "", ""))
    project = split.path.lstrip("/")
    if project.endswith(".git"):
        project = project[: -len(".git")]
    return base, token, project or None


def resolve_config(url=None, token=None, project=None, ssl_verify=None, cwd=None):
    """Resolve GitLab config from explicit args, env vars, then the git remote."""
    remote_url = _run_git(["remote", "get-url", "origin"], cwd=cwd)
    remote_base, remote_token, remote_project = parse_remote_url(remote_url)

    url = (
        url
        or os.environ.get("AIDER_GITLAB_URL")
        or os.environ.get("GITLAB_URL")
        or os.environ.get("GITLAB_HOST")
        or os.environ.get("CI_SERVER_URL")
        or remote_base
    )
    token = (
        token
        or os.environ.get("AIDER_GITLAB_TOKEN")
        or os.environ.get("GITLAB_TOKEN")
        or os.environ.get("GITLAB_PRIVATE_TOKEN")
        or os.environ.get("CI_JOB_TOKEN")
        or remote_token
    )
    project = (
        project
        or os.environ.get("AIDER_GITLAB_PROJECT")
        or os.environ.get("CI_PROJECT_PATH")
        or os.environ.get("CI_PROJECT_ID")
        or remote_project
    )
    if ssl_verify is None:
        env_verify = os.environ.get("GITLAB_SSL_VERIFY")
        ssl_verify = env_verify.lower() not in ("0", "false", "no") if env_verify else True

    missing = [n for n, v in (("url", url), ("token", token), ("project", project)) if not v]
    if missing:
        raise GitLabError(
            "Missing GitLab config: "
            + ", ".join(missing)
            + ". Set --gitlab-url/--gitlab-token or GITLAB_URL/GITLAB_TOKEN, or run inside a"
            " clone whose origin remote carries them."
        )
    return GitLabConfig(url=url.rstrip("/"), token=token, project=project, ssl_verify=ssl_verify)


def _load_gitlab_module():
    try:
        import gitlab  # noqa: F401

        return gitlab
    except ImportError:
        raise GitLabError(
            "python-gitlab is not installed. Install aider with the gitlab extra"
            " (pip install 'aider-chat[gitlab]'). In an air-gapped image, bake it in at"
            " build time; runtime installation is not attempted."
        )


def slugify(text, max_len=40):
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug[:max_len].strip("-") or "issue"


class GitLabClient:
    def __init__(self, config):
        self.config = config
        gitlab = _load_gitlab_module()
        self.gl = gitlab.Gitlab(
            config.url, private_token=config.token, ssl_verify=config.ssl_verify
        )
        self._project = None

    @property
    def project(self):
        if self._project is None:
            self._project = self.gl.projects.get(self.config.project)
        return self._project

    def get_issue(self, iid):
        return self.project.issues.get(iid)

    def branch_name_for_issue(self, issue):
        return f"{issue.iid}-{slugify(getattr(issue, 'title', ''))}"

    def create_merge_request(
        self,
        source_branch,
        target_branch,
        title,
        description,
        draft=True,
        remove_source_branch=True,
    ):
        if draft and not title.lower().startswith("draft:"):
            title = f"Draft: {title}"
        return self.project.mergerequests.create(
            {
                "source_branch": source_branch,
                "target_branch": target_branch,
                "title": title,
                "description": description,
                "remove_source_branch": remove_source_branch,
            }
        )

    def list_unresolved_discussions(self, mr_iid):
        mr = self.project.mergerequests.get(mr_iid)
        unresolved = []
        for disc in mr.discussions.list(get_all=True):
            notes = disc.attributes.get("notes", [])
            if any(n.get("resolvable") and not n.get("resolved") for n in notes):
                unresolved.append(disc)
        return mr, unresolved

    def reply_to_discussion(self, mr, discussion, body):
        return discussion.notes.create({"body": body})

    def resolve_discussion(self, mr, discussion):
        discussion.resolved = True
        discussion.save()


def build_mr_description(issue, close_keyword=DEFAULT_CLOSE_KEYWORD, summary=None):
    parts = []
    if summary:
        parts.append(summary)
    parts.append(f"{close_keyword} #{issue.iid}")
    return "\n\n".join(parts)


def build_task_prompt(issue):
    labels = ", ".join(getattr(issue, "labels", []) or [])
    body = getattr(issue, "description", "") or ""
    prompt = f"Resolve GitLab issue #{issue.iid}: {issue.title}\n\n{body}"
    if labels:
        prompt += f"\n\nLabels: {labels}"
    return prompt


def run_issue_to_mr(
    coder,
    iid,
    url=None,
    token=None,
    project=None,
    close_keyword=DEFAULT_CLOSE_KEYWORD,
    target_branch=None,
):
    """Fetch an issue, let the coder edit, push a branch, open a draft MR.

    Returns the created merge request object, or None on failure. Designed to
    work in conaider: if the session already checked out a feature branch, that
    branch is reused as the MR source instead of creating a new one.
    """
    io = coder.io
    repo = coder.repo
    if not repo:
        io.tool_error("GitLab issue-to-MR requires a git repository.")
        return None

    try:
        config = resolve_config(
            url=url, token=token, project=project, cwd=repo.root
        )
        client = GitLabClient(config)
        issue = client.get_issue(iid)
    except GitLabError as err:
        io.tool_error(str(err))
        return None
    except Exception as err:  # network / API errors from python-gitlab
        io.tool_error(f"GitLab API error: {err}")
        return None

    io.tool_output(f"Fetched issue #{issue.iid}: {issue.title}")

    default_branch = target_branch or _default_branch(repo)
    current = repo.get_current_branch()

    # Reuse an already-checked-out feature branch (conaider's session branch),
    # otherwise create one named after the issue.
    if current and current != default_branch:
        source_branch = current
        io.tool_output(f"Using current branch '{source_branch}' as the MR source.")
    else:
        source_branch = client.branch_name_for_issue(issue)
        try:
            repo.create_branch(source_branch)
            io.tool_output(f"Created branch '{source_branch}'.")
        except Exception as err:
            io.tool_error(f"Could not create branch: {err}")
            return None

    head_before = repo.get_head_commit_sha()

    coder.run(with_message=build_task_prompt(issue), preproc=False)

    head_after = repo.get_head_commit_sha()
    if head_after == head_before:
        io.tool_warning("No commits were made; skipping merge request creation.")
        return None

    try:
        repo.push(branch=source_branch)
        io.tool_output(f"Pushed '{source_branch}' to origin.")
    except Exception as err:
        io.tool_error(f"Push failed: {err}")
        return None

    description = build_mr_description(
        issue,
        close_keyword=close_keyword,
        summary=getattr(coder, "last_aider_commit_message", None),
    )
    try:
        mr = client.create_merge_request(
            source_branch=source_branch,
            target_branch=default_branch,
            title=issue.title,
            description=description,
            draft=True,
        )
    except Exception as err:
        io.tool_error(f"Could not create merge request: {err}")
        return None

    io.tool_output(f"Opened draft MR !{mr.iid}: {mr.web_url}")
    return mr


def _default_branch(repo):
    try:
        return repo.repo.git.symbolic_ref("refs/remotes/origin/HEAD").split("/")[-1]
    except Exception:
        pass
    for candidate in ("main", "master"):
        try:
            repo.repo.git.rev_parse("--verify", candidate)
            return candidate
        except Exception:
            continue
    return "main"
