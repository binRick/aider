"""Agent-mode tool registry and dispatcher.

Each tool is a plain function taking (registry, validated-args dict) and
returning a string result for the model. Edits reuse the pure SEARCH/REPLACE
matching from editblock_coder rather than reinventing it; do_replace is
deliberately avoided because it touches the filesystem even on dry runs.
"""

import json
import os
import re
import subprocess

from aider.agent import permissions as perm

# Caps keep tool output from flooding a small local model's context window.
MAX_RESULT_CHARS = 16_000
MAX_GREP_MATCHES = 50
DEFAULT_BASH_TIMEOUT = 60

GIT_READONLY_SUBCOMMANDS = {"status", "diff", "log", "show", "branch", "ls-files", "blame"}


class ToolError(Exception):
    """Raised for tool failures the model should see and react to."""


def truncate_result(text, limit=MAX_RESULT_CHARS):
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 4 :]
    dropped = len(text) - len(head) - len(tail)
    return f"{head}\n... [{dropped} characters truncated] ...\n{tail}"


TOOL_SCHEMAS = [
    dict(
        name="read_file",
        description=(
            "Read a file from the repository. Returns the content with line numbers."
            " Use start_line/limit for large files."
        ),
        parameters=dict(
            type="object",
            properties=dict(
                path=dict(type="string", description="File path relative to the repo root"),
                start_line=dict(type="integer", description="1-based first line to read"),
                limit=dict(type="integer", description="Maximum number of lines to return"),
            ),
            required=["path"],
        ),
    ),
    dict(
        name="list_files",
        description="List repository files, optionally filtered by a substring or glob pattern.",
        parameters=dict(
            type="object",
            properties=dict(
                pattern=dict(type="string", description="Substring or glob to filter paths"),
            ),
            required=[],
        ),
    ),
    dict(
        name="grep",
        description="Search repository files with a regular expression; returns matching lines.",
        parameters=dict(
            type="object",
            properties=dict(
                pattern=dict(type="string", description="Python regular expression"),
                path=dict(type="string", description="Limit search to paths containing this"),
            ),
            required=["pattern"],
        ),
    ),
    dict(
        name="edit_file",
        description=(
            "Edit a file by replacing exactly one occurrence of `search` with `replace`."
            " `search` must match the file content (whitespace-flexible). To create a new"
            " file, pass an empty `search` and the full content as `replace`."
        ),
        parameters=dict(
            type="object",
            properties=dict(
                path=dict(type="string", description="File path relative to the repo root"),
                search=dict(type="string", description="Existing text to find (empty for new file)"),
                replace=dict(type="string", description="Replacement text"),
            ),
            required=["path", "search", "replace"],
        ),
    ),
    dict(
        name="write_file",
        description=(
            "Replace the entire content of a file (or create it). Only allowed for files"
            " you have already read this session, or new files."
        ),
        parameters=dict(
            type="object",
            properties=dict(
                path=dict(type="string", description="File path relative to the repo root"),
                content=dict(type="string", description="The complete new file content"),
            ),
            required=["path", "content"],
        ),
    ),
    dict(
        name="bash",
        description="Run a shell command in the repo root and return its output.",
        parameters=dict(
            type="object",
            properties=dict(
                command=dict(type="string", description="The shell command to run"),
                timeout=dict(type="integer", description="Timeout in seconds (default 60)"),
            ),
            required=["command"],
        ),
    ),
    dict(
        name="git",
        description=(
            "Run a read-only git command. Allowed subcommands:"
            f" {', '.join(sorted(GIT_READONLY_SUBCOMMANDS))}."
        ),
        parameters=dict(
            type="object",
            properties=dict(
                args=dict(type="string", description="git arguments, e.g. 'diff HEAD~1' or 'status'"),
            ),
            required=["args"],
        ),
    ),
    dict(
        name="done",
        description=(
            "Call this when the task is complete (or cannot proceed). Pass a short summary"
            " of what you did as `summary`."
        ),
        parameters=dict(
            type="object",
            properties=dict(
                summary=dict(type="string", description="What was accomplished"),
            ),
            required=["summary"],
        ),
    ),
]

TOOL_CATEGORIES = dict(
    read_file=perm.READ,
    list_files=perm.READ,
    grep=perm.READ,
    edit_file=perm.EDIT,
    write_file=perm.EDIT,
    bash=perm.SHELL,
    git=perm.READ,
    done=perm.READ,
)


class ToolRegistry:
    """Executes tool calls against a repo checkout on behalf of AgentCoder."""

    def __init__(self, root, io, repo=None, policy=None, dry_run=False):
        self.root = os.path.abspath(root)
        self.io = io
        self.repo = repo
        self.policy = policy
        self.dry_run = dry_run
        self.read_files = set()
        self.edited_files = set()
        # Files whose pre-existing (user) changes were already dirty-committed
        # or acknowledged this turn; agent edits to them stay exempt after that.
        self.gated_files = set()
        self.schemas = {schema["name"]: schema for schema in TOOL_SCHEMAS}

    def openai_tools(self):
        return [dict(type="function", function=schema) for schema in TOOL_SCHEMAS]

    def resolve_path(self, path):
        if not path or not str(path).strip():
            raise ToolError("path must not be empty")
        abs_path = os.path.abspath(os.path.join(self.root, str(path)))
        if os.path.commonpath([abs_path, self.root]) != self.root:
            raise ToolError(f"path {path!r} is outside the repository root")
        return abs_path

    def validate_args(self, name, args):
        schema = self.schemas.get(name)
        if not schema:
            known = ", ".join(sorted(self.schemas))
            raise ToolError(f"unknown tool {name!r}; available tools: {known}")
        if not isinstance(args, dict):
            raise ToolError(f"arguments for {name} must be an object, got {type(args).__name__}")
        params = schema["parameters"]
        for req in params.get("required", []):
            if req not in args:
                raise ToolError(f"tool {name} is missing required argument {req!r}")
        for key in args:
            if key not in params["properties"]:
                allowed = ", ".join(params["properties"])
                raise ToolError(f"tool {name} got unknown argument {key!r}; allowed: {allowed}")
        return args

    def check_permission(self, name, args):
        category = TOOL_CATEGORIES.get(name, perm.SHELL)
        if not self.policy:
            return True, None
        description, subject = self.describe_action(name, args)
        return self.policy.check(category, description, subject=subject)

    def describe_action(self, name, args):
        if name == "edit_file":
            return f"edit {args.get('path')}", None
        if name == "write_file":
            return f"write {args.get('path')}", None
        if name == "bash":
            return "run a shell command", args.get("command")
        return name, None

    def dispatch(self, name, args):
        args = self.validate_args(name, args)
        handler = getattr(self, f"tool_{name}")
        return truncate_result(handler(args))

    # ------------------------------------------------------------------ tools

    def tool_read_file(self, args):
        abs_path = self.resolve_path(args["path"])
        if not os.path.isfile(abs_path):
            raise ToolError(f"{args['path']} does not exist")
        content = self.io.read_text(abs_path)
        if content is None:
            raise ToolError(f"could not read {args['path']}")
        self.read_files.add(abs_path)
        lines = content.splitlines()
        start = max(int(args.get("start_line", 1)), 1)
        limit = int(args.get("limit", 400))
        chunk = lines[start - 1 : start - 1 + limit]
        numbered = [f"{i}|{line}" for i, line in enumerate(chunk, start=start)]
        suffix = ""
        if start - 1 + limit < len(lines):
            suffix = f"\n... file has {len(lines)} lines; use start_line to read more"
        return "\n".join(numbered) + suffix if numbered else "(empty file)"

    def tool_list_files(self, args):
        files = self.tracked_files()
        pattern = args.get("pattern")
        if pattern:
            if any(ch in pattern for ch in "*?["):
                import fnmatch

                files = [f for f in files if fnmatch.fnmatch(f, pattern)]
            else:
                files = [f for f in files if pattern in f]
        if not files:
            return "no files matched"
        listing = sorted(files)
        if len(listing) > 500:
            listing = listing[:500] + [f"... and {len(files) - 500} more"]
        return "\n".join(listing)

    def tool_grep(self, args):
        try:
            regex = re.compile(args["pattern"])
        except re.error as err:
            raise ToolError(f"invalid regular expression: {err}")
        path_filter = args.get("path", "")
        matches = []
        for rel_path in self.tracked_files():
            if path_filter and path_filter not in rel_path:
                continue
            abs_path = os.path.join(self.root, rel_path)
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                    for lineno, line in enumerate(fh, start=1):
                        if regex.search(line):
                            matches.append(f"{rel_path}:{lineno}:{line.rstrip()}")
                            if len(matches) >= MAX_GREP_MATCHES:
                                matches.append("... more matches not shown")
                                return "\n".join(matches)
            except OSError:
                continue
        return "\n".join(matches) if matches else "no matches"

    def tool_edit_file(self, args):
        abs_path = self.resolve_path(args["path"])
        search, replace = args["search"], args["replace"]
        exists = os.path.isfile(abs_path)

        if not search.strip():
            if exists:
                raise ToolError(
                    f"{args['path']} already exists; empty `search` is only for creating"
                    " new files. Use a non-empty `search`, or write_file to replace it."
                )
            return self.create_file(abs_path, args["path"], replace)

        if not exists:
            raise ToolError(f"{args['path']} does not exist; pass an empty `search` to create it")

        content = self.io.read_text(abs_path)
        if content is None:
            raise ToolError(f"could not read {args['path']}")

        # Imported lazily: importing aider.coders.editblock_coder triggers the
        # coders package __init__, which imports AgentCoder -> this module.
        from aider.coders.editblock_coder import replace_most_similar_chunk

        new_content = replace_most_similar_chunk(content, search, replace)
        if new_content is None:
            raise ToolError(self.failed_edit_report(args["path"], search, replace, content))

        self.apply_write(abs_path, args["path"], new_content)
        return f"edited {args['path']}"

    def tool_write_file(self, args):
        abs_path = self.resolve_path(args["path"])
        exists = os.path.isfile(abs_path)
        if exists and abs_path not in self.read_files:
            raise ToolError(
                f"refusing to overwrite {args['path']}: read it first with read_file so the"
                " rewrite preserves its current content"
            )
        if exists:
            self.apply_write(abs_path, args["path"], args["content"])
            return f"rewrote {args['path']}"
        return self.create_file(abs_path, args["path"], args["content"])

    def tool_bash(self, args):
        timeout = int(args.get("timeout", DEFAULT_BASH_TIMEOUT))
        try:
            result = subprocess.run(
                args["command"],
                shell=True,
                cwd=self.root,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise ToolError(f"command timed out after {timeout}s")
        output = (result.stdout or "") + (result.stderr or "")
        status = f"[exit status {result.returncode}]"
        return f"{output.rstrip()}\n{status}" if output.strip() else status

    def tool_git(self, args):
        words = str(args["args"]).split()
        if not words:
            raise ToolError("git tool needs a subcommand")
        if words[0] not in GIT_READONLY_SUBCOMMANDS:
            allowed = ", ".join(sorted(GIT_READONLY_SUBCOMMANDS))
            raise ToolError(f"git subcommand {words[0]!r} is not allowed; use one of: {allowed}")
        try:
            result = subprocess.run(
                ["git"] + words,
                cwd=self.root,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=DEFAULT_BASH_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            raise ToolError("git command timed out")
        output = (result.stdout or "") + (result.stderr or "")
        return output.strip() or f"[exit status {result.returncode}]"

    def tool_done(self, args):
        return args.get("summary", "done")

    # ------------------------------------------------------------------ helpers

    def tracked_files(self):
        if self.repo:
            return list(self.repo.get_tracked_files())
        found = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for fname in filenames:
                rel = os.path.relpath(os.path.join(dirpath, fname), self.root)
                found.append(rel)
        return found

    def create_file(self, abs_path, rel_path, content):
        if self.dry_run:
            self.edited_files.add(abs_path)
            return f"created {rel_path} (dry run)"
        os.makedirs(os.path.dirname(abs_path) or self.root, exist_ok=True)
        self.io.write_text(abs_path, content)
        self.edited_files.add(abs_path)
        if self.repo:
            try:
                self.repo.repo.git.add(abs_path)
            except Exception:
                pass
        return f"created {rel_path}"

    def apply_write(self, abs_path, rel_path, new_content):
        if self.dry_run:
            self.edited_files.add(abs_path)
            return
        self.io.write_text(abs_path, new_content)
        self.edited_files.add(abs_path)

    def failed_edit_report(self, rel_path, search, replace, content):
        from aider.coders.editblock_coder import find_similar_lines

        report = f"SearchReplaceNoExactMatch: the `search` text did not match anything in {rel_path}"
        similar = find_similar_lines(search, content)
        if similar:
            report += f"\nDid you mean to match these lines?\n{similar}"
        if replace and replace in content:
            report += (
                f"\nNote: the `replace` text is already present in {rel_path}; the edit may"
                " already be applied."
            )
        report += "\nRe-read the file and retry with an exact `search` snippet."
        return report


def format_tool_docs():
    """Render tool schemas as plain text for the text protocol's system prompt."""
    docs = []
    for schema in TOOL_SCHEMAS:
        params = schema["parameters"]
        arg_lines = []
        for pname, pinfo in params["properties"].items():
            required = " (required)" if pname in params.get("required", []) else ""
            arg_lines.append(f"    - {pname}: {pinfo.get('description', '')}{required}")
        docs.append(f"- {schema['name']}: {schema['description']}\n" + "\n".join(arg_lines))
    return "\n".join(docs)


def json_default(obj):
    return str(obj)


def render_args(args):
    return json.dumps(args, default=json_default, ensure_ascii=False)
