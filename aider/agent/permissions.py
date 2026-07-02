"""Permission policy for agent-mode tools, layered on InputOutput.confirm_ask.

Categories, not individual tools, carry policy so new tools inherit sane
defaults. Shell stays behind both an opt-in flag and explicit_yes_required,
matching how aider treats model-suggested shell commands (--yes-always alone
cannot approve them).
"""

from aider.io import ConfirmGroup

READ = "read"
EDIT = "edit"
SHELL = "shell"
GIT_WRITE = "git_write"


class PermissionPolicy:
    def __init__(self, io, auto_approve_edits=False, allow_shell=False):
        self.io = io
        self.auto_approve_edits = auto_approve_edits
        self.allow_shell = allow_shell
        # One group per category so "(A)ll" answers batch within a category.
        self.groups = {EDIT: ConfirmGroup(), GIT_WRITE: ConfirmGroup()}

    def check(self, category, description, subject=None):
        """Return (allowed: bool, reason: str|None)."""
        if category == READ:
            return True, None

        if category == EDIT:
            if self.auto_approve_edits:
                return True, None
            ok = self.io.confirm_ask(
                f"Allow the agent to {description}?",
                subject=subject,
                group=self.groups[EDIT],
                allow_never=True,
            )
            return bool(ok), None if ok else "denied by user"

        if category == SHELL:
            if not self.allow_shell:
                return False, "shell tool is disabled (enable with --agent-allow-shell)"
            ok = self.io.confirm_ask(
                f"Allow the agent to {description}?",
                subject=subject,
                explicit_yes_required=True,
                allow_never=True,
            )
            return bool(ok), None if ok else "denied by user"

        if category == GIT_WRITE:
            ok = self.io.confirm_ask(
                f"Allow the agent to {description}?",
                subject=subject,
                group=self.groups[GIT_WRITE],
                allow_never=True,
            )
            return bool(ok), None if ok else "denied by user"

        return False, f"unknown permission category {category!r}"
