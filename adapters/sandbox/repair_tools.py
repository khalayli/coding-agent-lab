"""Provide workspace-confined file operations and sandboxed tests."""

from pathlib import Path

from adapters.sandbox.runner import DockerRunner
from core.models import TestResult


def safe_path(workspace: Path, requested_path: str | Path) -> Path:
    """Resolve a relative path and reject escapes, including symbolic links."""
    requested = Path(requested_path)
    if requested.is_absolute():
        raise ValueError("Absolute paths are not allowed")
    root = Path(workspace).resolve()
    target = (root / requested).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Path must remain inside the workspace")
    return target


class RepairTools:
    """Implement the three RepairToolsPort operations for one workspace."""

    def __init__(self, workspace: Path, runner: DockerRunner) -> None:
        self.workspace = Path(workspace).resolve()
        if not self.workspace.is_dir():
            raise ValueError("Workspace must be an existing directory")
        self.runner = runner
        self.changed_files: list[str] = []

    def read_file(self, path: str) -> str:
        """Read a workspace-relative file as UTF-8 text."""
        return safe_path(self.workspace, path).read_text(encoding="utf-8")

    def edit_file(self, path: str, content: str) -> None:
        """Save UTF-8 text and record each changed relative path once."""
        target = safe_path(self.workspace, path)
        if target.exists() and target.read_text(encoding="utf-8") == content:
            return
        target.write_text(content, encoding="utf-8")
        relative = target.relative_to(self.workspace).as_posix()
        if relative not in self.changed_files:
            self.changed_files.append(relative)

    def run_tests(self) -> TestResult:
        """Delegate testing to the Docker-backed runner."""
        return self.runner.run_tests(self.workspace)
