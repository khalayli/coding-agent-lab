"""Implementation-independent contracts used by the repair workflow."""

from typing import Any, Protocol

from core.models import TestResult


class ChatClient(Protocol):
    """Provide a text reply to a conversation."""

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Return the next reply for role/content messages."""
        ...


class RepairToolsPort(Protocol):
    """Expose the permitted file and test operations in a workspace."""

    def read_file(self, path: str) -> str:
        """Return UTF-8 text from a workspace-relative path."""
        ...

    def edit_file(self, path: str, content: str) -> None:
        """Replace a workspace-relative file with UTF-8 content."""
        ...

    def run_tests(self) -> TestResult:
        """Run the workspace tests through the configured sandbox."""
        ...


class TraceRecorder(Protocol):
    """Describe an optional recorder for public repair events."""

    def record(self, event: str, payload: dict[str, Any]) -> None:
        """Record an event name and its public data."""
        ...
