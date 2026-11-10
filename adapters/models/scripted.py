"""Provide deterministic reply strings without contacting a model."""

from collections import deque


class ScriptedClient:
    """Return pre-written replies in order, like a stack of answer cards."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = deque(replies)

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Return the next reply; messages do not affect the fixed sequence."""
        if not self._replies:
            raise RuntimeError("ScriptedClient has no replies remaining")
        return self._replies.popleft()
