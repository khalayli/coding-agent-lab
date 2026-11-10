"""Assemble the core workflow, concrete tools, and scripted answer cards."""

import json
from pathlib import Path

from adapters.models.scripted import ScriptedClient
from adapters.sandbox.repair_tools import RepairTools
from adapters.sandbox.runner import DockerRunner, EVALS_ROOT
from core.ports import ChatClient
from core.repair_agent import RepairAgent
from core.repair_service import RepairService, TraceSink


# Each entry supplies the source path, test path, and complete corrected source.
_SCRIPTED_REPAIRS = {
    "arithmetic": (
        "calculator.py", "test_calculator.py",
        'def add(left, right):\n'
        '    """Return the sum of two numbers."""\n'
        '    return left + right\n',
    ),
    "balanced_brackets": (
        "brackets.py", "test_brackets.py",
        'def is_balanced(text):\n'
        '    """Return whether parentheses, brackets, and braces are properly nested."""\n'
        '    opening = "([{"\n'
        '    closing = ")]}"\n'
        '    pairs = {")": "(", "]": "[", "}": "{"}\n'
        '    stack = []\n'
        '\n'
        '    for character in text:\n'
        '        if character in opening:\n'
        '            stack.append(character)\n'
        '        elif character in closing:\n'
        '            if not stack:\n'
        '                return False\n'
        '            if stack.pop() != pairs[character]:\n'
        '                return False\n'
        '\n'
        '    return not stack\n',
    ),
    "median": (
        "statistics_tools.py", "test_statistics_tools.py",
        'def median(values):\n'
        '    """Return the median value without changing the input sequence."""\n'
        '    ordered = sorted(values)\n'
        '    middle = len(ordered) // 2\n'
        '    if len(ordered) % 2 == 0:\n'
        '        return (ordered[middle - 1] + ordered[middle]) / 2\n'
        '    return ordered[middle]\n',
    ),
    "multi_function": (
        "text_tools.py", "test_text_tools.py",
        'def normalize_name(name):\n'
        '    """Trim surrounding whitespace and lowercase a supplied name."""\n'
        '    return name.strip().lower()\n'
        '\n'
        '\n'
        'def greeting(name):\n'
        '    """Create a greeting from a normalized name."""\n'
        '    return f"Hello, {normalize_name(name).title()}!"\n'
        '\n'
        '\n'
        'def initials(name):\n'
        '    """Return uppercase initials for the whitespace-separated parts of a name."""\n'
        '    return "".join(part[0].upper() for part in name.split())\n',
    ),
    "unique_words": (
        "word_tools.py", "test_word_tools.py",
        'def unique_words(text):\n'
        '    """Return lowercase words once, in the order first seen."""\n'
        '    return list(dict.fromkeys(text.lower().split()))\n',
    ),
}


def scripted_replies(case_name: str) -> list[str]:
    """Prepare inspection, reproduction, edit, and verification answer cards."""
    try:
        source_path, test_path, replacement = _SCRIPTED_REPAIRS[case_name]
    except KeyError:
        raise ValueError(f"Unknown scripted case: {case_name}") from None
    actions = [
        {"tool": "read_file", "path": test_path, "summary": "Inspect the tests"},
        {"tool": "read_file", "path": source_path, "summary": "Inspect the source"},
        {"tool": "run_tests", "summary": "Reproduce the failure"},
        {"tool": "edit_file", "path": source_path, "content": replacement,
         "summary": "Fix the observed bug"},
        {"tool": "run_tests", "summary": "Verify the repair"},
    ]
    return [json.dumps(action) for action in actions]


class _TracingClient:
    """Forward chat calls while streaming public conversation observations."""

    def __init__(self, client: ChatClient, on_event: TraceSink) -> None:
        self.client = client
        self.on_event = on_event
        self.message_count: int | None = None
        self.decision = 0

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Publish the previous observation and each newly received reply."""
        if self.message_count is not None:
            for message in messages[self.message_count:]:
                if message["role"] != "user":
                    continue
                heading, output = message["content"].split("\n", 1)
                tool = heading.removeprefix("Tool result for ").removesuffix(":")
                self.on_event({
                    "event": "tool_result", "decision": self.decision,
                    "tool": tool, **json.loads(output),
                })
        self.message_count = len(messages)
        self.decision += 1
        raw = self.client.chat(messages)
        self.on_event({
            "event": "model_response", "decision": self.decision, "response": raw,
        })
        return raw


def build_service(provider: str = "scripted") -> RepairService:
    """Connect a fresh client and agent to real workspace tools for each repair."""
    if provider != "scripted":
        raise ValueError(f"Unsupported provider: {provider}")
    runner = DockerRunner()

    def make_agent(
        case_name: str, workspace: Path, on_event: TraceSink | None
    ) -> RepairAgent:
        tools = RepairTools(workspace, runner)
        client: ChatClient = ScriptedClient(scripted_replies(case_name))
        if on_event is not None:
            client = _TracingClient(client, on_event)
        return RepairAgent(client, tools)

    return RepairService(EVALS_ROOT, runner, make_agent, tuple(_SCRIPTED_REPAIRS))
