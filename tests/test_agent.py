"""Exercise repair decisions with scripted replies and in-memory tools."""

from dataclasses import asdict
import json
from pathlib import Path

import pytest

from adapters.models.scripted import ScriptedClient
from core.models import RepairResult, TestResult
from core.prompts import case_request, system_instruction, tool_result_message
from core.repair_agent import RepairAgent


READ = '{"tool":"read_file","path":"calculator.py","summary":"Inspect code"}'
RUN = '{"tool":"run_tests"}'
EDIT = '{"tool":"edit_file","path":"calculator.py","content":"corrected source"}'
FAILED = TestResult(False, "An assertion failed", 1)
PASSED = TestResult(True, "Tests passed", 0)
FILES = ["calculator.py", "test_calculator.py"]


class RecordingClient(ScriptedClient):
    """Keep conversation snapshots while returning scripted answer cards."""

    def __init__(self, replies: list[str]) -> None:
        super().__init__(replies)
        self.histories: list[list[dict[str, str]]] = []
        self.last_messages: list[dict[str, str]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.histories.append([message.copy() for message in messages])
        self.last_messages = messages
        return super().chat(messages)


class FakeTools:
    """Read and edit synthetic text and return prepared test observations."""

    def __init__(self, results=()) -> None:
        self.files = {"calculator.py": "original source"}
        self.results = iter(results)
        self.calls: list[tuple[str, dict[str, str]]] = []

    def read_file(self, path: str) -> str:
        self.calls.append(("read_file", {"path": path}))
        normalized = Path(path).as_posix()
        if normalized not in self.files:
            raise FileNotFoundError(path)
        return self.files[normalized]

    def edit_file(self, path: str, content: str) -> None:
        self.calls.append(("edit_file", {"path": path, "content": content}))
        normalized = Path(path).as_posix()
        if normalized not in self.files:
            raise FileNotFoundError(path)
        self.files[normalized] = content

    def run_tests(self) -> TestResult:
        self.calls.append(("run_tests", {}))
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        return result


def test_successful_repair_records_history_trace_and_results(tmp_path):
    replies = [READ, RUN, EDIT, RUN]
    client = RecordingClient(replies)
    tools = FakeTools([FAILED, PASSED])

    result = RepairAgent(client, tools).run("arithmetic", FILES, tmp_path)

    assert result.passed
    assert result.changed_files == ["calculator.py"]
    assert result.workspace == tmp_path
    assert result.last_test_result is PASSED
    assert tools.files["calculator.py"] == "corrected source"
    assert [tool for tool, _ in tools.calls] == [
        "read_file", "run_tests", "edit_file", "run_tests"
    ]
    assert client.histories[0] == [
        {"role": "system", "content": system_instruction()},
        {"role": "user", "content": case_request("arithmetic", FILES)},
    ]
    assert [message["content"] for message in client.last_messages
            if message["role"] == "assistant"] == replies
    assert [message["role"] for message in client.last_messages] == (
        ["system", "user"] + ["assistant", "user"] * 4
    )
    assert client.histories[2][-1]["content"] == tool_result_message(
        "run_tests", json.dumps({"status": "ok", "result": asdict(FAILED)})
    )
    assert client.last_messages[-1]["content"] == tool_result_message(
        "run_tests", json.dumps({"status": "ok", "result": asdict(PASSED)})
    )
    assert [entry["event"] for entry in result.trace] == (
        ["repair_started"] + ["model_response", "tool_result"] * 4
        + ["repair_finished"]
    )
    assert result.trace[1]["summary"] == "Inspect code"
    assert result.trace[-1]["reason"] == "repair_verified"


@pytest.mark.parametrize("reply", [
    "Here is my next action: " + RUN,
    "```json\n" + RUN + "\n```",
    "[]", "null", "true", '"run_tests"', "42",
    '{"tool":',
    RUN + " " + RUN,
    RUN + " trailing prose",
    '{"tool":"run_tests","tool":"run_tests"}',
    '{"tool":"read_file","path":"a","path":"b"}',
    '{"tool":"run_tests","payload":{"key":1,"key":2}}',
    '{"tool":"shell","command":"anything"}',
    '{}',
    '{"tool":"read_file"}',
    '{"tool":"edit_file","path":"calculator.py"}',
    '{"tool":"edit_file","content":"replacement"}',
    '{"tool":"run_tests","path":"calculator.py"}',
    '{"tool":"read_file","path":"calculator.py","content":"extra"}',
    '{"tool":"run_tests","arguments":{}}',
    '{"tool":"run_tests","reasoning":"extra field"}',
    '{"tool":"run_tests","summary":null}',
    '{"tool":"run_tests","summary":12}',
    '{"tool":"read_file","path":null}',
    '{"tool":"edit_file","path":"calculator.py","content":{}}',
    '{"tool":["run_tests"]}',
    '{"tool":"read_file","path":NaN}',
    '{"tool":"read_file","path":Infinity}',
])
def test_invalid_reply_stops_without_calling_a_tool(tmp_path, reply):
    client = RecordingClient([reply, READ])
    tools = FakeTools()

    result = RepairAgent(client, tools).run("arithmetic", FILES, tmp_path)

    assert isinstance(result, RepairResult)
    assert not result.passed
    assert result.trace[-1]["reason"] == "invalid_response"
    assert tools.calls == []
    assert len(client.histories) == 1
    assert all(message["role"] != "assistant" for message in client.last_messages)


@pytest.mark.parametrize("replies", [
    [EDIT, READ, RUN, EDIT, RUN],
    [READ, EDIT, RUN, EDIT, RUN],
    [RUN, EDIT, READ, EDIT, RUN],
])
def test_early_edit_is_visible_and_skipped_until_both_observations(tmp_path, replies):
    client = RecordingClient(replies)
    tools = FakeTools([FAILED, PASSED])

    result = RepairAgent(client, tools).run("arithmetic", FILES, tmp_path)

    assert result.passed
    assert sum(tool == "edit_file" for tool, _ in tools.calls) == 1
    policy_errors = [entry for entry in result.trace
                     if entry.get("status") == "policy_error"]
    assert len(policy_errors) == 1
    assert "successful read_file" in policy_errors[0]["error"]
    assert "failing run_tests" in policy_errors[0]["error"]
    assert any('"status": "policy_error"' in message["content"]
               for message in client.last_messages if message["role"] == "user")
    assert [message["content"] for message in client.last_messages
            if message["role"] == "assistant"] == replies


def test_budget_exhaustion_stops_after_exactly_eight_decisions(tmp_path):
    client = RecordingClient([READ] * 9)
    tools = FakeTools()

    result = RepairAgent(client, tools).run("arithmetic", FILES, tmp_path)

    assert not result.passed
    assert result.trace[-1]["reason"] == "budget_exhausted"
    assert result.trace[-1]["decisions"] == 8
    assert len(client.histories) == len(tools.calls) == 8
    assert result.last_test_result is None


def test_repair_can_be_verified_on_eighth_decision(tmp_path):
    client = RecordingClient([READ] * 5 + [RUN, EDIT, RUN])
    tools = FakeTools([FAILED, PASSED])

    result = RepairAgent(client, tools).run("arithmetic", FILES, tmp_path)

    assert result.passed
    assert len(client.histories) == 8


def test_passing_baseline_is_not_reported_as_a_repair(tmp_path):
    client = RecordingClient([RUN, EDIT])
    tools = FakeTools([PASSED])

    result = RepairAgent(client, tools).run("arithmetic", FILES, tmp_path)

    assert not result.passed
    assert result.last_test_result is PASSED
    assert result.changed_files == []
    assert result.trace[-1]["reason"] == "no_reproducible_bug"
    assert "no reproducible bug" in result.trace[-1]["message"]
    assert len(client.histories) == 1
    assert tools.calls == [("run_tests", {})]


@pytest.mark.parametrize("replies", [[], [READ, RUN, EDIT]])
def test_client_error_still_returns_a_result_with_collected_evidence(tmp_path, replies):
    client = RecordingClient(replies)
    tools = FakeTools([FAILED])

    result = RepairAgent(client, tools).run("arithmetic", FILES, tmp_path)

    assert isinstance(result, RepairResult)
    assert not result.passed
    assert result.workspace == tmp_path
    assert result.trace[-1]["reason"] == "client_error"
    assert "no replies remaining" in result.trace[-1]["message"]
    assert result.trace[-2]["status"] == "client_error"
    assert result.changed_files == (["calculator.py"] if replies else [])
    assert result.last_test_result is (FAILED if replies else None)


def test_failed_read_does_not_unlock_edit_and_model_can_recover(tmp_path):
    missing = '{"tool":"read_file","path":"missing.py"}'
    client = RecordingClient([RUN, missing, EDIT, READ, EDIT, RUN])
    tools = FakeTools([FAILED, PASSED])

    result = RepairAgent(client, tools).run("arithmetic", FILES, tmp_path)

    assert result.passed
    assert sum(tool == "edit_file" for tool, _ in tools.calls) == 1
    assert any(entry.get("status") == "tool_error" for entry in result.trace)
    assert any(entry.get("status") == "policy_error" for entry in result.trace)
    assert any("FileNotFoundError" in message["content"]
               for message in client.last_messages if message["role"] == "user")


def test_test_runner_error_does_not_count_as_a_failing_test(tmp_path):
    client = RecordingClient([READ, RUN, EDIT, RUN, EDIT, RUN])
    tools = FakeTools([RuntimeError("runner unavailable"), FAILED, PASSED])

    result = RepairAgent(client, tools).run("arithmetic", FILES, tmp_path)

    assert result.passed
    assert sum(tool == "edit_file" for tool, _ in tools.calls) == 1
    assert any(entry.get("status") == "policy_error" for entry in result.trace)
    assert any("runner unavailable" in message["content"]
               for message in client.last_messages if message["role"] == "user")


def test_failed_edit_is_not_counted_as_a_repair(tmp_path):
    missing_edit = '{"tool":"edit_file","path":"missing.py","content":"new"}'
    client = RecordingClient([READ, RUN, missing_edit, RUN])
    tools = FakeTools([FAILED, PASSED])

    result = RepairAgent(client, tools).run("arithmetic", FILES, tmp_path)

    assert not result.passed
    assert result.changed_files == []
    assert result.last_test_result is PASSED
    assert any(entry.get("status") == "tool_error" for entry in result.trace)


def test_repeated_edits_record_each_relative_path_once(tmp_path):
    second_edit = json.dumps({
        "tool": "edit_file", "path": "./calculator.py", "content": "second edit",
        "summary": "Refine the edit",
    })
    client = RecordingClient([READ, RUN, EDIT, second_edit, RUN])
    tools = FakeTools([FAILED, PASSED])

    result = RepairAgent(client, tools).run("arithmetic", FILES, tmp_path)

    assert result.passed
    assert result.changed_files == ["calculator.py"]
    assert tools.files["calculator.py"] == "second edit"


def test_evidence_is_reset_between_attempts(tmp_path):
    client = RecordingClient([READ, RUN, EDIT, RUN, EDIT, RUN])
    tools = FakeTools([FAILED, PASSED, PASSED])
    agent = RepairAgent(client, tools)

    first = agent.run("first", FILES, tmp_path)
    second = agent.run("second", FILES, tmp_path)

    assert first.passed
    assert not second.passed
    assert second.changed_files == []
    assert any(entry.get("status") == "policy_error" for entry in second.trace)
    assert sum(tool == "edit_file" for tool, _ in tools.calls) == 1
