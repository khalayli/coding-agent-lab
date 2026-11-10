"""Control a bounded repair attempt through chat and file-tool contracts."""

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from core.models import ModelAction, RepairResult, TestResult
from core.ports import ChatClient, RepairToolsPort
from core.prompts import case_request, system_instruction, tool_result_message


DECISION_BUDGET = 8
_REQUIRED_FIELDS = {
    "read_file": {"tool", "path"},
    "run_tests": {"tool"},
    "edit_file": {"tool", "path", "content"},
}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while rejecting duplicate keys at every depth."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    """Reject non-JSON numeric constants accepted by Python's decoder."""
    raise ValueError(f"Invalid JSON constant: {value}")


def _parse_action(raw: str) -> ModelAction:
    """Accept exactly one JSON object matching one permitted tool schema."""
    if not isinstance(raw, str):
        raise ValueError("Model reply must be a JSON string")
    data = json.loads(
        raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant
    )
    if not isinstance(data, dict):
        raise ValueError("Model reply must contain exactly one JSON object")
    tool = data.get("tool")
    if not isinstance(tool, str) or tool not in _REQUIRED_FIELDS:
        raise ValueError("Missing or unknown tool")
    required = _REQUIRED_FIELDS[tool]
    missing = required - data.keys()
    extra = data.keys() - required - {"summary"}
    if missing:
        raise ValueError(f"Missing fields: {', '.join(sorted(missing))}")
    if extra:
        raise ValueError(f"Fields not allowed for {tool}: {', '.join(sorted(extra))}")
    for field, value in data.items():
        if not isinstance(value, str):
            raise ValueError(f"Field {field} must be a string")
    return ModelAction(
        tool=tool,
        arguments={key: value for key, value in data.items()
                   if key not in {"tool", "summary"}},
        summary=data.get("summary"),
    )


class RepairAgent:
    """Validate one proposed action at a time and enforce repair evidence."""

    def __init__(self, client: ChatClient, tools: RepairToolsPort) -> None:
        self.client = client
        self.tools = tools

    def run(
        self, case_name: str, relative_files: list[str], workspace: Path
    ) -> RepairResult:
        """Attempt a repair using at most eight calls to the chat client."""
        workspace = Path(workspace)
        messages = [
            {"role": "system", "content": system_instruction()},
            {"role": "user", "content": case_request(case_name, relative_files)},
        ]
        trace: list[dict[str, Any]] = [{
            "event": "repair_started",
            "case": case_name,
            "workspace": str(workspace),
            "decision_budget": DECISION_BUDGET,
        }]
        changed_files: list[str] = []
        last_test_result: TestResult | None = None
        has_read = False
        has_failure = False
        edited = False

        def finish(
            passed: bool, reason: str, message: str, decisions: int
        ) -> RepairResult:
            """Record the outcome and preserve all evidence collected so far."""
            trace.append({
                "event": "repair_finished", "passed": passed,
                "reason": reason, "message": message, "decisions": decisions,
            })
            return RepairResult(
                passed=passed, trace=trace, changed_files=changed_files,
                workspace=workspace, last_test_result=last_test_result,
            )

        def observe(
            action: ModelAction, observation: dict[str, Any], decision: int
        ) -> None:
            """Expose a tool result or rejection in both trace and history."""
            trace.append({
                "event": "tool_result", "decision": decision,
                "tool": action.tool, **observation,
            })
            messages.append({
                "role": "user",
                "content": tool_result_message(
                    action.tool, json.dumps(observation, ensure_ascii=False)
                ),
            })

        for decision in range(1, DECISION_BUDGET + 1):
            try:
                raw = self.client.chat(messages)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                trace.append({
                    "event": "model_response", "decision": decision,
                    "status": "client_error", "error": error,
                })
                return finish(False, "client_error", error, decision)

            response = {
                "event": "model_response", "decision": decision, "response": raw,
            }
            trace.append(response)
            try:
                action = _parse_action(raw)
            except (ValueError, TypeError, RecursionError) as exc:
                response["status"] = "invalid_response"
                return finish(False, "invalid_response", str(exc), decision)

            response["tool"] = action.tool
            if action.summary is not None:
                response["summary"] = action.summary
            messages.append({"role": "assistant", "content": raw})

            if action.tool == "edit_file" and not (has_read and has_failure):
                observe(action, {
                    "status": "policy_error",
                    "error": "edit_file requires a successful read_file and a "
                             "failing run_tests before editing.",
                }, decision)
                continue

            try:
                if action.tool == "read_file":
                    output = self.tools.read_file(action.arguments["path"])
                    observation = {"status": "ok", "output": output}
                    has_read = True
                elif action.tool == "edit_file":
                    self.tools.edit_file(
                        action.arguments["path"], action.arguments["content"]
                    )
                    path = Path(action.arguments["path"]).as_posix()
                    if path not in changed_files:
                        changed_files.append(path)
                    edited = True
                    observation = {"status": "ok", "path": path}
                else:
                    test_result = self.tools.run_tests()
                    observation = {"status": "ok", "result": asdict(test_result)}
                    last_test_result = test_result
                    if not test_result.passed:
                        has_failure = True
            except Exception as exc:
                observe(action, {
                    "status": "tool_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }, decision)
                continue

            observe(action, observation, decision)
            if (
                action.tool == "run_tests"
                and last_test_result is not None
                and last_test_result.passed
            ):
                if edited:
                    return finish(
                        True, "repair_verified", "Tests passed after the repair.",
                        decision,
                    )
                return finish(
                    False, "no_reproducible_bug",
                    "Tests passed without a successful edit; no reproducible bug "
                    "was demonstrated for repair.", decision,
                )

        return finish(
            False, "budget_exhausted",
            "The eight-decision budget was exhausted without a verified repair.",
            DECISION_BUDGET,
        )
