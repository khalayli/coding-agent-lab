"""Shared data containers for repair actions and results."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TestResult:
    """Outcome and captured output of a test run."""

    __test__ = False

    passed: bool
    output: str
    returncode: int


@dataclass
class ModelAction:
    """A requested tool call with an optional short public summary."""

    tool: str
    arguments: dict[str, Any]
    summary: str | None = None


@dataclass
class RepairResult:
    """Repair outcome, public trace, changed paths, and latest test result."""

    passed: bool
    trace: list[dict[str, Any]]
    changed_files: list[str]
    workspace: Path
    last_test_result: TestResult | None = None
