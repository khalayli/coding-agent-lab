"""Select supplied evaluation projects and run repairs on disposable copies."""

from collections.abc import Callable, Collection
from pathlib import Path
from typing import Any, Protocol

from core.models import RepairResult
from core.repair_agent import DECISION_BUDGET, RepairAgent


TraceSink = Callable[[dict[str, Any]], None]
AgentFactory = Callable[[str, Path, TraceSink | None], RepairAgent]


class WorkspaceRunner(Protocol):
    """Describe the workspace-copy operation needed by the service."""

    def copy_project(self, project: str | Path) -> Path:
        """Return a disposable copy of a selected evaluation project."""
        ...


class RepairService:
    """Validate a selection, prepare its workspace, and invoke a fresh agent."""

    def __init__(
        self,
        evals_root: Path,
        runner: WorkspaceRunner,
        agent_factory: AgentFactory,
        cases: Collection[str],
    ) -> None:
        self.evals_root = Path(evals_root).resolve()
        self.runner = runner
        self.agent_factory = agent_factory
        self.cases = tuple(cases)

    def repair(
        self, project: str | Path, on_event: TraceSink | None = None
    ) -> RepairResult:
        """Repair one known evals/<case>, optionally streaming public events."""
        requested = Path(project)
        if (
            requested.is_absolute()
            or len(requested.parts) != 2
            or requested.parts[0] != "evals"
        ):
            raise ValueError("Select a relative project path of the form evals/<case>")
        if requested.name not in self.cases:
            raise ValueError(f"Unknown evaluation project: {requested.name}")
        source = self.evals_root / requested.name
        if (
            source.is_symlink()
            or source.resolve().parent != self.evals_root
            or not source.is_dir()
        ):
            raise ValueError("Project must be an existing direct child of evals/")

        workspace = self.runner.copy_project(requested.as_posix())
        relative_files = sorted(
            path.relative_to(workspace).as_posix()
            for path in workspace.rglob("*") if path.is_file()
        )
        emitted = 0

        def emit(event: dict[str, Any]) -> None:
            """Count streamed events so the final trace is not printed twice."""
            nonlocal emitted
            if on_event is not None:
                on_event(event)
                emitted += 1

        agent = self.agent_factory(
            requested.name, workspace, emit if on_event is not None else None
        )
        if on_event is not None:
            emit({
                "event": "repair_started", "case": requested.name,
                "workspace": str(workspace), "decision_budget": DECISION_BUDGET,
            })
        result = agent.run(requested.name, relative_files, workspace)
        if on_event is not None:
            # The final observation and finish event have no following chat call.
            for event in result.trace[emitted:]:
                emit(event)
        return result
