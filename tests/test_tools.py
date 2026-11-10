"""Check file boundaries and Docker delegation without executing eval code."""

from pathlib import Path
import subprocess

import pytest

from adapters.sandbox import runner as runner_module
from adapters.sandbox.repair_tools import RepairTools, safe_path
from adapters.sandbox.runner import DockerRunner
from core.models import TestResult


class StubRunner:
    """Return a prepared result without starting a process."""

    def __init__(self) -> None:
        self.calls: list[Path] = []
        self.result = TestResult(False, "A prepared test failure", 1)

    def run_tests(self, workspace: Path) -> TestResult:
        self.calls.append(workspace)
        return self.result


@pytest.fixture
def tools(tmp_path):
    """Provide file tools over a small synthetic workspace."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "note.txt").write_text("Hello, café!\n", encoding="utf-8")
    return RepairTools(workspace, StubRunner())


def test_read_normal_workspace_file(tools):
    assert tools.read_file("note.txt") == "Hello, café!\n"
    assert safe_path(tools.workspace, "note.txt") == tools.workspace / "note.txt"


@pytest.mark.parametrize("operation", ["read_file", "edit_file"])
@pytest.mark.parametrize("kind", ["absolute", "parent", "sibling", "symlink"])
def test_file_operations_reject_escapes(tools, tmp_path, operation, kind):
    outside = tmp_path / "outside.txt"
    outside.write_text("Private text", encoding="utf-8")
    sibling = tmp_path / "workspace-other"
    sibling.mkdir()
    (sibling / "outside.txt").write_text("Private text", encoding="utf-8")
    (tools.workspace / "link.txt").symlink_to(outside)
    paths = {
        "absolute": str(outside),
        "parent": "../outside.txt",
        "sibling": "../workspace-other/outside.txt",
        "symlink": "link.txt",
    }

    with pytest.raises(ValueError):
        if operation == "read_file":
            tools.read_file(paths[kind])
        else:
            tools.edit_file(paths[kind], "Changed")

    assert outside.read_text(encoding="utf-8") == "Private text"
    assert (sibling / "outside.txt").read_text(encoding="utf-8") == "Private text"
    assert tools.changed_files == []


def test_edit_saves_utf8_and_records_path_once(tools):
    tools.edit_file("note.txt", "First edit: café\n")
    tools.edit_file("./note.txt", "Second edit: café\n")

    assert (tools.workspace / "note.txt").read_text(encoding="utf-8") == (
        "Second edit: café\n"
    )
    assert tools.changed_files == ["note.txt"]


def test_unchanged_text_is_not_recorded(tools):
    tools.edit_file("note.txt", tools.read_file("note.txt"))
    assert tools.changed_files == []


def test_run_tests_delegates_without_starting_host_python(tools, monkeypatch):
    def unexpected_process(*args, **kwargs):
        pytest.fail("File tools must delegate instead of starting a process")

    monkeypatch.setattr(subprocess, "run", unexpected_process)

    assert tools.run_tests() is tools.runner.result
    assert tools.runner.calls == [tools.workspace]


@pytest.fixture
def sample_case(tmp_path, monkeypatch):
    """Use synthetic source files and a pretend home for copy tests."""
    home = tmp_path / "home"
    home.mkdir()
    evals = tmp_path / "evals"
    case = evals / "arithmetic"
    case.mkdir(parents=True)
    (case / "calculator.py").write_text("# Synthetic source\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(runner_module, "EVALS_ROOT", evals)
    return home, case


def test_copy_is_disposable_under_home_and_leaves_source_unchanged(sample_case):
    home, source = sample_case
    runner = DockerRunner()

    first = runner.copy_project("evals/arithmetic")
    second = runner.copy_project("evals/arithmetic")

    assert first != second
    assert first.parent == home / ".cache" / "coding-agent-lab"
    assert first.is_relative_to(home)
    assert first.stat().st_mode & 0o777 == 0o755
    copied = first / "calculator.py"
    assert copied.stat().st_mode & 0o777 == 0o644
    copied.write_text("# Edited copy\n", encoding="utf-8")
    assert (source / "calculator.py").read_text(encoding="utf-8") == (
        "# Synthetic source\n"
    )
    assert (second / "calculator.py").read_text(encoding="utf-8") == (
        "# Synthetic source\n"
    )


@pytest.mark.parametrize(
    "project", ["../arithmetic", "evals/../arithmetic", "evals/missing", "/etc"]
)
def test_copy_rejects_other_sources(sample_case, project):
    with pytest.raises(ValueError):
        DockerRunner().copy_project(project)


@pytest.mark.parametrize("kind", ["case", "file"])
def test_copy_rejects_source_symlinks(sample_case, tmp_path, kind):
    _, case = sample_case
    if kind == "case":
        (case.parent / "linked").symlink_to(case, target_is_directory=True)
        project = "evals/linked"
    else:
        outside = tmp_path / "outside.txt"
        outside.write_text("Private text", encoding="utf-8")
        (case / "link.txt").symlink_to(outside)
        project = "evals/arithmetic"

    with pytest.raises(ValueError, match="symbolic links|direct child"):
        DockerRunner().copy_project(project)


def test_copy_rejects_cache_resolving_outside_home(sample_case, tmp_path):
    home, _ = sample_case
    cache = home / ".cache"
    cache.mkdir()
    (cache / "coding-agent-lab").symlink_to(tmp_path, target_is_directory=True)

    with pytest.raises(ValueError, match="home directory"):
        DockerRunner().copy_project("evals/arithmetic")


@pytest.mark.parametrize("returncode", [0, 1, 5, 124, 125])
def test_runner_uses_only_docker_with_one_readonly_mount(
    sample_case, monkeypatch, returncode
):
    runner = DockerRunner()
    workspace = runner.copy_project("evals/arithmetic")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        assert kwargs["stdout"] == subprocess.PIPE
        assert kwargs["stderr"] == subprocess.STDOUT
        assert kwargs["check"] is False
        assert kwargs["text"] is True
        return subprocess.CompletedProcess(command, returncode, "Captured output")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert runner.run_tests(workspace) == TestResult(
        returncode == 0, "Captured output", returncode
    )
    assert len(calls) == 1
    command = calls[0]
    assert command[:3] == ["docker", "run", "--rm"]
    assert "--pull=never" in command
    assert command[command.index("--network") + 1] == "none"
    assert command.count("--mount") == 1
    assert command[command.index("--mount") + 1] == (
        f"type=bind,source={workspace},target=/workspace,readonly"
    )
    assert command[command.index("--workdir") + 1] == "/workspace"
    assert "--volume" not in command and "-v" not in command
    assert command[-7:] == [
        runner.image, "python", "-m", "pytest", "-q", "-p", "no:cacheprovider"
    ]


def test_runner_reports_missing_docker_without_host_fallback(sample_case, monkeypatch):
    runner = DockerRunner()
    workspace = runner.copy_project("evals/arithmetic")

    def missing_docker(*args, **kwargs):
        raise FileNotFoundError("docker executable not found")

    monkeypatch.setattr(subprocess, "run", missing_docker)

    result = runner.run_tests(workspace)
    assert not result.passed
    assert result.returncode == 127
    assert "docker executable not found" in result.output


def test_runner_rejects_mounting_an_arbitrary_directory(sample_case, monkeypatch):
    _, source = sample_case

    def unexpected_process(*args, **kwargs):
        pytest.fail("Invalid workspaces must be rejected before starting Docker")

    monkeypatch.setattr(subprocess, "run", unexpected_process)

    with pytest.raises(ValueError, match="copied home-cache workspace"):
        DockerRunner().run_tests(source)
