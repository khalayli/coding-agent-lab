"""Copy supplied cases and execute their tests only inside Docker."""

from pathlib import Path
import shutil
import subprocess
import tempfile

from core.models import TestResult


EVALS_ROOT = Path(__file__).resolve().parents[2] / "evals"
RUNNER_IMAGE = "coding-agent-lab-runner"


def _workspace_root() -> Path:
    """Return a cache directory that resolves inside the user's home."""
    home = Path.home().resolve()
    root = (home / ".cache" / "coding-agent-lab").resolve()
    if not root.is_relative_to(home) or root == home:
        raise ValueError("Workspace cache must remain inside the home directory")
    return root


class DockerRunner:
    """Prepare disposable case copies and collect sandboxed test results."""

    def __init__(self, image: str = RUNNER_IMAGE) -> None:
        self.image = image

    def copy_project(self, project: str | Path) -> Path:
        """Copy an existing evals/<case> into a unique home-cache workspace."""
        requested = Path(project)
        if (
            requested.is_absolute()
            or len(requested.parts) != 2
            or requested.parts[0] != "evals"
        ):
            raise ValueError("Select a project using evals/<case>")

        evals_root = EVALS_ROOT.resolve()
        source = evals_root / requested.name
        if (
            source.is_symlink()
            or source.resolve().parent != evals_root
            or not source.is_dir()
        ):
            raise ValueError("Project must be an existing direct child of evals/")
        if any(path.is_symlink() for path in source.rglob("*")):
            raise ValueError("Evaluation projects must not contain symbolic links")

        root = _workspace_root()
        root.mkdir(parents=True, exist_ok=True)
        workspace = Path(tempfile.mkdtemp(prefix=f"{source.name}-", dir=root))
        try:
            shutil.copytree(source, workspace, dirs_exist_ok=True)
            # mkdtemp uses mode 0700; the container's non-root user needs access.
            workspace.chmod(0o755)
            for path in workspace.rglob("*"):
                path.chmod(0o755 if path.is_dir() else 0o644)
        except Exception:
            shutil.rmtree(workspace)
            raise
        return workspace

    def run_tests(self, workspace: Path) -> TestResult:
        """Run pytest in a disposable, network-disabled Docker container."""
        workspace = Path(workspace).resolve()
        if workspace.parent != _workspace_root() or not workspace.is_dir():
            raise ValueError("Tests require a copied home-cache workspace")

        command = [
            "docker", "run", "--rm", "--pull=never",
            "--network", "none",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--mount", f"type=bind,source={workspace},target=/workspace,readonly",
            "--workdir", "/workspace",
            self.image,
            "python", "-m", "pytest", "-q", "-p", "no:cacheprovider",
        ]
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except OSError as exc:
            return TestResult(False, f"Could not start Docker: {exc}", 127)
        return TestResult(
            passed=completed.returncode == 0,
            output=completed.stdout,
            returncode=completed.returncode,
        )
