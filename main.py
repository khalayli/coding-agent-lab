"""Run a selected evaluation-project repair from the command line."""

import argparse
import json
import sys

from bootstrap import build_service


def _print_event(event: dict) -> None:
    """Print each public event immediately as one JSON line."""
    print(json.dumps(event, ensure_ascii=False), flush=True)


def main(argv: list[str] | None = None) -> int:
    """Parse the repair command and report its trace and final outcome."""
    parser = argparse.ArgumentParser(description="Repair a supplied Python project")
    commands = parser.add_subparsers(dest="command", required=True)
    repair = commands.add_parser("repair", help="Repair one project under evals/")
    repair.add_argument("project", help="Selected project, such as evals/arithmetic")
    repair.add_argument("--provider", choices=("scripted",), default="scripted")
    args = parser.parse_args(argv)

    try:
        service = build_service(args.provider)
        result = service.repair(args.project, on_event=_print_event)
    except (ValueError, OSError) as exc:
        print(f"Repair could not start: {exc}", file=sys.stderr, flush=True)
        return 1

    print(f"Repair outcome: {'passed' if result.passed else 'failed'}")
    print(f"Workspace: {result.workspace}")
    print("Changed relative files:")
    for path in result.changed_files:
        print(f"- {path}")
    if not result.changed_files:
        print("(none)")
    print("Final sandboxed test result:")
    test_result = result.last_test_result
    if test_result is None:
        print("No test result was produced.")
    else:
        print(f"passed={test_result.passed} returncode={test_result.returncode}")
        print(test_result.output, end="" if test_result.output.endswith("\n") else "\n")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
