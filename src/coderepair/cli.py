"""Command-line entry points for repair, evaluation, and approved Git application."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__
from .agent import Agent, AgentLimits
from .git_workflow import GitWorkflowError, apply_run_to_branch
from .model import AnthropicMessagesModel
from .reporting import Pricing, TraceWriter, default_output_directory, write_run_artifacts
from .sandbox import (
    DEFAULT_SANDBOX_IMAGE,
    DockerSandbox,
    SandboxLimits,
    approve_pytest_command,
)
from .tools import ToolDispatcher
from .workspace import WorkspaceError, create_workspace

DEFAULT_MODEL = "claude-sonnet-5-5"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coderepair",
        description="Bounded autonomous repair for clean Python Git repositories.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--debug", action="store_true", help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="attempt one repair in an isolated workspace")
    run_parser.add_argument("--repo", required=True, type=Path, help="clean Git repository root")
    run_parser.add_argument("--task", required=True, help="software-engineering task")
    run_parser.add_argument(
        "--test-command",
        default="pytest -q",
        help="trusted visible pytest invocation (default: pytest -q)",
    )
    run_parser.add_argument(
        "--model", default=os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL), help="Anthropic model ID"
    )
    run_parser.add_argument(
        "--test-image",
        default=os.getenv("CODEREPAIR_TEST_IMAGE", DEFAULT_SANDBOX_IMAGE),
        help="prebuilt local Docker image; it is never pulled automatically",
    )
    run_parser.add_argument("--output-dir", type=Path)
    run_parser.add_argument("--max-iterations", type=int, default=12)
    run_parser.add_argument("--max-tool-calls", type=int, default=40)
    run_parser.add_argument("--max-tokens", type=int, default=120_000)
    run_parser.add_argument("--max-output-tokens", type=int, default=8_192)
    run_parser.add_argument("--max-seconds", type=float, default=900.0)
    run_parser.add_argument("--test-timeout", type=float, default=120.0)
    run_parser.set_defaults(handler=_command_run)

    eval_parser = subparsers.add_parser("eval", help="run reproducible baseline/iterative tasks")
    eval_parser.add_argument("--manifest", required=True, type=Path)
    eval_parser.add_argument(
        "--mode", choices=("both", "baseline", "iterative"), default="both"
    )
    eval_parser.add_argument("--output-dir", type=Path, default=Path("benchmark-results"))
    eval_parser.add_argument(
        "--model", default=os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL), help="Anthropic model ID"
    )
    eval_parser.add_argument(
        "--test-image",
        default=os.getenv("CODEREPAIR_TEST_IMAGE", DEFAULT_SANDBOX_IMAGE),
    )
    eval_parser.set_defaults(handler=_command_eval)

    apply_parser = subparsers.add_parser(
        "apply", help="apply a reviewed run diff to a new branch after approval"
    )
    apply_parser.add_argument("--repo", required=True, type=Path)
    apply_parser.add_argument("--run-dir", required=True, type=Path)
    apply_parser.add_argument("--branch", required=True)
    apply_parser.add_argument("--commit-message", required=True)
    apply_parser.add_argument(
        "--approve",
        action="store_true",
        help="explicitly authorize modifying the original repository",
    )
    apply_parser.add_argument("--push", action="store_true")
    apply_parser.add_argument("--draft-pr", action="store_true")
    apply_parser.add_argument("--pr-title")
    apply_parser.add_argument("--pr-body")
    apply_parser.set_defaults(handler=_command_apply)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (ValueError, WorkspaceError, GitWorkflowError) as exc:
        print(f"coderepair: {exc}", file=sys.stderr)
        if args.debug:
            raise
        return 2
    except KeyboardInterrupt:
        print("coderepair: interrupted", file=sys.stderr)
        return 130


def _command_run(args: argparse.Namespace) -> int:
    repository = args.repo.resolve(strict=True)
    output = (args.output_dir or default_output_directory()).resolve()
    if _is_within(output, repository):
        raise ValueError("--output-dir must be outside the source repository")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    trace = TraceWriter(output / "trace.jsonl")
    approved_command = approve_pytest_command(args.test_command)
    limits = AgentLimits(
        max_iterations=args.max_iterations,
        max_tool_calls=args.max_tool_calls,
        max_total_tokens=args.max_tokens,
        max_seconds=args.max_seconds,
        max_output_tokens=args.max_output_tokens,
    )
    sandbox = DockerSandbox(
        image=args.test_image,
        limits=SandboxLimits(timeout_seconds=args.test_timeout),
    )
    pricing = Pricing.from_strings(
        os.getenv("CODEREPAIR_INPUT_COST_PER_MILLION"),
        os.getenv("CODEREPAIR_OUTPUT_COST_PER_MILLION"),
    )

    with create_workspace(repository) as workspace:
        model = AnthropicMessagesModel(args.model)
        dispatcher = ToolDispatcher(
            workspace.root,
            test_runner=sandbox,
            approved_test_command=approved_command,
            enable_test_tool=True,
        )
        outcome = Agent(model, limits=limits, event_sink=trace).run(
            task=args.task,
            tools=dispatcher,
        )
        result = write_run_artifacts(
            output,
            task=args.task,
            outcome=outcome,
            original_repository=repository,
            original_commit=workspace.original_commit,
            model=args.model,
            diff=workspace.diff(),
            modified_files=workspace.modified_files(),
            workspace_generation=dispatcher.generation,
            last_tested_generation=dispatcher.last_tested_generation,
            last_test_result=dispatcher.last_test_result,
            pricing=pricing,
        )

    visible = result["verification"]["visible_tests"]["status"]
    print(f"CodeRepair status: {result['status']}")
    print(f"Visible tests: {visible}")
    print(f"Modified files: {len(result['modified_files'])}")
    print(f"Artifacts: {output}")
    return 0 if visible == "passed" else 1


def _command_eval(args: argparse.Namespace) -> int:
    # Imported lazily so the simple run path does not pull evaluation machinery.
    from .evaluation import run_evaluation

    model_factory = lambda: AnthropicMessagesModel(args.model)  # noqa: E731
    report = run_evaluation(
        manifest_path=args.manifest,
        output_directory=args.output_dir,
        model_factory=model_factory,
        sandbox=DockerSandbox(image=args.test_image),
        modes=("baseline", "iterative") if args.mode == "both" else (args.mode,),
        model_name=args.model,
    )
    print(f"Evaluation tasks: {report['task_count']}")
    print(f"Result records: {report['run_count']}")
    print(f"Report: {Path(args.output_dir).resolve() / 'report.md'}")
    return 0 if report.get("infrastructure_failures", 0) == 0 else 1


def _command_apply(args: argparse.Namespace) -> int:
    patch_path = args.run_dir.resolve(strict=True) / "diff.patch"
    if not patch_path.is_file():
        raise GitWorkflowError(f"missing run diff: {patch_path}")
    patch = patch_path.read_text(encoding="utf-8")
    print("Proposed diff:\n")
    print(patch, end="" if patch.endswith("\n") else "\n")
    if not args.approve:
        print(
            "No changes made. Review the diff, then rerun with --approve to create the branch.",
            file=sys.stderr,
        )
        return 2
    result = apply_run_to_branch(
        repository=args.repo,
        run_directory=args.run_dir,
        branch=args.branch,
        commit_message=args.commit_message,
        approved=True,
        push=args.push,
        create_draft_pr=args.draft_pr,
        pr_title=args.pr_title,
        pr_body=args.pr_body,
    )
    print(f"Created commit {result.commit} on {result.branch}")
    if result.pushed:
        print("Pushed the new branch to origin.")
    if result.draft_pr_url:
        print(f"Draft pull request: {result.draft_pr_url}")
    return 0


def _is_within(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
        return True
    except ValueError:
        return False
