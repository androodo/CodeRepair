from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

import coderepair.tools as tools_module
from coderepair.tools import (
    MAX_READ_BYTES,
    ToolDispatcher,
    apply_structured_patch,
)


def replace_operation(path: str, old: str, new: str) -> dict[str, str]:
    return {"kind": "replace", "path": path, "old_text": old, "new_text": new}


def test_dispatch_rejects_unknown_tool_and_invalid_schema(tmp_path: Path) -> None:
    dispatcher = ToolDispatcher(tmp_path, enable_test_tool=False)

    unknown = dispatcher.execute("shell", {"command": "whoami"})
    invalid = dispatcher.execute("list_files", {"max_files": True})
    extra = dispatcher.execute("read_file", {"path": "x", "unexpected": 1})

    assert unknown.is_error and "unknown or unavailable" in unknown.output["error"]
    assert invalid.is_error and "must be an integer" in invalid.output["error"]
    assert extra.is_error and "unexpected fields" in extra.output["error"]
    assert "run_tests" not in {definition["name"] for definition in dispatcher.definitions}


def test_patch_is_all_or_nothing_when_one_python_file_has_invalid_ast(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("value = 1\n", encoding="utf-8")
    second.write_text("other = 2\n", encoding="utf-8")

    result = ToolDispatcher(tmp_path, enable_test_tool=False, protected_globs=()).execute(
        "apply_patch",
        {
            "operations": [
                replace_operation("first.py", "1", "10"),
                replace_operation("second.py", "other = 2", "def broken("),
            ]
        },
    )

    assert result.is_error
    assert "Python syntax error" in result.output["error"]
    assert first.read_text(encoding="utf-8") == "value = 1\n"
    assert second.read_text(encoding="utf-8") == "other = 2\n"


@pytest.mark.parametrize(("content", "old", "found"), [("x x\n", "x", 2), ("x\n", "y", 0)])
def test_patch_requires_an_exactly_once_match(
    tmp_path: Path, content: str, old: str, found: int
) -> None:
    path = tmp_path / "module.py"
    path.write_text(content, encoding="utf-8")

    result = ToolDispatcher(tmp_path, enable_test_tool=False, protected_globs=()).execute(
        "apply_patch", {"operations": [replace_operation("module.py", old, "z")]}
    )

    assert result.is_error
    assert f"found {found}" in result.output["error"]
    assert path.read_text(encoding="utf-8") == content


def test_default_policy_protects_visible_tests(tmp_path: Path) -> None:
    test_file = tmp_path / "tests" / "test_app.py"
    test_file.parent.mkdir()
    test_file.write_text("assert True\n", encoding="utf-8")

    result = ToolDispatcher(tmp_path, enable_test_tool=False).execute(
        "apply_patch",
        {"operations": [replace_operation("tests/test_app.py", "True", "False")]},
    )

    assert result.is_error
    assert "protected" in result.output["error"]
    assert test_file.read_text(encoding="utf-8") == "assert True\n"


def test_patch_rolls_back_a_partially_committed_multi_file_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("old-a", encoding="utf-8")
    second.write_text("old-b", encoding="utf-8")
    real_replace = os.replace
    calls = 0

    def fail_second_replace(source: str | Path, destination: str | Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated disk failure")
        real_replace(source, destination)

    monkeypatch.setattr(tools_module.os, "replace", fail_second_replace)

    with pytest.raises(tools_module.ToolError, match="atomic patch commit failed"):
        apply_structured_patch(
            tmp_path,
            [
                replace_operation("a.txt", "old-a", "new-a"),
                replace_operation("b.txt", "old-b", "new-b"),
            ],
            protected_globs=(),
            editable_globs=("**",),
        )

    assert first.read_text(encoding="utf-8") == "old-a"
    assert second.read_text(encoding="utf-8") == "old-b"
    assert not list(tmp_path.glob(".coderepair-*"))


def test_read_file_is_line_bounded_and_rejects_oversized_files(tmp_path: Path) -> None:
    lines = "".join(f"line {number}\n" for number in range(500))
    (tmp_path / "many.txt").write_text(lines, encoding="utf-8")
    (tmp_path / "huge.txt").write_bytes(b"a" * (MAX_READ_BYTES + 1))
    dispatcher = ToolDispatcher(tmp_path, enable_test_tool=False)

    bounded = dispatcher.execute("read_file", {"path": "many.txt"})
    too_many_lines = dispatcher.execute(
        "read_file", {"path": "many.txt", "start_line": 1, "end_line": 401}
    )
    oversized = dispatcher.execute("read_file", {"path": "huge.txt"})

    assert not bounded.is_error
    assert bounded.output["start_line"] == 1
    assert bounded.output["end_line"] == 200
    assert bounded.output["truncated"] is True
    assert too_many_lines.is_error and "limited to 400" in too_many_lines.output["error"]
    assert oversized.is_error and "byte read limit" in oversized.output["error"]


def test_listing_and_fallback_search_are_bounded_and_hide_sensitive_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "b.py").write_text("needle\nneedle\nneedle\n", encoding="utf-8")
    (tmp_path / "a.py").write_text("needle\nneedle\n", encoding="utf-8")
    (tmp_path / ".env").write_text("needle=secret\n", encoding="utf-8")
    ignored = tmp_path / "node_modules"
    ignored.mkdir()
    (ignored / "dependency.py").write_text("needle\n", encoding="utf-8")
    monkeypatch.setattr(tools_module.shutil, "which", lambda _name: None)
    dispatcher = ToolDispatcher(tmp_path, enable_test_tool=False)

    listing = dispatcher.execute("list_files", {"max_files": 20})
    search = dispatcher.execute("search_code", {"query": "needle", "max_results": 2})
    invalid_regex = dispatcher.execute("search_code", {"query": "[", "regex": True})

    assert listing.output["files"] == ["a.py", "b.py"]
    assert search.output["engine"] == "python-fallback"
    assert len(search.output["matches"]) == 2
    assert search.output["truncated"] is True
    assert all(match["path"] != ".env" for match in search.output["matches"])
    assert invalid_regex.is_error and "invalid regular expression" in invalid_regex.output["error"]


class PassingRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, Any]] = []

    def run_tests(self, workspace: Path, command: Any) -> dict[str, Any]:
        self.calls.append((workspace, command))
        return {"exit_code": 0, "passed": True, "stdout": "ok"}


def test_visible_pass_is_valid_only_for_latest_workspace_generation(
    tmp_path: Path,
) -> None:
    module = tmp_path / "module.py"
    module.write_text("value = 1\n", encoding="utf-8")
    runner = PassingRunner()
    approved = object()
    dispatcher = ToolDispatcher(
        tmp_path,
        test_runner=runner,
        approved_test_command=approved,
        protected_globs=(),
    )

    initial_test = dispatcher.execute("run_tests", {})
    assert not initial_test.is_error
    assert dispatcher.visible_tests_currently_passing

    first_patch = dispatcher.execute(
        "apply_patch",
        {"operations": [replace_operation("module.py", "1", "2")]},
    )
    assert first_patch.output["workspace_generation"] == 1
    assert not dispatcher.visible_tests_currently_passing

    retest = dispatcher.execute("run_tests", {})
    assert retest.output["workspace_generation"] == 1
    assert dispatcher.visible_tests_currently_passing
    dispatcher.execute(
        "apply_patch",
        {"operations": [replace_operation("module.py", "2", "3")]},
    )
    assert not dispatcher.visible_tests_currently_passing
    assert runner.calls == [(tmp_path.resolve(), approved), (tmp_path.resolve(), approved)]


def test_model_cannot_supply_or_replace_the_approved_test_command(tmp_path: Path) -> None:
    runner = PassingRunner()
    dispatcher = ToolDispatcher(
        tmp_path, test_runner=runner, approved_test_command=("pytest", "-q")
    )

    result = dispatcher.execute("run_tests", {"command": "pytest tests/easy"})

    assert result.is_error
    assert "unexpected fields" in result.output["error"]
    assert runner.calls == []
