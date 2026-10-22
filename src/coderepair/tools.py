"""Repository tool schemas, validation, and security-enforcing dispatch."""

from __future__ import annotations

import ast
import fnmatch
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import threading
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from .agent import ToolExecution
from .model import JsonObject
from .paths import PathSecurityError, is_sensitive_path, validate_relative_path
from .sandbox import SandboxError

IGNORED_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "node_modules",
    "venv",
}
DEFAULT_PROTECTED_GLOBS = (
    "tests/**",
    "**/tests/**",
    "test_*.py",
    "**/test_*.py",
    "*_test.py",
    "**/*_test.py",
    "conftest.py",
    "**/conftest.py",
    "pytest.ini",
    "pyproject.toml",
    "setup.cfg",
    "tox.ini",
)
MAX_READ_BYTES = 1_000_000
MAX_READ_LINES = 400
MAX_RESULT_CHARS = 50_000
MAX_PATCH_BYTES = 500_000


class ToolError(RuntimeError):
    """A safe error suitable for returning as a model observation."""


class TestRunner(Protocol):
    def run_tests(self, workspace: Path, command: Any) -> Any: ...


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: JsonObject

    def definition(self) -> JsonObject:
        return {
            "name": self.name,
            "description": self.description,
            "strict": True,
            "input_schema": self.input_schema,
        }


def _object_schema(
    properties: Mapping[str, JsonObject], required: Sequence[str] = ()
) -> JsonObject:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": False,
    }


TOOL_SPECS = (
    ToolSpec(
        "list_files",
        "List bounded, non-sensitive repository files in deterministic order.",
        _object_schema(
            {"max_files": {"type": "integer", "minimum": 1, "maximum": 500}}
        ),
    ),
    ToolSpec(
        "search_code",
        "Search bounded repository text with ripgrep (literal by default).",
        _object_schema(
            {
                "query": {"type": "string", "minLength": 1, "maxLength": 500},
                "path_glob": {"type": "string", "minLength": 1, "maxLength": 200},
                "regex": {"type": "boolean"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 200},
            },
            ("query",),
        ),
    ),
    ToolSpec(
        "read_file",
        "Read a bounded UTF-8 line range from one non-sensitive repository file.",
        _object_schema(
            {
                "path": {"type": "string", "minLength": 1, "maxLength": 500},
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
            },
            ("path",),
        ),
    ),
    ToolSpec(
        "apply_patch",
        (
            "Atomically apply structured exact-text operations. For replace, old_text must "
            "occur exactly once. Create and delete are explicit. Modified Python is AST-checked."
        ),
        _object_schema(
            {
                "operations": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 20,
                    "items": _object_schema(
                        {
                            "kind": {"type": "string", "enum": ["replace", "create", "delete"]},
                            "path": {"type": "string", "minLength": 1, "maxLength": 500},
                            "old_text": {"type": "string", "maxLength": MAX_PATCH_BYTES},
                            "new_text": {"type": "string", "maxLength": MAX_PATCH_BYTES},
                        },
                        ("kind", "path", "old_text", "new_text"),
                    ),
                }
            },
            ("operations",),
        ),
    ),
)

RUN_TESTS_SPEC = ToolSpec(
    "run_tests",
    "Run the single human-approved pytest command in the restricted Docker sandbox.",
    _object_schema({}),
)


class ToolDispatcher:
    """Allowlisted dispatcher bound to one temporary workspace."""

    def __init__(
        self,
        workspace_root: Path,
        *,
        test_runner: TestRunner | None = None,
        approved_test_command: Any | None = None,
        enable_test_tool: bool = True,
        protected_globs: Sequence[str] = DEFAULT_PROTECTED_GLOBS,
        editable_globs: Sequence[str] = ("**",),
    ) -> None:
        self.root = workspace_root.resolve(strict=True)
        self.test_runner = test_runner
        self.approved_test_command = approved_test_command
        self.enable_test_tool = enable_test_tool
        self.protected_globs = tuple(protected_globs)
        self.editable_globs = tuple(editable_globs)
        self.generation = 0
        self.last_tested_generation = -1
        self.last_test_passed = False
        self.last_test_result: JsonObject | None = None
        specs = list(TOOL_SPECS)
        if enable_test_tool:
            specs.append(RUN_TESTS_SPEC)
        self._specs = {spec.name: spec for spec in specs}

    @property
    def definitions(self) -> Sequence[JsonObject]:
        return tuple(spec.definition() for spec in self._specs.values())

    @property
    def visible_tests_currently_passing(self) -> bool:
        return self.last_test_passed and self.last_tested_generation == self.generation

    def execute(self, name: str, arguments: Any) -> ToolExecution:
        spec = self._specs.get(name)
        if spec is None:
            return ToolExecution(
                {"ok": False, "error": f"unknown or unavailable tool: {name}"}, is_error=True
            )
        try:
            validated = validate_schema(arguments, spec.input_schema)
            handler = getattr(self, f"_tool_{name}")
            output = handler(validated)
            output.setdefault("ok", True)
            return ToolExecution(output)
        except (ToolError, PathSecurityError, SandboxError, OSError, UnicodeError) as exc:
            return ToolExecution(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, is_error=True
            )

    def _tool_list_files(self, arguments: JsonObject) -> JsonObject:
        maximum = arguments.get("max_files", 200)
        files, truncated = list_repository_files(self.root, maximum)
        return {"files": files, "count": len(files), "truncated": truncated}

    def _tool_search_code(self, arguments: JsonObject) -> JsonObject:
        path_glob = arguments.get("path_glob")
        if path_glob is not None:
            _validate_glob(path_glob)
        results, engine, truncated = search_repository(
            self.root,
            arguments["query"],
            regex=arguments.get("regex", False),
            path_glob=path_glob,
            max_results=arguments.get("max_results", 100),
        )
        return {
            "matches": results,
            "count": len(results),
            "engine": engine,
            "truncated": truncated,
        }

    def _tool_read_file(self, arguments: JsonObject) -> JsonObject:
        start = arguments.get("start_line", 1)
        end = arguments.get("end_line", start + 199)
        if end < start:
            raise ToolError("end_line must be greater than or equal to start_line")
        if end - start + 1 > MAX_READ_LINES:
            raise ToolError(f"a read is limited to {MAX_READ_LINES} lines")
        path = validate_relative_path(
            self.root,
            arguments["path"],
            must_exist=True,
            allow_directory=False,
            reject_sensitive=True,
        )
        if path.stat().st_size > MAX_READ_BYTES:
            raise ToolError(f"file exceeds the {MAX_READ_BYTES}-byte read limit")
        with path.open("rb") as handle:
            raw = handle.read(MAX_READ_BYTES + 1)
        if len(raw) > MAX_READ_BYTES:
            raise ToolError(f"file exceeds the {MAX_READ_BYTES}-byte read limit")
        if b"\x00" in raw:
            raise ToolError("binary files cannot be read")
        text = raw.decode("utf-8")
        lines = text.splitlines()
        selected = lines[start - 1 : end]
        rendered = "\n".join(
            f"{number:>6} | {line[:2_000]}"
            for number, line in enumerate(selected, start=start)
        )
        rendered, chars_truncated = _truncate(rendered, MAX_RESULT_CHARS)
        return {
            "path": arguments["path"],
            "start_line": start,
            "end_line": start + len(selected) - 1 if selected else start - 1,
            "total_lines": len(lines),
            "content": rendered,
            "truncated": end < len(lines) or chars_truncated,
        }

    def _tool_apply_patch(self, arguments: JsonObject) -> JsonObject:
        operations = arguments["operations"]
        encoded_size = len(json.dumps(operations, ensure_ascii=False).encode("utf-8"))
        if encoded_size > MAX_PATCH_BYTES:
            raise ToolError(f"patch exceeds the {MAX_PATCH_BYTES}-byte limit")
        changed = apply_structured_patch(
            self.root,
            operations,
            protected_globs=self.protected_globs,
            editable_globs=self.editable_globs,
        )
        if changed:
            self.generation += 1
        return {
            "changed_files": list(changed),
            "workspace_generation": self.generation,
            "python_syntax_valid": True,
        }

    def _tool_run_tests(self, arguments: JsonObject) -> JsonObject:
        del arguments
        if self.test_runner is None or self.approved_test_command is None:
            raise ToolError("test execution was not configured by the user")
        result = self.test_runner.run_tests(self.root, self.approved_test_command)
        data = _result_to_dict(result)
        passed = bool(data.get("passed", data.get("exit_code") == 0))
        self.last_tested_generation = self.generation
        self.last_test_passed = passed
        data["workspace_generation"] = self.generation
        self.last_test_result = dict(data)
        return data


def validate_schema(value: Any, schema: Mapping[str, Any], path: str = "arguments") -> Any:
    """Validate the JSON-Schema subset used by the five built-in tools."""

    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, dict):
            raise ToolError(f"{path} must be an object")
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        missing = [name for name in required if name not in value]
        if missing:
            raise ToolError(f"{path} is missing required fields: {', '.join(missing)}")
        extras = sorted(set(value) - set(properties))
        if extras and schema.get("additionalProperties") is False:
            raise ToolError(f"{path} has unexpected fields: {', '.join(extras)}")
        return {
            key: validate_schema(item, properties[key], f"{path}.{key}")
            for key, item in value.items()
            if key in properties
        }
    if expected == "array":
        if not isinstance(value, list):
            raise ToolError(f"{path} must be an array")
        _check_length(value, schema, path)
        return [
            validate_schema(item, schema["items"], f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if expected == "string":
        if not isinstance(value, str):
            raise ToolError(f"{path} must be a string")
        _check_length(value, schema, path)
        if "enum" in schema and value not in schema["enum"]:
            raise ToolError(f"{path} must be one of: {', '.join(schema['enum'])}")
        return value
    if expected == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ToolError(f"{path} must be an integer")
        if value < schema.get("minimum", value):
            raise ToolError(f"{path} is below its minimum")
        if value > schema.get("maximum", value):
            raise ToolError(f"{path} is above its maximum")
        return value
    if expected == "boolean":
        if not isinstance(value, bool):
            raise ToolError(f"{path} must be a boolean")
        return value
    raise ToolError(f"unsupported schema type at {path}: {expected}")


def list_repository_files(root: Path, maximum: int) -> tuple[list[str], bool]:
    files: list[str] = []
    truncated = False
    for current, directories, names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        safe_directories: list[str] = []
        for name in sorted(directories):
            candidate = current_path / name
            relative = candidate.relative_to(root).as_posix()
            if name in IGNORED_DIRECTORIES or candidate.is_symlink() or is_sensitive_path(relative):
                continue
            safe_directories.append(name)
        directories[:] = safe_directories
        for name in sorted(names):
            candidate = current_path / name
            relative = candidate.relative_to(root).as_posix()
            if candidate.is_symlink() or not candidate.is_file() or is_sensitive_path(relative):
                continue
            files.append(relative)
            if len(files) >= maximum:
                truncated = True
                return files, truncated
    return files, truncated


def search_repository(
    root: Path,
    query: str,
    *,
    regex: bool,
    path_glob: str | None,
    max_results: int,
) -> tuple[list[JsonObject], str, bool]:
    rg = shutil.which("rg")
    if rg is not None:
        return _search_with_rg(root, rg, query, regex, path_glob, max_results)
    return _search_with_python(root, query, regex, path_glob, max_results)


def _search_with_rg(
    root: Path,
    executable: str,
    query: str,
    regex: bool,
    path_glob: str | None,
    max_results: int,
) -> tuple[list[JsonObject], str, bool]:
    command = [
        executable,
        "--line-number",
        "--column",
        "--no-heading",
        "--color=never",
        "--max-filesize=2M",
    ]
    if not regex:
        command.append("--fixed-strings")
    for directory in sorted(IGNORED_DIRECTORIES):
        command.extend(("--glob", f"!{directory}/**", "--glob", f"!**/{directory}/**"))
    command.extend(("--glob", "!.env", "--glob", "!.env.*", "--glob", "!*.pem"))
    if path_glob is not None:
        command.extend(("--glob", path_glob))
    command.extend(("--", query, "."))
    timed_out = False
    process: subprocess.Popen[str]
    with tempfile.TemporaryFile() as stderr_file:
        try:
            process = subprocess.Popen(
                command,
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
            )
        except OSError as exc:
            raise ToolError(f"could not start ripgrep: {exc}") from exc

        def expire() -> None:
            nonlocal timed_out
            if process.poll() is None:
                timed_out = True
                process.kill()

        timer = threading.Timer(10.0, expire)
        timer.daemon = True
        timer.start()
        matches: list[JsonObject] = []
        truncated = False
        char_count = 0
        assert process.stdout is not None
        try:
            for raw_line in process.stdout:
                parts = raw_line.rstrip("\r\n").split(":", 3)
                if len(parts) != 4:
                    continue
                raw_path, raw_line_number, raw_column, text = parts
                relative = raw_path.removeprefix("./").removeprefix(".\\").replace("\\", "/")
                if is_sensitive_path(relative):
                    continue
                try:
                    line_number = int(raw_line_number)
                    column = int(raw_column)
                except ValueError:
                    continue
                text = text[:2_000]
                next_chars = char_count + len(relative) + len(text)
                if len(matches) >= max_results or next_chars > MAX_RESULT_CHARS:
                    truncated = True
                    process.kill()
                    break
                char_count = next_chars
                matches.append(
                    {"path": relative, "line": line_number, "column": column, "text": text}
                )
            return_code = process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            return_code = process.wait()
        finally:
            timer.cancel()
            process.stdout.close()
        stderr_file.seek(0)
        stderr = stderr_file.read(2_001).decode("utf-8", errors="replace")

    if timed_out:
        raise ToolError("ripgrep search timed out")
    if return_code not in {0, 1} and not truncated:
        stderr, _ = _truncate(stderr.strip(), 2_000)
        raise ToolError(f"ripgrep failed with exit code {return_code}: {stderr}")
    return matches, "ripgrep", truncated


def _search_with_python(
    root: Path,
    query: str,
    regex: bool,
    path_glob: str | None,
    max_results: int,
) -> tuple[list[JsonObject], str, bool]:
    try:
        pattern = re.compile(query) if regex else None
    except re.error as exc:
        raise ToolError(f"invalid regular expression: {exc}") from exc
    files, _ = list_repository_files(root, 10_000)
    matches: list[JsonObject] = []
    char_count = 0
    for relative in files:
        if path_glob is not None and not fnmatch.fnmatchcase(relative, path_glob):
            continue
        path = root / PurePosixPath(relative)
        if path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            match = pattern.search(line) if pattern is not None else None
            column = match.start() + 1 if match is not None else line.find(query) + 1
            if column <= 0:
                continue
            rendered = line[:2_000]
            char_count += len(relative) + len(rendered)
            if len(matches) >= max_results or char_count > MAX_RESULT_CHARS:
                return matches, "python-fallback", True
            matches.append(
                {"path": relative, "line": line_number, "column": column, "text": rendered}
            )
    return matches, "python-fallback", False


def apply_structured_patch(
    root: Path,
    operations: Sequence[Mapping[str, str]],
    *,
    protected_globs: Sequence[str],
    editable_globs: Sequence[str],
) -> tuple[str, ...]:
    """Validate every operation before atomically replacing any target file."""

    originals: dict[Path, bytes | None] = {}
    modes: dict[Path, int] = {}
    final_bytes: dict[Path, bytes | None] = {}
    relative_names: dict[Path, str] = {}

    for operation in operations:
        raw_path = operation["path"]
        path = validate_relative_path(
            root,
            raw_path,
            must_exist=False,
            allow_directory=False,
            reject_sensitive=True,
        )
        relative = path.relative_to(root).as_posix()
        _require_editable(relative, protected_globs, editable_globs)
        if path not in originals:
            exists = path.exists()
            if exists and (path.is_symlink() or not path.is_file()):
                raise ToolError(f"patch target is not a regular file: {relative}")
            if exists and path.stat().st_size > MAX_PATCH_BYTES * 4:
                raise ToolError(f"patch target is too large: {relative}")
            originals[path] = path.read_bytes() if exists else None
            modes[path] = stat.S_IMODE(path.stat().st_mode) if exists else 0o644
            final_bytes[path] = originals[path]
            relative_names[path] = relative

        kind = operation["kind"]
        current = final_bytes[path]
        old_text = operation["old_text"]
        new_text = operation["new_text"]
        if kind == "create":
            if current is not None:
                raise ToolError(f"cannot create existing file: {relative}")
            if old_text:
                raise ToolError("create operations require empty old_text")
            final_bytes[path] = new_text.encode("utf-8")
            continue
        if current is None:
            raise ToolError(f"patch target does not exist: {relative}")
        try:
            current_text = current.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError(f"patch target is not UTF-8 text: {relative}") from exc
        if kind == "delete":
            if new_text:
                raise ToolError("delete operations require empty new_text")
            if current_text != old_text:
                raise ToolError(f"delete old_text must equal the complete file: {relative}")
            final_bytes[path] = None
            continue
        if kind != "replace":
            raise ToolError(f"unsupported patch operation: {kind}")
        if not old_text:
            raise ToolError("replace operations require non-empty old_text")
        occurrences = current_text.count(old_text)
        if occurrences != 1:
            raise ToolError(
                f"replace old_text must occur exactly once in {relative}; found {occurrences}"
            )
        final_bytes[path] = current_text.replace(old_text, new_text, 1).encode("utf-8")

    changed_paths = [path for path in final_bytes if final_bytes[path] != originals[path]]
    total_size = sum(len(content or b"") for content in final_bytes.values())
    if total_size > MAX_PATCH_BYTES * 4:
        raise ToolError("resulting files exceed the patch output limit")
    for path in changed_paths:
        content = final_bytes[path]
        if content is not None and path.suffix.lower() == ".py":
            try:
                compile(content, relative_names[path], "exec", ast.PyCF_ONLY_AST)
            except SyntaxError as exc:
                raise ToolError(
                    f"Python syntax error in {relative_names[path]}:{exc.lineno}:{exc.offset}: "
                    f"{exc.msg}"
                ) from exc
        if content is not None and not path.parent.is_dir():
            raise ToolError(f"parent directory does not exist: {relative_names[path]}")

    staged: dict[Path, Path] = {}
    committed: list[Path] = []
    try:
        for path in changed_paths:
            content = final_bytes[path]
            if content is None:
                continue
            descriptor, temporary_name = tempfile.mkstemp(prefix=".coderepair-", dir=path.parent)
            temporary = Path(temporary_name)
            staged[path] = temporary
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, modes[path])
        for path in changed_paths:
            content = final_bytes[path]
            if content is None:
                path.unlink()
            else:
                temporary = staged[path]
                os.replace(temporary, path)
                staged.pop(path)
            committed.append(path)
    except Exception as exc:
        rollback_errors: list[str] = []
        for path in reversed(committed):
            rollback_temporary: Path | None = None
            try:
                original = originals[path]
                if original is None:
                    path.unlink(missing_ok=True)
                else:
                    descriptor, temporary_name = tempfile.mkstemp(
                        prefix=".coderepair-rollback-", dir=path.parent
                    )
                    rollback_temporary = Path(temporary_name)
                    with os.fdopen(descriptor, "wb") as handle:
                        handle.write(original)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.chmod(rollback_temporary, modes[path])
                    os.replace(rollback_temporary, path)
                    rollback_temporary = None
            except Exception as rollback_exc:  # pragma: no cover - catastrophic filesystem error
                rollback_errors.append(f"{relative_names[path]}: {rollback_exc}")
            finally:
                if rollback_temporary is not None:
                    rollback_temporary.unlink(missing_ok=True)
        detail = f"; rollback errors: {rollback_errors}" if rollback_errors else ""
        raise ToolError(f"atomic patch commit failed: {exc}{detail}") from exc
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)

    return tuple(sorted(relative_names[path] for path in changed_paths))


def _require_editable(
    relative: str, protected_globs: Sequence[str], editable_globs: Sequence[str]
) -> None:
    if any(_glob_matches(relative, pattern) for pattern in protected_globs):
        raise ToolError(f"path is protected from model edits: {relative}")
    if not any(_glob_matches(relative, pattern) for pattern in editable_globs):
        raise ToolError(f"path is outside the task's editable globs: {relative}")


def _glob_matches(relative: str, pattern: str) -> bool:
    folded_relative = relative.casefold()
    folded_pattern = pattern.casefold()
    return fnmatch.fnmatchcase(folded_relative, folded_pattern) or PurePosixPath(
        folded_relative
    ).match(folded_pattern)


def _validate_glob(value: str) -> None:
    if "\x00" in value or "\\" in value:
        raise ToolError("path_glob must use safe POSIX syntax")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ToolError("path_glob cannot be absolute or traverse parents")


def _check_length(value: Sequence[Any], schema: Mapping[str, Any], path: str) -> None:
    minimum = schema.get("minLength", schema.get("minItems"))
    maximum = schema.get("maxLength", schema.get("maxItems"))
    if minimum is not None and len(value) < minimum:
        raise ToolError(f"{path} is shorter than allowed")
    if maximum is not None and len(value) > maximum:
        raise ToolError(f"{path} is longer than allowed")


def _result_to_dict(result: Any) -> JsonObject:
    if isinstance(result, dict):
        return dict(result)
    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, dict):
            return value
    if is_dataclass(result) and not isinstance(result, type):
        return asdict(result)
    raise ToolError(f"test runner returned unsupported result: {type(result).__name__}")


def _truncate(value: str, maximum: int) -> tuple[str, bool]:
    if len(value) <= maximum:
        return value, False
    return value[:maximum] + "\n... [truncated by CodeRepair]", True
