from __future__ import annotations
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

DEFAULT_SIMULATION_LOG_DIR = Path("output/run_logs")
RUN_LOG_DIR_NAMES = {"run_logs"}
RANK_LOG_DIR_NAMES = {"logs_per_rank"}
INITIALIZATION_LOG_DIR_NAMES = {"initialization_logs"}


@dataclass(frozen=True)
class JsonlLoad:
    rows: list[dict[str, Any]]
    stats: dict[str, Any]


@dataclass(frozen=True)
class RunData:
    log_dir: Path

    @classmethod
    def resolve(cls, log_dir: Path | str = DEFAULT_SIMULATION_LOG_DIR, *, required_stems: Sequence[str] = ("events",)) -> "RunData":
        return cls(resolve_log_dir(Path(log_dir), required_stems=required_stems))

    @property
    def run_dir(self) -> Path:
        if self.log_dir.name in RUN_LOG_DIR_NAMES or self.log_dir.name in RANK_LOG_DIR_NAMES or self.log_dir.name in INITIALIZATION_LOG_DIR_NAMES:
            return self.log_dir.parent
        return self.log_dir

    @property
    def rank_log_dir(self) -> Path:
        return self.run_dir / "logs_per_rank"

    @property
    def initialization_log_dir(self) -> Path:
        return self.run_dir / "initialization_logs"

    @property
    def metrics_dir(self) -> Path:
        return self.run_dir / "metrics"

    @property
    def default_table_dir(self) -> Path:
        return self.run_dir / "tables"

    @property
    def event_paths(self) -> list[Path]:
        return jsonl_paths(self.log_dir, "events", rank_directory=self.rank_log_dir)

    @property
    def spatial_snapshot_paths(self) -> list[Path]:
        return jsonl_paths(self.log_dir, "spatial_snapshots", rank_directory=self.rank_log_dir)

    @property
    def initialization_agent_path(self) -> Path:
        return self.initialization_log_dir / "initialization_agents.jsonl"

    @property
    def initialization_manifest_path(self) -> Path:
        return self.initialization_log_dir / "initialization_manifest.json"

    @property
    def tick_metrics_path(self) -> Path:
        return find_tick_metrics(self.run_dir)

    def iter_events(self, *, strict: bool = False) -> Iterator[dict[str, Any]]:
        yield from iter_many_jsonl(self.event_paths, strict=strict)

    def iter_spatial_snapshots(self, *, strict: bool = False) -> Iterator[dict[str, Any]]:
        yield from iter_many_jsonl(self.spatial_snapshot_paths, strict=strict)

    def load_events(self) -> JsonlLoad:
        return load_many_jsonl(self.event_paths)

    def load_initialization_agents(self) -> JsonlLoad:
        return load_jsonl(self.initialization_agent_path)

    def tick_metrics_rows(self) -> list[dict[str, str]]:
        return read_tick_metrics_rows(self.run_dir)

    def tick_metrics_numeric_rows(self) -> list[dict[str, Any]]:
        return read_tick_metrics_numeric(self.run_dir)

    def input_status(self) -> dict[str, Any]:
        tick_metrics = self.tick_metrics_path
        return {
            "log_dir": str(self.log_dir),
            "event_files": [str(path) for path in self.event_paths],
            "spatial_snapshot_files": [str(path) for path in self.spatial_snapshot_paths],
            "initialization_agents_jsonl": str(self.initialization_agent_path) if self.initialization_agent_path.exists() else None,
            "initialization_manifest_json": str(self.initialization_manifest_path) if self.initialization_manifest_path.exists() else None,
            "tick_metrics_csv": str(tick_metrics) if tick_metrics.exists() else None
        }


def candidate_log_dirs(log_dir: Path | str) -> list[Path]:
    base = Path(log_dir)
    candidates = [base]
    if base.name == "run_logs":
        candidates.append(base.parent / "logs_per_rank")
        candidates.append(base.parent / "initialization_logs")
    elif base.name == "logs_per_rank":
        candidates.append(base.parent / "run_logs")
        candidates.append(base.parent / "initialization_logs")
    elif base.name == "initialization_logs":
        candidates.append(base.parent / "run_logs")
    else:
        candidates.extend([base / "run_logs", base / "logs_per_rank", base / "initialization_logs"])
    if base.parent != base:
        candidates.extend([base.parent / "run_logs", base.parent / "logs_per_rank", base.parent / "initialization_logs"])
    return _dedupe_paths(candidates)


def resolve_log_dir(log_dir: Path | str, *, required_stems: Sequence[str] = ("events",)) -> Path:
    candidates = candidate_log_dirs(log_dir)
    if not required_stems:
        return candidates[0]
    for candidate in candidates:
        if all(jsonl_paths(candidate, stem) for stem in required_stems):
            return candidate
    required = ", ".join(required_stems)
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"No log directory with required JSONL stem(s) {required}. Searched: {searched}")


def jsonl_paths(directory: Path | str, stem: str, *, rank_directory: Path | str | None = None) -> list[Path]:
    directory = Path(directory)
    merged = directory / f"{stem}.jsonl"
    if has_jsonl_rows(merged):
        return [merged]
    rank_base = Path(rank_directory) if rank_directory is not None else directory
    rank_paths = [
        path
        for path in sorted(rank_base.glob(f"{stem}_rank*.jsonl"), key=rank_file_sort_key)
        if has_jsonl_rows(path)
    ]
    if rank_paths:
        return rank_paths
    return [
        path
        for path in sorted(directory.glob(f"{stem}_rank*.jsonl"), key=rank_file_sort_key)
        if has_jsonl_rows(path)
    ]


def has_jsonl_rows(path: Path | str) -> bool:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return False
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                return True
    return False


def iter_many_jsonl(paths: Iterable[Path | str], *, strict: bool = False) -> Iterator[dict[str, Any]]:
    for path in paths:
        yield from iter_jsonl(path, strict=strict)


def iter_jsonl(path: Path | str, *, strict: bool = False) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                if strict:
                    raise ValueError(f"Malformed JSONL in {path} at line {line_no}.") from exc
                continue
            if isinstance(row, dict):
                yield row
            elif strict:
                raise ValueError(f"Non-object JSONL row in {path} at line {line_no}.")


def load_jsonl(path: Path | str) -> JsonlLoad:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    stats = _empty_jsonl_stats()
    if not path.exists():
        return JsonlLoad(rows, stats)
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                stats["blank"] += 1
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["malformed"] += 1
                _add_jsonl_example(stats, {"line": line_no, "text": line[:160].rstrip()})
                continue
            if not isinstance(row, dict):
                stats["malformed"] += 1
                _add_jsonl_example(stats, {"line": line_no, "text": line[:160].rstrip(), "error": "non_object_row"})
                continue
            rows.append(row)
            stats["valid"] += 1
    return JsonlLoad(rows, stats)


def load_many_jsonl(paths: Iterable[Path | str]) -> JsonlLoad:
    rows: list[dict[str, Any]] = []
    total = _empty_jsonl_stats()
    for raw_path in paths:
        path = Path(raw_path)
        loaded = load_jsonl(path)
        rows.extend(loaded.rows)
        for key in ("valid", "malformed", "blank"):
            total[key] += loaded.stats[key]
        for example in loaded.stats["examples"]:
            _add_jsonl_example(total, {"file": path.name, **example})
    return JsonlLoad(rows, total)


def rank_file_sort_key(path: Path | str) -> tuple[int, str]:
    path = Path(path)
    match = re.search(r"_rank(\d+)\.jsonl$", path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def tick_metrics_candidates(log_dir: Path | str) -> list[Path]:
    path = Path(log_dir)
    if path.suffix.lower() == ".csv":
        return [path]
    candidates = [path / "tick_metrics.csv"]
    if path.name in RUN_LOG_DIR_NAMES or path.name in RANK_LOG_DIR_NAMES or path.name in INITIALIZATION_LOG_DIR_NAMES:
        parent = path.parent
        candidates.extend(
            [
                parent / "metrics" / "tick_metrics.csv",
                parent / "tick_metrics.csv",
            ]
        )
    else:
        candidates.extend([path / "metrics" / "tick_metrics.csv", path / "run_logs" / "tick_metrics.csv"])
    return _dedupe_paths(candidates)


def find_tick_metrics(log_dir: Path | str) -> Path:
    candidates = tick_metrics_candidates(log_dir)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def read_tick_metrics_rows(log_dir_or_csv: Path | str) -> list[dict[str, str]]:
    path = find_tick_metrics(log_dir_or_csv)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def read_tick_metrics_numeric(log_dir_or_csv: Path | str) -> list[dict[str, Any]]:
    return [
        {key: _coerce_number(value)
        for key, value in row.items()}
        for row in read_tick_metrics_rows(log_dir_or_csv)]


def _empty_jsonl_stats() -> dict[str, Any]:
    return {"valid": 0, "malformed": 0, "blank": 0, "examples": []}


def _add_jsonl_example(stats: dict[str, Any], example: dict[str, Any]) -> None:
    if len(stats["examples"]) < 10:
        stats["examples"].append(example)


def _coerce_number(value: Any) -> Any:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if number.is_integer():
        return int(number)
    return number


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique
