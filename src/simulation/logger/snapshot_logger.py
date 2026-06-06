from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SnapshotRecord:
    tick: int
    rank: int
    uid: str
    agent_class: str
    state: Optional[str]
    x: int
    y: int
    compartment: Optional[str]
    owner_uid: Optional[str]
    aggregate_id: Optional[int]


class SnapshotLogger:
    def __init__(self, rank: int, comm=None, output_dir: Path | str = "output/run_logs", rank_output_dir: Path | str | None = None, enabled: bool = False):
        self.rank = rank
        self.comm = comm
        self.output_dir = Path(output_dir)
        self.rank_output_dir = resolve_rank_output_dir(self.output_dir, rank_output_dir)
        self.path = self.rank_output_dir / f"snapshots_rank{self.rank}.jsonl"
        self.merged_path = self.output_dir / "snapshots.jsonl"
        self.enabled = enabled
        self._stream = None
        if not self.enabled:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rank_output_dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        if self.rank == 0:
            self.merged_path.write_text("", encoding="utf-8")
        self._barrier()
        self._stream = self.path.open("a", encoding="utf-8")

    def record_agent(self, tick: int, agent, position) -> None:
        if not self.enabled or self._stream is None or position is None:
            return
        record = SnapshotRecord(
            tick=int(tick),
            rank=self.rank,
            uid=uid_of(agent) or "",
            agent_class=type(agent).__name__,
            state=value_of(getattr(agent, "state", None)),
            x=int(getattr(position, "x")),
            y=int(getattr(position, "y")),
            compartment=value_of(getattr(agent, "compartment", None)),
            owner_uid=uid_of(getattr(agent, "owner_neuron", None)),
            aggregate_id=getattr(agent, "aggregate_id", None)
        )
        self._stream.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")

    def close(self) -> None:
        if not self.enabled:
            return
        if self._stream is not None:
            self._stream.flush()
            self._stream.close()
            self._stream = None
        self._barrier()
        if self.rank == 0:
            self._merge_rank_files("snapshots_rank*.jsonl", self.merged_path, source_dir=self.rank_output_dir)
        self._barrier()

    def _merge_rank_files(self, pattern: str, destination: Path, *, source_dir: Path) -> None:
        with destination.open("w", encoding="utf-8") as output:
            for path in sorted(source_dir.glob(pattern), key=_rank_path_key):
                with path.open("r", encoding="utf-8", errors="replace") as stream:
                    for line in stream:
                        if line.strip():
                            output.write(line)

    def _barrier(self) -> None:
        barrier = getattr(self.comm, "Barrier", None)
        if callable(barrier):
            barrier()


def _rank_path_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    marker = "_rank"
    if marker in stem:
        try:
            return int(stem.rsplit(marker, 1)[1]), path.name
        except ValueError:
            pass
    return 10**9, path.name


def uid_of(agent) -> Optional[str]:
    if agent is None:
        return None
    uid = getattr(agent, "uid", None)
    if uid is not None:
        if isinstance(uid, tuple):
            return ":".join(str(item) for item in uid)
        return str(uid)
    local_id = getattr(agent, "local_id", getattr(agent, "id", ""))
    ptype = getattr(agent, "ptype", getattr(agent, "type_id", ""))
    rank = getattr(agent, "rank", "")
    if local_id == "" and ptype == "" and rank == "":
        return None
    return f"{local_id}:{ptype}:{rank}"


def value_of(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def resolve_rank_output_dir(output_dir: Path, rank_output_dir: Path | str | None) -> Path:
    if rank_output_dir is not None:
        return Path(rank_output_dir)
    if output_dir.name == "run_logs":
        return output_dir.parent / "logs_per_rank"
    return output_dir / "logs_per_rank"
