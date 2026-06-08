from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Iterator
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT_PATH = PROJECT_ROOT / "output" / "run_logs" / "spatial_snapshots.jsonl"

class SpatialSnapshotError(RuntimeError):
    pass

class SpatialReconstructionService:
    REQUIRED_COLUMNS = {"tick", "uid", "agent_class", "x", "y"}
    OPTIONAL_COLUMNS = {"run_id", "rank", "state", "compartment", "owner_uid", "aggregate_id"}
    def __init__(self, snapshot_path: Path | str = DEFAULT_SNAPSHOT_PATH) -> None:
        self.snapshot_path = Path(snapshot_path)

    def has_simulation(self) -> bool:
        return self.snapshot_path.exists() and self.snapshot_path.is_file() and self.snapshot_path.stat().st_size > 0

    def load(self) -> pd.DataFrame:
        if not self.has_simulation():
            return pd.DataFrame()
        rows = list(self._iter_rows())
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows)
        missing = self.REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise SpatialSnapshotError(f"Snapshot log is missing required columns: {missing_text}")
        frame = frame.drop(columns=["z"], errors="ignore")
        frame["tick"] = pd.to_numeric(frame["tick"], errors="coerce")
        frame["x"] = pd.to_numeric(frame["x"], errors="coerce")
        frame["y"] = pd.to_numeric(frame["y"], errors="coerce")
        frame = frame.dropna(subset=["tick", "x", "y", "uid", "agent_class"])
        if frame.empty:
            return pd.DataFrame()
        frame["tick"] = frame["tick"].astype(int)
        frame["x"] = frame["x"].astype(int)
        frame["y"] = frame["y"].astype(int)
        if "rank" in frame.columns:
            frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce").astype("Int64")
        if "aggregate_id" in frame.columns:
            frame["aggregate_id"] = pd.to_numeric(frame["aggregate_id"], errors="coerce").astype("Int64")
        for column in ("run_id", "uid", "agent_class", "state", "compartment", "owner_uid"):
            if column in frame.columns:
                frame[column] = frame[column].astype("string")
        return frame.sort_values(["tick","agent_class","uid",], kind="stable").reset_index(drop=True)

    def available_ticks(self, snapshots: pd.DataFrame) -> list[int]:
        if snapshots.empty or "tick" not in snapshots.columns:
            return []
        return sorted(snapshots["tick"].dropna().astype(int).unique().tolist())

    def available_agent_classes(self, snapshots: pd.DataFrame) -> list[str]:
        if snapshots.empty or "agent_class" not in snapshots.columns:
            return []
        return sorted(snapshots["agent_class"].dropna().astype(str).unique().tolist())

    def available_states(self, snapshots: pd.DataFrame) -> list[str]:
        if snapshots.empty or "state" not in snapshots.columns:
            return []
        return sorted(snapshots["state"].dropna().astype(str).unique().tolist())

    def frame_at(self, snapshots: pd.DataFrame, tick: int, *, agent_classes: list[str] | None = None, states: list[str] | None = None) -> pd.DataFrame:
        if snapshots.empty:
            return snapshots.copy()
        frame = snapshots[snapshots["tick"] == int(tick)].copy()
        if agent_classes:
            frame = frame[frame["agent_class"].astype(str).isin(agent_classes)]
        if states and "state" in frame.columns:
            frame = frame[frame["state"].astype(str).isin(states)]
        return frame.reset_index(drop=True)

    def grid_bounds(self, snapshots: pd.DataFrame) -> tuple[int, int, int, int]:
        if snapshots.empty:
            return 0, 0, 0, 0
        return (int(snapshots["x"].min()), int(snapshots["x"].max()), int(snapshots["y"].min()), int(snapshots["y"].max()))

    def _iter_rows(self) -> Iterator[dict[str, Any]]:
        with self.snapshot_path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    yield row
