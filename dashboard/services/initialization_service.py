from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Iterator
import pandas as pd
from collections.abc import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INITIALIZATION_DIR = PROJECT_ROOT / "output" / "initialization_logs"
DEFAULT_AGENTS_PATH = DEFAULT_INITIALIZATION_DIR / "initialization_agents.jsonl"
DEFAULT_MANIFEST_PATH = DEFAULT_INITIALIZATION_DIR / "initialization_manifest.json"
DEFAULT_SUMMARY_PATH = DEFAULT_INITIALIZATION_DIR / "initialization_summary.json"


class InitializationDataError(RuntimeError):
    pass

class InitializationService:
    REQUIRED_AGENT_COLUMNS = {"uid", "agent_class", "rank", "initial_state"}
    def __init__(self, *, agents_path: Path | str = DEFAULT_AGENTS_PATH, manifest_path: Path | str = DEFAULT_MANIFEST_PATH, summary_path: Path | str = DEFAULT_SUMMARY_PATH) -> None:
        self.agents_path = Path(agents_path)
        self.manifest_path = Path(manifest_path)
        self.summary_path = Path(summary_path)

    def has_initialization_data(self) -> bool:
        return self.agents_path.exists() and self.agents_path.is_file() and self.agents_path.stat().st_size > 0

    def load_agents(self) -> pd.DataFrame:
        if not self.has_initialization_data():
            return pd.DataFrame()
        rows = list(self._iter_jsonl(self.agents_path))
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows)
        missing = self.REQUIRED_AGENT_COLUMNS.difference(frame.columns)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise InitializationDataError(f"Initialization log is missing required columns: {missing_text}")
        frame = self._normalize_agent_frame(frame)
        return (frame.sort_values(
                ["agent_class", "rank", "local_id", "uid"], kind="stable",
                na_position="last").reset_index(drop=True))

    def load_summary(self) -> dict[str, Any]:
        return self._read_json_object(self.summary_path)

    def load_manifest(self) -> dict[str, Any]:
        return self._read_json_object(self.manifest_path)

    def available_classes(self, agents: pd.DataFrame) -> list[str]:
        return self._unique_strings(agents, "agent_class")

    def available_states(self, agents: pd.DataFrame) -> list[str]:
        return self._unique_strings(agents, "initial_state")

    def available_compartments(self, agents: pd.DataFrame) -> list[str]:
        return self._unique_strings(agents, "compartment", include_missing=True)

    def available_ranks(self, agents: pd.DataFrame) -> list[int]:
        if agents.empty or "rank" not in agents.columns:
            return []
        values = pd.to_numeric(agents["rank"], errors="coerce").dropna()
        return sorted(values.astype(int).unique().tolist())

    def available_owner_uids(self, agents: pd.DataFrame) -> list[str]:
        return self._unique_strings(agents, "owner_uid")

    def filter_agents(self, agents: pd.DataFrame, *, search: str = "", agent_classes: list[str] | None = None, states: list[str] | None = None, compartments: list[str] | None = None, ranks: list[int] | None = None, owner_uid: str | None = None) -> pd.DataFrame:
        if agents.empty:
            return agents.copy()
        frame = agents.copy()
        if agent_classes:
            frame = frame[frame["agent_class"].astype(str).isin(agent_classes)]
        if states:
            frame = frame[frame["initial_state"].astype(str).isin(states)]
        if compartments:
            normalized_compartments = [None if value == "None" else value for value in compartments]
            compartment_mask = pd.Series(False, index=frame.index)
            for compartment in normalized_compartments:
                if compartment is None:
                    compartment_mask |= frame["compartment"].isna()
                else:
                    compartment_mask |= frame["compartment"].astype(str) == compartment
            frame = frame[compartment_mask]
        if ranks:
            frame = frame[pd.to_numeric(frame["rank"], errors="coerce").isin(ranks)]
        if owner_uid:
            frame = frame[frame["owner_uid"].astype(str) == owner_uid]
        search_text = search.strip().lower()
        if search_text:
            searchable_columns = [
                column
                for column in ("uid", "agent_class", "initial_state", "compartment", "owner_uid", "target_uid", "target_class", "display_label", "display_group", "visual_level")
                if column in frame.columns]
            if searchable_columns:
                search_mask = pd.Series(False, index=frame.index)
                for column in searchable_columns:
                    search_mask |= frame[column].fillna("").astype(str).str.lower().str.contains(search_text, regex=False)
                frame = frame[search_mask]
        return frame.reset_index(drop=True)

    def agent_by_uid(self, agents: pd.DataFrame, uid: str) -> dict[str, Any] | None:
        if agents.empty:
            return None
        match = agents[agents["uid"].astype(str) == str(uid)]
        if match.empty:
            return None
        return match.iloc[0].to_dict()

    def child_agents(self, agents: pd.DataFrame, owner_uid: str) -> pd.DataFrame:
        if agents.empty or "owner_uid" not in agents.columns:
            return pd.DataFrame()
        return agents[agents["owner_uid"].astype(str) == str(owner_uid)].sort_values(["agent_class", "uid"], kind="stable").reset_index(drop=True)

    def owner_agent(self, agents: pd.DataFrame, agent: dict[str, Any]) -> dict[str, Any] | None:
        owner_uid = agent.get("owner_uid")
        if not owner_uid or pd.isna(owner_uid):
            return None
        return self.agent_by_uid(agents, str(owner_uid))

    def neuron_manifest_entry(self, manifest: dict[str, Any], uid: str) -> dict[str, Any] | None:
        neurons = manifest.get("neurons")
        if not isinstance(neurons, dict):
            return None
        value = neurons.get(str(uid))
        return value if isinstance(value, dict) else None

    def summary_counts_by_class(self, summary: dict[str, Any]) -> pd.DataFrame:
        values = summary.get("counts_by_class", {})
        if not isinstance(values, dict):
            return pd.DataFrame(columns=["agent_class", "count"])
        return pd.DataFrame([
            {"agent_class": agent_class, "count": count}
            for agent_class, count in values.items()]).sort_values("count", ascending=False)

    def summary_counts_by_rank(self, summary: dict[str, Any]) -> pd.DataFrame:
        values = summary.get("counts_by_rank", {})
        if not isinstance(values, dict):
            return pd.DataFrame(columns=["rank", "agent_class", "count"])
        rows: list[dict[str, Any]] = []
        for rank, counts in values.items():
            if not isinstance(counts, dict):
                continue
            for agent_class, count in counts.items():
                rows.append({"rank": int(rank), "agent_class": agent_class, "count": count})
        return pd.DataFrame(rows)

    def summary_counts_by_state(self, summary: dict[str, Any]) -> pd.DataFrame:
        values = summary.get("counts_by_initial_state", {})
        if not isinstance(values, dict):
            return pd.DataFrame(columns=["agent_class", "initial_state", "count"])
        rows: list[dict[str, Any]] = []
        for agent_class, states in values.items():
            if not isinstance(states, dict):
                continue
            for state, count in states.items():
                rows.append({"agent_class": agent_class, "initial_state": state, "count": count})
        return pd.DataFrame(rows)

    def compact_agent_table(self, agents: pd.DataFrame) -> pd.DataFrame:
        frame = self.add_threshold_columns(agents)
        columns = [
            column
            for column in ("uid", "display_label", "agent_class", "compartment", "rank", "owner_uid", "position_x", "position_y", "threshold_count", "threshold_summary")
            if column in frame.columns]
        return frame[columns].copy()

    def _normalize_agent_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        numeric_columns = ("rank", "local_id", "type_id", "aggregate_id")
        for column in numeric_columns:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
        string_columns = ("uid", "agent_class", "initial_state", "compartment", "owner_uid", "target_uid", "target_class")
        for column in string_columns:
            if column in frame.columns:
                frame[column] = frame[column].astype("string")
        if "position" in frame.columns:
            frame["position_x"] = frame["position"].apply(lambda value: self._dict_value(value, "x"))
            frame["position_y"] = frame["position"].apply(lambda value: self._dict_value(value, "y"))
        if "display" in frame.columns:
            frame["display_label"] = frame["display"].apply(lambda value: self._dict_value(value, "label"))
            frame["display_group"] = frame["display"].apply(lambda value: self._dict_value(value, "group"))
            frame["owner_label"] = frame["display"].apply(lambda value: self._dict_value(value, "owner_label"))
            frame["visual_level"] = frame["display"].apply(lambda value: self._dict_value(value, "visual_level"))
        for column in ("position_x", "position_y"):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
        return frame

    @staticmethod
    def _dict_value(value: Any, key: str) -> Any:
        if isinstance(value, dict):
            return value.get(key)
        return None

    @staticmethod
    def _unique_strings(frame: pd.DataFrame, column: str, *, include_missing: bool = False) -> list[str]:
        if frame.empty or column not in frame.columns:
            return []
        values = frame[column]
        result = sorted(values.dropna().astype(str).unique().tolist())
        if include_missing and values.isna().any():
            result.append("None")
        return result

    @staticmethod
    def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    yield value

    @staticmethod
    def _read_json_object(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def threshold_frame(self, agents: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        if agents.empty:
            return pd.DataFrame(columns=["uid", "agent_class", "rank", "owner_uid", "threshold", "value"])
        for _, agent in agents.iterrows():
            config = agent.get("config")
            if not isinstance(config, Mapping):
                continue
            thresholds = self.extract_thresholds(config)
            for threshold_name, threshold_value in thresholds.items():
                rows.append({"uid": str(agent.get("uid", "")),
                        "agent_class": str(agent.get("agent_class", "")),
                        "rank": agent.get("rank"),
                        "owner_uid": agent.get("owner_uid"),
                        "threshold": threshold_name,
                        "value": threshold_value})
        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(columns=["uid", "agent_class", "rank", "owner_uid", "threshold", "value"])
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        return frame.dropna(subset=["value"]).sort_values(["agent_class", "threshold", "uid"], kind="stable").reset_index(drop=True)

    def extract_thresholds(self, config: Mapping[str, Any]) -> dict[str, float]:
        result: dict[str, float] = {}
        self._collect_thresholds(value=config, prefix="", result=result,inside_threshold_group=False)
        return result

    def add_threshold_columns(self, agents: pd.DataFrame) -> pd.DataFrame:
        if agents.empty:
            return agents.copy()
        frame = agents.copy()
        extracted = frame["config"].apply(lambda value: self.extract_thresholds(value) if isinstance(value, Mapping) else {})
        frame["threshold_count"] = extracted.apply(len)
        frame["threshold_summary"] = extracted.apply(self.format_threshold_summary)
        return frame

    def thresholds_for_agent(self, thresholds: pd.DataFrame, uid: str) -> pd.DataFrame:
        if thresholds.empty:
            return thresholds.copy()

        return thresholds[thresholds["uid"].astype(str) == str(uid)].sort_values("threshold", kind="stable").reset_index(drop=True)

    def available_thresholds(self, thresholds: pd.DataFrame, *, agent_class: str | None = None) -> list[str]:
        if thresholds.empty:
            return []
        frame = thresholds
        if agent_class is not None:
            frame = frame[frame["agent_class"].astype(str) == str(agent_class)]
        return sorted(frame["threshold"].dropna().astype(str).unique().tolist())

    def threshold_distribution(self, thresholds: pd.DataFrame, *, threshold: str, agent_class: str | None = None) -> pd.DataFrame:
        if thresholds.empty:
            return thresholds.copy()
        frame = thresholds[thresholds["threshold"].astype(str) == str(threshold)].copy()
        if agent_class is not None:
            frame = frame[frame["agent_class"].astype(str) == str(agent_class)]
        return frame.reset_index(drop=True)

    @staticmethod
    def format_threshold_summary(thresholds: dict[str, float], *, limit: int = 3) -> str:
        if not thresholds:
            return ""
        items = sorted(thresholds.items())
        visible = [
            f"{InitializationService.short_threshold_name(key)}={value:g}"
            for key, value in items[:limit]
        ]
        remaining = len(items) - limit
        if remaining > 0:
            visible.append(f"+{remaining} more")
        return "; ".join(visible)

    @staticmethod
    def short_threshold_name(path: str) -> str:
        parts = path.split(".")
        if len(parts) <= 2:
            return path
        return ".".join(parts[-2:])

    @classmethod
    def _collect_thresholds(cls, *, value: Any, prefix: str, result: dict[str, float], inside_threshold_group: bool) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key)
                path = f"{prefix}.{key_text}" if prefix else key_text
                key_is_threshold = ("threshold" in key_text.lower())
                cls._collect_thresholds(value=child,
                    prefix=path, result=result,
                    inside_threshold_group=inside_threshold_group or key_is_threshold)
            return
        if not inside_threshold_group:
            return
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            result[prefix] = float(value)

    def threshold_value_for_agent(self, thresholds: pd.DataFrame, *, uid: str, threshold: str) -> float | None:
        if thresholds.empty:
            return None
        matches = thresholds[(thresholds["uid"].astype(str) == str(uid)) & (thresholds["threshold"].astype(str) == str(threshold))]
        if matches.empty:
            return None
        value = pd.to_numeric(matches.iloc[0]["value"], errors="coerce")
        if pd.isna(value):
            return None
        return float(value)
