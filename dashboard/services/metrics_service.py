from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import pandas as pd
from src.analysis.metrics.mechanism_metrics import summarize_mechanisms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TICK_METRICS_PATH = PROJECT_ROOT / "output" / "metrics" / "tick_metrics.csv"
DEFAULT_EVENT_LOG_PATH = PROJECT_ROOT / "output" / "run_logs" / "events.jsonl"
DEFAULT_EVENT_LOG_DIR = DEFAULT_EVENT_LOG_PATH.parent
DEFAULT_MECHANISM_METRICS_PATH = PROJECT_ROOT / "output" / "metrics" / "mechanism_metrics_latest.json"

class MetricsDataError(RuntimeError):
    pass


class MetricsService:
    def __init__(self, *, tick_metrics_path: Path | str = DEFAULT_TICK_METRICS_PATH, event_log_path: Path | str = DEFAULT_EVENT_LOG_PATH, mechanism_metrics_path: Path | str = DEFAULT_MECHANISM_METRICS_PATH) -> None:
        self.tick_metrics_path = Path(tick_metrics_path)
        self.event_log_path = Path(event_log_path)
        self.event_log_dir = self.event_log_path.parent
        self.mechanism_metrics_path = Path(mechanism_metrics_path)

    def has_simulation_output(self) -> bool:
        return self.has_tick_metrics() or self.has_event_logs()

    def has_tick_metrics(self) -> bool:
        return self._is_non_empty_file(self.tick_metrics_path)

    def has_event_logs(self) -> bool:
        return self._is_non_empty_file(self.event_log_path)

    def has_mechanism_metrics(self) -> bool:
        return self._is_non_empty_file(self.mechanism_metrics_path)

    def load_tick_metrics(self) -> pd.DataFrame:
        if not self.has_tick_metrics():
            return pd.DataFrame()
        try:
            frame = pd.read_csv(self.tick_metrics_path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
        except (OSError, pd.errors.ParserError) as exc:
            raise MetricsDataError(f"Unable to read tick metrics from `{self.tick_metrics_path}`: {exc}") from exc
        if frame.empty:
            return frame
        frame.columns = [str(column).strip() for column in frame.columns]
        if "tick" not in frame.columns:
            raise MetricsDataError("tick_metrics.csv does not contain the required `tick` column.")
        frame["tick"] = pd.to_numeric(frame["tick"], errors="coerce")
        frame = frame.dropna(subset=["tick"])
        if frame.empty:
            return pd.DataFrame(columns=frame.columns)
        frame["tick"] = frame["tick"].astype(int)
        for column in frame.columns:
            if column == "tick":
                continue
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame.sort_values("tick", kind="stable").drop_duplicates(subset=["tick"], keep="last").reset_index(drop=True)

    @staticmethod
    def tick_metric_columns(metrics: pd.DataFrame) -> list[str]:
        if metrics.empty:
            return []
        return [column for column in metrics.columns if column != "tick"]

    def compute_mechanism_metrics(self, *, include_by_tick: bool = True) -> dict[str, Any]:
        if not self.has_event_logs():
            raise FileNotFoundError(f"Event log not found or empty: {self.event_log_path}")
        run_report = summarize_mechanisms(self.event_log_dir, include_by_tick=include_by_tick)
        report: dict[str, Any] = {"runs": [run_report]}
        self._write_json_atomic(self.mechanism_metrics_path, report)
        return report

    def load_mechanism_metrics(self) -> dict[str, Any]:
        if not self.has_mechanism_metrics():
            return {}
        try:
            value = json.loads(self.mechanism_metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MetricsDataError(f"Unable to read mechanism metrics from `{self.mechanism_metrics_path}`: {exc}") from exc
        if not isinstance(value, dict):
            raise MetricsDataError("The mechanism metrics report must contain a JSON object.")
        return value

    @staticmethod
    def current_run_report(report: dict[str, Any]) -> dict[str, Any]:
        runs = report.get("runs")
        if isinstance(runs, list) and runs and isinstance(runs[0], dict):
            return runs[0]
        if "all_mechanisms" in report or "selected_mechanisms" in report:
            return report
        return {}

    @staticmethod
    def mechanism_counts_frame(run_report: dict[str, Any]) -> pd.DataFrame:
        counts = run_report.get("all_mechanisms", {}).get("counts", {})
        columns = ["mechanism", "count"]
        if not isinstance(counts, dict):
            return pd.DataFrame(columns=columns)
        rows = [{"mechanism": str(mechanism), "count": count} for mechanism, count in counts.items()]
        if not rows:
            return pd.DataFrame(columns=columns)
        frame = pd.DataFrame(rows, columns=columns)
        frame["count"] = pd.to_numeric(frame["count"], errors="coerce").fillna(0).astype(int)
        return frame.sort_values(["count", "mechanism"], ascending=[False, True], kind="stable").reset_index(drop=True)

    @staticmethod
    def mechanism_timeline_frame(run_report: dict[str, Any]) -> pd.DataFrame:
        by_tick = run_report.get("by_tick", {})
        if not isinstance(by_tick, dict):
            return pd.DataFrame()
        rows: list[dict[str, Any]] = []
        for raw_tick, raw_counts in by_tick.items():
            if not isinstance(raw_counts, dict):
                continue
            try:
                tick = int(raw_tick)
            except (TypeError, ValueError):
                continue
            row: dict[str, Any] = {"tick": tick}
            for mechanism, count in raw_counts.items():
                row[str(mechanism)] = count
            rows.append(row)
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows).fillna(0)
        for column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
        frame["tick"] = frame["tick"].astype(int)
        return frame.sort_values("tick", kind="stable").reset_index(drop=True)

    @classmethod
    def mechanism_profile_frame(cls, run_report: dict[str, Any], mechanism: str) -> pd.DataFrame:
        timeline = cls.mechanism_timeline_frame(run_report)
        if timeline.empty:
            return pd.DataFrame(columns=["tick", "count", "share_percent"])
        if mechanism not in timeline.columns:
            timeline[mechanism] = 0
        profile = timeline[["tick", mechanism]].copy()
        profile = profile.rename(columns={mechanism: "count"})
        profile["count"] = pd.to_numeric(profile["count"], errors="coerce").fillna(0).astype(int)
        total = int(profile["count"].sum())
        if total > 0:
            profile["share_percent"] = profile["count"] / total * 100.0
        else:
            profile["share_percent"] = 0.0
        return profile.reset_index(drop=True)

    @staticmethod
    def mechanism_details(run_report: dict[str, Any], mechanism: str) -> dict[str, Any]:
        all_mechanisms = run_report.get("all_mechanisms", {})
        if not isinstance(all_mechanisms, dict):
            return {}
        counts = all_mechanisms.get("counts", {})
        first_ticks = all_mechanisms.get("first_tick", {})
        last_ticks = all_mechanisms.get("last_tick", {})
        probability_summary = all_mechanisms.get("probability_summary", {})
        rng_summary = all_mechanisms.get("rng_summary", {})
        return {"mechanism": mechanism,
            "count": counts.get(mechanism, 0) if isinstance(counts, dict) else 0,
            "first_tick": first_ticks.get(mechanism) if isinstance(first_ticks, dict) else None,
            "last_tick": last_ticks.get(mechanism) if isinstance(last_ticks, dict) else None,
            "probability_summary": probability_summary.get(mechanism, {}) if isinstance(probability_summary, dict) else {},
            "rng_summary": rng_summary.get(mechanism, {}) if isinstance(rng_summary, dict) else {}}

    @staticmethod
    def mechanism_lifecycle_frame(run_report: dict[str, Any]) -> pd.DataFrame:
        all_mechanisms = run_report.get("all_mechanisms", {})
        columns = ["mechanism", "count", "first_tick", "last_tick"]
        if not isinstance(all_mechanisms, dict):
            return pd.DataFrame(columns=columns)
        counts = all_mechanisms.get("counts", {})
        first_ticks = all_mechanisms.get("first_tick", {})
        last_ticks = all_mechanisms.get("last_tick", {})
        if not isinstance(counts, dict):
            return pd.DataFrame(columns=columns)
        first_ticks = first_ticks if isinstance(first_ticks, dict) else {}
        last_ticks = last_ticks if isinstance(last_ticks, dict) else {}
        rows = [{"mechanism": str(mechanism), "count": count,
                "first_tick": first_ticks.get(mechanism), "last_tick": last_ticks.get(mechanism)}
            for mechanism, count in counts.items()]
        if not rows:
            return pd.DataFrame(columns=columns)
        frame = pd.DataFrame(rows, columns=columns)
        for column in ("count", "first_tick", "last_tick"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
        return frame.sort_values(["count", "mechanism"], ascending=[False, True], kind="stable").reset_index(drop=True)

    @classmethod
    def selected_mechanism_summary_frame(cls, run_report: dict[str, Any]) -> pd.DataFrame:
        selected = run_report.get("selected_mechanisms", {})
        columns = ["group", "metric", "value"]
        if not isinstance(selected, dict):
            return pd.DataFrame(columns=columns)
        rows: list[dict[str, Any]] = []
        for group, value in selected.items():
            cls._flatten_metric_mapping(rows=rows, group=str(group), prefix="", value=value)
        return pd.DataFrame(rows, columns=columns)

    @classmethod
    def _flatten_metric_mapping(cls, *, rows: list[dict[str, Any]], group: str, prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                cls._flatten_metric_mapping(rows=rows, group=group, prefix=path, value=child)
            return
        if isinstance(value, list):
            displayed_value: Any = json.dumps(value, ensure_ascii=False)
        else:
            displayed_value = value
        rows.append({"group": group,"metric": prefix, "value": displayed_value})

    @staticmethod
    def _is_non_empty_file(path: Path) -> bool:
        try:
            return path.exists() and path.is_file() and path.stat().st_size > 0
        except OSError:
            return False

    @staticmethod
    def _write_json_atomic(path: Path,payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        temporary.replace(path)
