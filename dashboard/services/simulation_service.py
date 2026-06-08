from __future__ import annotations
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pandas as pd
import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "metrics"

@dataclass(frozen=True)
class SimulationStatus:
    running: bool
    pid: int | None
    return_code: int | None = None

class SimulationService:
    def __init__(self, *, project_root: Path = PROJECT_ROOT, output_dir: Path = DEFAULT_OUTPUT_DIR, mpi_executable: str = "mpiexec", ranks: int = 4, mode: str = "rule") -> None:
        self.project_root = Path(project_root)
        self.output_dir = Path(output_dir)
        self.mpi_executable = mpi_executable
        self.ranks = ranks
        self.mode = mode

    @property
    def engine_path(self) -> Path:
        return self.project_root / "src" / "simulation" / "engine.py"

    @property
    def process_path(self) -> Path:
        return self.output_dir / "simulation_process.json"

    @property
    def console_path(self) -> Path:
        return self.output_dir / "console.log"

    @property
    def tick_metrics_path(self) -> Path:
        return self.output_dir / "tick_metrics.csv"

    def build_command(self) -> list[str]:
        return [self.mpi_executable, "-n", str(self.ranks), sys.executable, str(self.engine_path), "--mode", self.mode]

    def start(self) -> int:
        if self.is_running():
            raise RuntimeError("A simulation is already running.")
        if not self.engine_path.exists():
            raise FileNotFoundError(f"Simulation engine not found: {self.engine_path}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.process_path.unlink(missing_ok=True)
        self.console_path.unlink(missing_ok=True)
        command = self.build_command()
        console_stream = self.console_path.open("a", encoding="utf-8")
        try:
            process = subprocess.Popen(
                command, cwd=self.project_root,
                stdin=subprocess.DEVNULL, stdout=console_stream, stderr=subprocess.STDOUT,
                creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0),
                start_new_session=sys.platform != "win32")
        except Exception:
            console_stream.close()
            raise
        self._write_process_metadata({"pid": process.pid, "command": command, "running": True})
        return process.pid

    def status(self) -> SimulationStatus:
        metadata = self._read_process_metadata()
        pid = self._parse_pid(metadata.get("pid"))
        if pid is None:
            return SimulationStatus(running=False, pid=None)
        running = psutil.pid_exists(pid)
        return SimulationStatus(running=running, pid=pid, return_code=None)

    def is_running(self) -> bool:
        return self.status().running

    def read_tick_metrics(self) -> pd.DataFrame:
        if not self.tick_metrics_path.exists():
            return pd.DataFrame()
        for _ in range(2):
            try:
                frame = pd.read_csv(self.tick_metrics_path)
            except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
                continue
            if "tick" in frame.columns:
                frame["tick"] = pd.to_numeric(frame["tick"], errors="coerce")
                frame = frame.dropna(subset=["tick"])
                frame["tick"] = frame["tick"].astype(int)
            return frame
        return pd.DataFrame()

    def read_console_tail(self, line_count: int = 80) -> str:
        if not self.console_path.exists():
            return ""
        lines = self.console_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-line_count:])

    def _write_process_metadata(self, payload: dict[str, Any]) -> None:
        temporary = self.process_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.process_path)

    def _read_process_metadata(self) -> dict[str, Any]:
        if not self.process_path.exists():
            return {}
        try:
            payload = json.loads(self.process_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _parse_pid(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
