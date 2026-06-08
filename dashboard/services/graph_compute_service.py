from __future__ import annotations
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = PROJECT_ROOT / "output" / "run_logs"
DEFAULT_GRAPH_DIR = PROJECT_ROOT / "output" / "graphs"

@dataclass(frozen=True)
class GraphComputeResult:
    level: str
    command: list[str]
    return_code: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        return self.return_code == 0

class GraphComputeService:
    """Run graph CLI entrypoints from the dashboard."""

    def __init__(self, *, project_root: Path = PROJECT_ROOT, log_dir: Path = DEFAULT_LOG_DIR, graph_dir: Path = DEFAULT_GRAPH_DIR) -> None:
        self.project_root = Path(project_root)
        self.log_dir = Path(log_dir)
        self.graph_dir = Path(graph_dir)

    def compute(self, level: str) -> GraphComputeResult:
        level = level.upper()
        command = self.command_for(level)
        if level == "G3" and not (self.graph_dir / "g2.gexf").exists():
            return GraphComputeResult(level=level, command=command, return_code=1, stdout="", stderr=f"G3 requires an existing G2 graph at {self.graph_dir / 'g2.gexf'}.")
        completed = subprocess.run(command, cwd=self.project_root, capture_output=True, text=True, encoding="utf-8", errors="replace")
        return GraphComputeResult(level=level, command=command, return_code=completed.returncode, stdout=completed.stdout.strip(), stderr=completed.stderr.strip())

    def command_for(self, level: str) -> list[str]:
        level = level.upper()
        if level == "G0":
            return [sys.executable, "main.py", "graph-g0", str(self.log_dir), "--output-dir", str(self.graph_dir)]
        if level == "G1":
            return [sys.executable, "main.py", "graph-g1", str(self.log_dir), "--output-dir", str(self.graph_dir)]
        if level == "G2":
            return [sys.executable, "main.py", "graph-g2", str(self.log_dir), "--output-dir", str(self.graph_dir)]
        if level == "G3":
            return [sys.executable, "main.py", "graph-g3", "--g2", str(self.graph_dir / "g2.gexf"), "--output-dir", str(self.graph_dir)]
        raise ValueError(f"Unsupported graph level: {level}")
