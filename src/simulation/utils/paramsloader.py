from pathlib import Path
import yaml


class Params:
    def __init__(self, yaml_name: str):
        self.path = Path("src/simulation/sim-params"+yaml_name)
        self.params = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            raise FileNotFoundError(f"YAML file not found: {self.path}")
        with self.path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data or {}

    def get(self, key: str, default=None):
        return self.params.get(key, default)

    def require(self, key: str):
        if key not in self.params:
            raise KeyError(f"Missing required parameter: {key}")
        return self.params[key]

    def as_dict(self) -> dict:
        return dict(self.params)