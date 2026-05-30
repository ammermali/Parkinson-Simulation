from pathlib import Path
from typing import Union
import yaml


class Params:
    """Small YAML-backed parameter accessor.

    Names without a directory are resolved inside src/simulation/sim-params.
    Both bare names such as "neuron" and filenames such as "neuron.yaml" are
    accepted, while explicit absolute or relative paths are honored directly.
    """

    DEFAULT_DIR = Path("src/simulation/sim-params")

    def __init__(self, yaml_name: Union[str, Path]):
        self.path = self._resolve_path(yaml_name)
        self.params = self._load()

    @classmethod
    def _resolve_path(cls, yaml_name: Union[str, Path]) -> Path:
        raw_path = Path(yaml_name)
        if raw_path.suffix == "":
            raw_path = raw_path.with_suffix(".yaml")

        if raw_path.is_absolute() or raw_path.parent != Path("."):
            if raw_path.exists():
                return raw_path
            default_candidate = cls.DEFAULT_DIR / raw_path.name
            if default_candidate.exists():
                return default_candidate
            return raw_path
        return cls.DEFAULT_DIR / raw_path

    def _load(self) -> dict:
        """Load the YAML document as a dictionary."""

        if not self.path.exists():
            raise FileNotFoundError(f"YAML file not found: {self.path}")
        with self.path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data or {}

    def get(self, key: str, default=None):
        """Return a top-level value, or default when the key is absent."""

        return self.params.get(key, default)

    def require(self, key: str):
        """Return a required top-level value or raise KeyError."""

        if key not in self.params:
            raise KeyError(f"Missing required parameter: {key}")
        return self.params[key]

    def section(self, key: str) -> dict:
        """Return a required top-level mapping section."""

        value = self.require(key)
        if not isinstance(value, dict):
            raise TypeError(f"Parameter section '{key}' must be a mapping.")
        return value

    def as_dict(self) -> dict:
        """Return a shallow copy of the loaded parameters."""

        return dict(self.params)
