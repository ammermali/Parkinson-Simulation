from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from src.configuration.param_editing import DEFAULT_PARAM_DIR, flatten_parameters, list_param_files, load_yaml, parse_value, read_path, resolve_param_file, set_param_values, validate_value_type

@dataclass(frozen=True)
class ParameterRow:
    key: str
    type: str
    value: str

@dataclass(frozen=True)
class ParameterValidationError:
    key: str
    message: str

class ParameterService:
    def __init__(self, param_dir: Path | str = DEFAULT_PARAM_DIR) -> None:
        self.param_dir = Path(param_dir)

    def available_files(self) -> list[Path]:
        return list_param_files(self.param_dir)

    def resolve(self, param_file: Path | str) -> Path:
        return resolve_param_file(param_file, param_dir=self.param_dir)

    def load_document(self, param_file: Path | str) -> dict[str, Any]:
        return load_yaml(self.resolve(param_file))

    def editable_rows(self, param_file: Path | str) -> list[dict[str, Any]]:
        document = self.load_document(param_file)
        return flatten_parameters(document)

    def parse_and_validate(self, param_file: Path | str, edited_rows: Iterable[dict[str, Any]]) -> tuple[dict[str, Any], list[ParameterValidationError]]:
        document = self.load_document(param_file)
        updates: dict[str, Any] = {}
        errors: list[ParameterValidationError] = []
        for row in edited_rows:
            key = str(row.get("key", "")).strip()
            raw_value = str(row.get("value", ""))
            if not key:
                errors.append(ParameterValidationError(key="", message="Parameter key cannot be empty."))
                continue
            try:
                original_value = read_path(document, key)
                parsed_value = parse_value(raw_value)
                validate_value_type(original_value, parsed_value)
                parsed_value = self.coerce_to_original_type(original_value, parsed_value)
            except Exception as exc:
                errors.append(ParameterValidationError(key=key, message=str(exc)))
                continue
            updates[key] = parsed_value
        return updates, errors

    def save_rows(self, param_file: Path | str, edited_rows: Iterable[dict[str, Any]], *, dry_run: bool = False) -> dict[str, Any]:
        updates, errors = self.parse_and_validate(param_file, edited_rows)
        if errors:
            details = "; ".join(f"{error.key}: {error.message}" for error in errors)
            raise ValueError(details)
        return set_param_values(param_file, updates, param_dir=self.param_dir, create=False, dry_run=dry_run)

    def changed_values(self, param_file: Path | str, updates: dict[str, Any]) -> dict[str, dict[str, Any]]:
        document = self.load_document(param_file)
        changes: dict[str, dict[str, Any]] = {}
        for key, updated_value in updates.items():
            original_value = read_path(document, key)
            if original_value != updated_value:
                changes[key] = {"before": original_value, "after": updated_value}
        return changes

    from typing import Any
    @staticmethod
    def coerce_to_original_type(original: Any, value: Any) -> Any:
        if isinstance(original, bool):
            return value
        if isinstance(original, float):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
        if isinstance(original, int) and not isinstance(original, bool):
            if isinstance(value, int) and not isinstance(value, bool):
                return value
            if isinstance(value, float) and value.is_integer():
                return int(value)
        return value
