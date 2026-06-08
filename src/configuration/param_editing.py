from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_PARAM_DIR = Path(__file__).resolve().parent / "param"


def list_param_files(param_dir: Path | str = DEFAULT_PARAM_DIR) -> list[Path]:
    return sorted(Path(param_dir).glob("*.yaml"))


def get_param_value(param_file: Path | str, key: str, *, param_dir: Path | str = DEFAULT_PARAM_DIR) -> Any:
    data = load_yaml(resolve_param_file(param_file, param_dir))
    return read_path(data, key)


def set_param_value(
    param_file: Path | str,
    key: str,
    value: Any,
    *,
    param_dir: Path | str = DEFAULT_PARAM_DIR,
    create: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Set a nested YAML parameter by dot-separated key and return the new document."""

    path = resolve_param_file(param_file, param_dir)
    data = load_yaml(path)
    updated = copy.deepcopy(data)
    write_path(updated, key, value, create=create)
    if not dry_run:
        dump_yaml(path, updated)
    return updated


def set_param_values(
    param_file: Path | str,
    values: dict[str, Any],
    *,
    param_dir: Path | str = DEFAULT_PARAM_DIR,
    create: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Set multiple dot-separated keys and write the file once."""

    path = resolve_param_file(param_file, param_dir)
    data = load_yaml(path)
    updated = copy.deepcopy(data)
    for key, value in values.items():
        write_path(updated, key, value, create=create)
    if not dry_run:
        dump_yaml(path, updated)
    return updated


def resolve_param_file(param_file: Path | str, param_dir: Path | str = DEFAULT_PARAM_DIR) -> Path:
    """Resolve bare names, filenames and explicit paths to a YAML parameter file."""

    candidate = Path(param_file)
    if candidate.suffix == "":
        candidate = candidate.with_suffix(".yaml")
    if candidate.is_absolute() or candidate.parent != Path("."):
        return candidate
    return Path(param_dir) / candidate


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML mapping."""

    if not path.exists():
        raise FileNotFoundError(f"Parameter file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Parameter file must contain a mapping: {path}")
    return data


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    """Write one YAML mapping atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    payload = yaml.safe_dump(
        data,
        sort_keys=False,
        allow_unicode=True,
    )

    temporary_path.write_text(payload, encoding="utf-8")
    temporary_path.replace(path)

def read_path(data: dict[str, Any], key: str) -> Any:
    """Read one dot-separated nested key."""

    if key in data:
        return data[key]
    current: Any = data
    for part in split_key(key):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Missing parameter key: {key}")
        current = current[part]
    return current


def write_path(data: dict[str, Any], key: str, value: Any, *, create: bool) -> None:
    """Write one dot-separated nested key."""

    if key in data:
        data[key] = value
        return
    parts = split_key(key)
    current: Any = data
    for part in parts[:-1]:
        if not isinstance(current, dict):
            raise TypeError(f"Cannot descend into non-mapping segment: {part}")
        if part not in current:
            if not create:
                raise KeyError(f"Missing parameter key: {'.'.join(parts)}")
            current[part] = {}
        current = current[part]
    if not isinstance(current, dict):
        raise TypeError(f"Cannot set key on non-mapping parent: {key}")
    if parts[-1] not in current and not create:
        raise KeyError(f"Missing parameter key: {key}")
    current[parts[-1]] = value


def split_key(key: str) -> list[str]:
    """Split and validate a dot-separated parameter key."""

    parts = [part for part in key.split(".") if part]
    if not parts:
        raise ValueError("Parameter key cannot be empty.")
    return parts


def iter_leaf_items(data: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    """Return editable dot-path leaves from a nested parameter mapping."""

    leaves: list[tuple[str, Any]] = []
    for key in sorted(data):
        path = f"{prefix}.{key}" if prefix else str(key)
        value = data[key]
        # Nested mappings are navigational groups; scalars, lists and empty
        # mappings are edited as concrete values.
        if isinstance(value, dict) and value:
            leaves.extend(iter_leaf_items(value, path))
        else:
            leaves.append((path, value))
    return leaves


def flatten_parameters(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a tabular representation that Streamlit can display directly."""

    return [
        {
            "key": key,
            "type": type_label(value),
            "value": format_value(value),
        }
        for key, value in iter_leaf_items(data)
    ]


def type_label(value: Any) -> str:
    """Return a stable short label for UI tables and diagnostics."""

    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    if value is None:
        return "null"
    return type(value).__name__


def format_value(value: Any) -> str:
    text = yaml.safe_dump(value, sort_keys=False, allow_unicode=True, default_flow_style=True).strip()
    if text.endswith("\n..."):
        text = text[:-4].rstrip()
    return text


def parse_value(raw_value: str) -> Any:
    return yaml.safe_load(raw_value)


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and edit YAML parameter files.")
    parser.add_argument("--param-dir", type=Path, default=DEFAULT_PARAM_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)
    list_cmd = subparsers.add_parser("list", help="List available parameter files.")
    list_cmd.set_defaults(handler=command_list)
    get_cmd = subparsers.add_parser("get", help="Read one parameter value.")
    get_cmd.add_argument("file", help="Parameter file name, filename or path.")
    get_cmd.add_argument("key", help="Dot-separated key, e.g. logging.output_dir.")
    get_cmd.set_defaults(handler=command_get)
    set_cmd = subparsers.add_parser("set", help="Set one parameter value.")
    set_cmd.add_argument("file", help="Parameter file name, filename or path.")
    set_cmd.add_argument("key", help="Dot-separated key, e.g. logging.progress_stdout.")
    set_cmd.add_argument("value", help="YAML-parsed value, e.g. false, 0.2, '[1, 2]'.")
    set_cmd.add_argument("--create", action="store_true", help="Create missing nested keys.")
    set_cmd.add_argument("--dry-run", action="store_true", help="Print the updated YAML without writing.")
    set_cmd.set_defaults(handler=command_set)
    return parser


def command_list(args: argparse.Namespace) -> int:
    for path in list_param_files(args.param_dir):
        print(path.name)
    return 0


def command_get(args: argparse.Namespace) -> int:
    print(json_text(get_param_value(args.file, args.key, param_dir=args.param_dir)))
    return 0


def command_set(args: argparse.Namespace) -> int:
    path = resolve_param_file(args.file, args.param_dir)
    updated = set_param_value(path, args.key, parse_value(args.value), create=args.create, dry_run=args.dry_run)
    if args.dry_run:
        print(yaml.safe_dump(updated, sort_keys=False, allow_unicode=True))
    else:
        print(path)
    return 0


def validate_value_type(original: Any, updated: Any, *, allow_numeric_conversion: bool = True) -> None:
    if original is None:
        return
    if isinstance(original, bool):
        if not isinstance(updated, bool):
            raise TypeError("Expected a boolean value.")
        return
    if isinstance(original, int) and not isinstance(original, bool):
        if isinstance(updated, int) and not isinstance(updated, bool):
            return
        if allow_numeric_conversion and isinstance(updated, float) and updated.is_integer():
            return
        raise TypeError("Expected an integer value.")
    if isinstance(original, float):
        if isinstance(updated, (int, float)) and not isinstance(updated, bool):
            return
        raise TypeError("Expected a numeric value.")
    if not isinstance(updated, type(original)):
        raise TypeError(
            f"Expected {type_label(original)}, got {type_label(updated)}."
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
