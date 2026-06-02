from __future__ import annotations
from collections.abc import Mapping
from typing import Any

def get_param(params: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if key in params:
        return params[key]
    parts = key.split(".")
    for split_at in range(len(parts), 0, -1):
        prefix = ".".join(parts[:split_at])
        if prefix not in params:
            continue
        value = params[prefix]
        for part in parts[split_at:]:
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value
    value: Any = params
    for part in parts:
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value