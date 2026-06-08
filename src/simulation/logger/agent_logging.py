from __future__ import annotations

from enum import Enum
from typing import Any, Optional


LOGGER_METHODS = ("record_event", "state_transition", "field_effect", "internal_field_effect", "degradation", "aggregation")
LOGGER_ATTRS = ("event_logger", "causal_logger", "logger")
REPORTER_ATTRS = ("reporter", "runtime_reporter")

__all__ = ["bind_causal_logger", "bind_event_logger", "causal_logger_from", "event_logger_from", "uid_of", "value_of"]


def bind_event_logger(agent, model):
    logger = event_logger_from(model)
    _set_if_possible(agent, "causal_logger", logger)
    _set_if_possible(agent, "event_logger", logger)
    return logger


def bind_causal_logger(agent, model):
    return bind_event_logger(agent, model)


def event_logger_from(source):
    if source is None:
        return None
    for attr in LOGGER_ATTRS:
        logger = getattr(source, attr, None)
        if _is_logger_like(logger):
            return logger
    for attr in REPORTER_ATTRS:
        logger = event_logger_from(getattr(source, attr, None))
        if logger is not None:
            return logger
    if _is_logger_like(source):
        return source
    return None


def causal_logger_from(source):
    return event_logger_from(source)


def _set_if_possible(target, attr: str, value) -> None:
    try:
        setattr(target, attr, value)
    except Exception:
        return


def _is_logger_like(value) -> bool:
    if value is None:
        return False
    return any(callable(getattr(value, method, None)) for method in LOGGER_METHODS)


def uid_of(agent) -> Optional[str]:
    if agent is None:
        return None
    uid = getattr(agent, "uid", None)
    if uid is not None:
        if isinstance(uid, tuple):
            return ":".join(str(item) for item in uid)
        return str(uid)
    local_id = getattr(agent, "local_id", getattr(agent, "id", ""))
    ptype = getattr(agent, "ptype", getattr(agent, "type_id", ""))
    rank = getattr(agent, "rank", "")
    if local_id == "" and ptype == "" and rank == "":
        return None
    return f"{local_id}:{ptype}:{rank}"


def value_of(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)
