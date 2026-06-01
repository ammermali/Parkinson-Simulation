def clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    """Clamp a numeric value between min_value and max_value."""

    return max(min_value, min(value, max_value))
