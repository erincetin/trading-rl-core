from typing import Any


def make_json_safe(obj: Any):
    """
    Convert common non-JSON-safe objects into readable strings.
    """
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, type):
        return obj.__name__
    if callable(obj):
        return obj.__name__
    return obj
