from __future__ import annotations

import importlib
from typing import Any, Mapping


def resolve_factory(reference: str):
    """Resolve ``package.module:callable`` without constraining adapter packages."""

    module_name, separator, qualname = str(reference).partition(":")
    if not separator or not module_name or not qualname:
        raise ValueError("factory must use 'package.module:callable' syntax")
    value: Any = importlib.import_module(module_name)
    for component in qualname.split("."):
        value = getattr(value, component)
    if not callable(value):
        raise TypeError(f"factory is not callable: {reference}")
    return value


def instantiate(specification: Mapping[str, Any]) -> Any:
    unknown = set(specification).difference({"factory", "kwargs"})
    if unknown:
        raise ValueError(f"unknown factory specification fields: {sorted(unknown)}")
    factory = resolve_factory(str(specification["factory"]))
    kwargs = dict(specification.get("kwargs", {}))
    return factory(**kwargs)
