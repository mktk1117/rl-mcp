"""The extension registry: how capabilities are declared and found.

Registration is a decorator that runs on import: an in-package capability is
the module that implements it plus one import line in the package
``__init__``, with no command table or tool list to edit. Third-party packages
register through an entry point and are found without touching rlmcp at all.

Discovery is deliberately forgiving. An extension whose module fails to import,
whose constructor raises, or whose :meth:`Extension.available` says no, is left
out with a note; it never takes a training run down.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Type

from rlmcp.core.extensions import Extension

ENTRY_POINT_GROUP = "rlmcp.extensions"
"""Entry point group a third-party package uses to publish an extension."""

_REGISTRY: Dict[str, Type[Extension]] = {}
_PLUGINS_LOADED = False


def register(extension_type: Type[Extension]) -> Type[Extension]:
  """Class decorator that makes an extension discoverable.

  The class's ``name`` is its identifier -- in the status payload, in checkpoint
  state, and in ``--include`` / ``--exclude`` filters -- so it has to be set and
  unique.
  """
  name = getattr(extension_type, "name", "")
  if not name:
    raise ValueError(
        f"{extension_type.__name__} needs a non-empty 'name' to be registered."
    )
  existing = _REGISTRY.get(name)
  if existing is not None and existing is not extension_type:
    raise ValueError(
        f"An extension named '{name}' is already registered "
        f"({existing.__module__}.{existing.__name__}). Names must be unique, "
        "since they identify saved state."
    )
  _REGISTRY[name] = extension_type
  return extension_type


def unregister(name: str) -> bool:
  """Remove an extension from the registry. Returns whether it was there."""
  return _REGISTRY.pop(name, None) is not None


def registered() -> Dict[str, Type[Extension]]:
  """Every registered extension type, keyed by name."""
  load_plugins()
  return dict(_REGISTRY)


def catalog() -> List[Dict[str, Any]]:
  """What is installed, for ``rlmcp extensions`` and for documentation."""
  out = []
  for name, extension_type in sorted(registered().items()):
    doc = (extension_type.__doc__ or "").strip().splitlines()
    out.append(
        {
            "name": name,
            "class": f"{extension_type.__module__}.{extension_type.__name__}",
            "summary": doc[0] if doc else "",
        }
    )
  return out


def load_plugins() -> List[str]:
  """Import extensions published by other packages via entry points.

  Runs once. A plugin that fails to import warns and is skipped, because a
  broken third-party package should not stop a training run from starting.
  """
  global _PLUGINS_LOADED
  if _PLUGINS_LOADED:
    return []
  _PLUGINS_LOADED = True

  try:
    from importlib.metadata import entry_points
  except ImportError:  # pragma: no cover - Python < 3.8.
    return []

  loaded: List[str] = []
  try:
    points = entry_points(group=ENTRY_POINT_GROUP)
  except TypeError:  # pragma: no cover - older importlib.metadata API.
    points = entry_points().get(ENTRY_POINT_GROUP, [])  # type: ignore[attr-defined]
  for point in points:
    try:
      target = point.load()
    except Exception as exc:
      warnings.warn(
          f"rlmcp extension plugin '{point.name}' failed to import: {exc}",
          RuntimeWarning,
          stacklevel=2,
      )
      continue
    # Loading the module is usually enough: the decorator registers on import.
    if isinstance(target, type) and issubclass(target, Extension):
      register(target)
    loaded.append(point.name)
  return loaded


def discover(
    env: Any,
    plot_sink: Optional[Callable[..., Any]] = None,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> List[Extension]:
  """Instantiate every registered extension this environment supports.

  Args:
    env: the environment to bind to.
    plot_sink: an artifact writer, passed to extensions that accept one.
      *Deprecated:* constructor probing survives for third-party extensions
      written against it, but the supported way to reach controller
      facilities (artifact writing included) is the
      :class:`~rlmcp.core.extensions.ExtensionContext` the controller hands
      to :meth:`~rlmcp.core.extensions.Extension.bind` after registration.
    include: if given, only these names are considered.
    exclude: names to skip.
  """
  include_set = set(include) if include is not None else None
  exclude_set = set(exclude or ())

  found: List[Extension] = []
  for name, extension_type in sorted(registered().items()):
    if include_set is not None and name not in include_set:
      continue
    if name in exclude_set:
      continue
    extension = _build(extension_type, env, plot_sink)
    if extension is None:
      continue
    try:
      if extension.available():
        found.append(extension)
    except Exception:
      continue
  return found


def _build(
    extension_type: Type[Extension], env: Any, plot_sink: Optional[Callable[..., Any]]
) -> Optional[Extension]:
  """Construct an extension, with or without a plot sink, tolerating failure.

  *Deprecated mechanism:* the ``(env, plot_sink)`` arity probe is kept only so
  third-party extensions written against it keep building. New extensions
  should take ``(env)`` and receive controller facilities through
  :meth:`~rlmcp.core.extensions.Extension.bind`.
  """
  for args in ((env, plot_sink), (env,)):
    try:
      return extension_type(*args)  # type: ignore[arg-type]
    except TypeError:
      continue
    except Exception:
      return None
  return None
