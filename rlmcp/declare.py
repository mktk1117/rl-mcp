"""Markers a single-file environment uses to declare its tunable surface.

A single-file environment keeps every parameter in one dataclass at the top of
``env.py``, and rlmcp reads that dataclass directly -- no manager, no registry
call, no base class to inherit from. Two things the dataclass cannot say on
its own are said with the markers here:

* :class:`Static` -- a value the environment reads once, at construction.
  Writing it during a run would change nothing, so rlmcp lists it with
  liveness ``at_startup`` and refuses the write with that reason instead of
  reporting success::

      num_envs: Static[int] = 4096

* :class:`Term` -- one reward term: a weight, the parameters its function
  reads, and optionally the function itself. Declared with :func:`term` so
  every config instance gets its own copy::

      @dataclass
      class Rewards:
        tracking_lin_vel: Term = term(4.0, sigma=0.5)
        action_rate: Term = term(-0.1)

  The environment's reward loop reads the table with :func:`terms`, which is
  also how a term rlmcp appends mid-run (``rlmcp add-reward``) is picked up:
  a term that carries a ``func`` is scored by calling ``func(env, **params)``.

Nothing here imports anything outside the standard library, so ``env.py``
stays runnable in a bare interpreter, and the two markers can be vendored
into a project that would rather not depend on rlmcp at all -- detection
below is by shape (a ``Term`` is anything with ``weight`` and ``params``;
``Static`` is the string :data:`STATIC` inside ``typing.Annotated``), not by
identity.
"""

from __future__ import annotations

import dataclasses
import typing
from collections.abc import Callable, Iterator
from typing import Annotated, Any

STATIC = "rlmcp:static"
"""The ``Annotated`` metadata that marks a field as read once at construction."""


class Static:
  """``Static[int]`` is ``Annotated[int, STATIC]``: a construction-time value.

  Mark the fields the environment copies into tensors, sizes or buffers in
  ``__init__`` -- ``num_envs``, ``dt``, ``decimation``, the asset path, the
  default pose. Everything unmarked is taken to be live: re-read on every step
  or resample, so a write applies from the next one.
  """

  def __class_getitem__(cls, item: Any) -> Any:
    return Annotated[item, STATIC]


class Term:
  """One reward term: its weight, its parameters and, optionally, its function.

  ``params`` are read back as attributes too, so a reward loop can write
  ``term.sigma`` where the config wrote ``term(4.0, sigma=0.5)``. ``func`` is
  ``None`` for a term the environment computes inline, and
  ``func(env, **params) -> tensor[num_envs]`` for one that was added at
  runtime; the environment's loop is expected to call it.
  """

  __slots__ = ("func", "params", "weight")

  def __init__(
      self, weight: float, func: Callable[..., Any] | None = None, **params: Any
  ):
    self.weight = weight
    self.func = func
    self.params: dict[str, Any] = dict(params)

  def __getattr__(self, name: str) -> Any:
    # Only reached when normal lookup fails, i.e. for a parameter name.
    params = object.__getattribute__(self, "params")
    try:
      return params[name]
    except KeyError:
      raise AttributeError(
          f"Term has no parameter '{name}'; it has {sorted(params)}."
      ) from None

  def __repr__(self) -> str:
    inner = [f"weight={self.weight!r}"]
    if self.func is not None:
      inner.append(f"func={getattr(self.func, '__name__', self.func)!r}")
    inner.extend(f"{k}={v!r}" for k, v in self.params.items())
    return f"Term({', '.join(inner)})"


def term(weight: float, func: Callable[..., Any] | None = None, **params: Any) -> Any:
  """A dataclass field holding a fresh :class:`Term` per config instance.

  A bare ``Term(...)`` as a default would be shared by every ``EnvConfig()``
  ever built, so a weight rlmcp changes on one run would leak into the next
  config constructed in the same process. This is ``field(default_factory=...)``
  spelled for the thing being declared.
  """
  return dataclasses.field(default_factory=lambda: Term(weight, func, **params))


def is_term(value: Any) -> bool:
  """True for a :class:`Term` or anything shaped like one."""
  return (
      isinstance(value, Term)
      or (hasattr(value, "weight") and isinstance(getattr(value, "params", None), dict))
  )


def terms(group: Any) -> dict[str, Term]:
  """``{name: term}`` for every term on ``group``, in declaration order.

  Read off the instance rather than the dataclass fields, so a term added at
  runtime with ``setattr`` is in the table the moment it lands.
  """
  if group is None:
    return {}
  try:
    items = vars(group).items()
  except TypeError:
    return {}
  return {name: value for name, value in items if is_term(value)}


# Static-ness, asked of the dataclass that declares the field.


def _annotations(cls: type) -> dict[str, Any]:
  """Resolved annotations for ``cls``, with ``Annotated`` metadata kept.

  ``from __future__ import annotations`` turns every annotation into a string,
  so they have to be evaluated to see the metadata. A module that cannot be
  resolved (a forward reference to something not importable here) falls back
  to a textual check, which is enough to recognise ``Static[...]``.
  """
  try:
    return typing.get_type_hints(cls, include_extras=True)
  except Exception:
    raw = {}
    for klass in reversed(cls.__mro__):
      raw.update(getattr(klass, "__annotations__", {}) or {})
    return raw


def _is_static_annotation(annotation: Any) -> bool:
  if isinstance(annotation, str):
    return annotation.replace(" ", "").startswith("Static[")
  if typing.get_origin(annotation) is Annotated:
    return STATIC in typing.get_args(annotation)[1:]
  return False


def is_static(owner: Any, name: str) -> bool:
  """Whether field ``name`` of ``owner`` (an instance or class) is ``Static``."""
  cls = owner if isinstance(owner, type) else type(owner)
  return _is_static_annotation(_annotations(cls).get(name))


def static_fields(owner: Any) -> set[str]:
  """The names of ``owner``'s fields that are marked ``Static``."""
  cls = owner if isinstance(owner, type) else type(owner)
  if not dataclasses.is_dataclass(cls):
    return set()
  return {f.name for f in dataclasses.fields(cls) if is_static(cls, f.name)}


def walk(cfg: Any, prefix: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], Any, bool]]:
  """Yield ``(path, value, static)`` for every field under ``cfg``.

  Descends into nested dataclasses; a nested dataclass marked ``Static`` makes
  everything under it static. Terms are yielded whole rather than descended
  into, since their weight and params are served by name elsewhere.
  """
  if not dataclasses.is_dataclass(cfg) or isinstance(cfg, type):
    return
  statics = static_fields(cfg)
  for f in dataclasses.fields(cfg):
    value = getattr(cfg, f.name, None)
    path = (*prefix, f.name)
    static = f.name in statics
    if dataclasses.is_dataclass(value) and not isinstance(value, type) and not is_term(value):
      if static:
        for inner_path, inner_value, _ in walk(value, path):
          yield inner_path, inner_value, True
      else:
        yield from walk(value, path)
    else:
      yield path, value, static


__all__ = [
    "STATIC",
    "Static",
    "Term",
    "is_static",
    "is_term",
    "static_fields",
    "term",
    "terms",
    "walk",
]
