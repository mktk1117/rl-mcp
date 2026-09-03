"""Reflective read/write over config objects, addressed by dotted path.

This is the piece that keeps the adapter from growing a hand-written branch per
parameter family. mjlab config objects are ordinary dataclasses and dicts, so a
parameter can be addressed by walking attributes and keys:

    reward.foot_slip.weight
    event.interval.push_robot.params.velocity_range.x
    command.twist.ranges.lin_vel_x

:func:`resolve` walks such a path to the object that *holds* the leaf, and
:func:`walk_leaves` goes the other way -- given a config object, it finds every
tunable leaf inside it. Discovery is therefore a property of the environment,
not of a list somebody remembered to update: when a task adds a reward term or
mjlab adds a field, it shows up on its own.

Nothing here imports mjlab or torch.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

MAX_DEPTH = 4
"""How deep to descend when hunting for leaves. Four covers
``params -> velocity_range -> x`` with room to spare, and stops runaway walks
through object graphs that link back to the scene."""

SKIP_TYPE_NAMES = frozenset(
    {
        "SceneEntityCfg",  # Entity/joint/body selectors, not tunables.
        "MjSpec",
        "Entity",
        "Scene",
        "ManagerBasedRlEnv",
    }
)

SKIP_KEYS = frozenset(
    {
        "func",  # The term's callable.
        "asset_cfg",  # Entity selector; a dataclass, so it would be descended into.
        "sensor_cfg",
        "class_type",
    }
)
"""Attribute names never worth descending into. Values that are merely *not
tunable* -- strings, regex tuples, callables -- need no entry here: the leaf
filter drops them anyway. This list exists for the ones that would otherwise be
walked into."""


# Path grammar.
#
# Segments are joined with dots, so a segment that *contains* a dot has to
# escape it. This is not hypothetical: mjlab keys the G1 pose reward's std dict
# by joint-name regex (``.*knee.*``), and without escaping those parameters are
# discoverable but unaddressable.

_UNESCAPED_DOT = re.compile(r"(?<!\\)\.")


def escape_segment(segment: str) -> str:
  """Escape a single path segment so dots inside it survive a round trip."""
  return str(segment).replace("\\", "\\\\").replace(".", "\\.")


def unescape_segment(segment: str) -> str:
  return segment.replace("\\.", ".").replace("\\\\", "\\")


def join_path(parts: Sequence[str]) -> str:
  """Compose a dotted path, escaping any dots inside the segments."""
  return ".".join(escape_segment(p) for p in parts)


def split_path(key: str) -> list[str]:
  """Split a dotted path on separators only, then unescape each segment."""
  return [unescape_segment(part) for part in _UNESCAPED_DOT.split(key)]


def is_number(value: Any) -> bool:
  """True for real numbers, excluding bool (which is a separate kind of knob)."""
  return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_range(value: Any) -> bool:
  """True for a two-element numeric pair, e.g. ``(0.3, 1.2)``."""
  return (
      isinstance(value, (tuple, list))
      and len(value) == 2
      and all(is_number(v) for v in value)
  )


def is_leaf(value: Any) -> bool:
  """True when a value is something an agent can meaningfully set."""
  return is_number(value) or isinstance(value, bool) or is_range(value)


def leaf_kind(value: Any) -> str:
  if is_range(value):
    return "range"
  if isinstance(value, bool):
    return "bool"
  if isinstance(value, int):
    return "int"
  return "float"


def _as_number(value: Any) -> float:
  """``value`` as a float, or a ValueError saying what a number looks like."""
  if isinstance(value, bool) or not isinstance(value, (int, float, str)):
    raise ValueError(f"expected a number; got {value!r}")
  try:
    return float(value)
  except ValueError:
    raise ValueError(f"expected a number; got {value!r}") from None


def coerce_like(current: Any, value: Any) -> Any:
  """Return ``value`` in whatever container shape ``current`` already uses.

  mjlab reads these fields directly every step, so a range that arrives as a
  JSON list has to go back as the tuple the config declared -- otherwise a term
  that unpacks it keeps working right up until the one that does not. A value
  that does not match the leaf's shape at all is refused here, before anything
  is written: a scalar over a range would not fail at write time but minutes
  later inside ``env.step``, with the config already corrupted.
  """
  if is_range(current):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
      raise ValueError(
          f"expected a two-element [low, high] range like {list(current)}; "
          f"got {value!r}"
      )
    if not all(is_number(v) for v in value):
      raise ValueError(f"range entries must be numbers; got {value!r}")
    low, high = float(value[0]), float(value[1])
    if low > high:
      raise ValueError(
          f"range low {low} exceeds high {high}; expected [low, high] with "
          "low <= high"
      )
    container = type(current) if isinstance(current, (tuple, list)) else tuple
    return container((low, high))
  if isinstance(current, bool):
    if isinstance(value, bool):
      return value
    if isinstance(value, int) and value in (0, 1):
      return bool(value)
    raise ValueError(f"expected true or false; got {value!r}")
  if isinstance(current, int):
    number = _as_number(value)
    if number != int(number):
      raise ValueError(
          f"expected a whole number (the config declares an int); {value!r} "
          f"would be truncated to {int(number)}"
      )
    return int(number)
  if isinstance(current, float):
    return _as_number(value)
  raise TypeError(
      f"current value {current!r} is not a tunable leaf (a number, bool or "
      "[low, high] range); refusing to overwrite it"
  )


@dataclass
class Resolved:
  """Where a path ended up: the container, and the key inside it."""

  parent: Any
  key: Any
  """Attribute name, or the mapping key -- which may not be a string."""
  is_item: bool
  """True when the leaf lives in a mapping, False when it is an attribute."""

  def read(self) -> Any:
    return self.parent[self.key] if self.is_item else getattr(self.parent, self.key)

  def write(self, value: Any) -> Any:
    """Set the leaf, keeping the container type the config already used."""
    coerced = coerce_like(self.read(), value)
    if self.is_item:
      self.parent[self.key] = coerced
    else:
      setattr(self.parent, self.key, coerced)
    return coerced

  def exists(self) -> bool:
    if self.is_item:
      return isinstance(self.parent, Mapping) and self.key in self.parent
    return hasattr(self.parent, str(self.key))


def mapping_key(container: Mapping, key: str) -> Any:
  """The real key inside ``container`` matching the path segment ``key``.

  Path segments are strings, but a config dict may be keyed by something else --
  mjlab's ``base_com`` randomization keys its ranges by integer axis index. The
  segment is matched literally first, then by the value it parses to.

  Returns the key object, or ``_MISSING`` when there is no match.
  """
  if key in container:
    return key
  for parse in (int, float):
    try:
      candidate = parse(key)
    except (TypeError, ValueError):
      continue
    if candidate in container:
      return candidate
  return _MISSING


class _Missing:
  def __repr__(self) -> str:  # pragma: no cover - debugging aid.
    return "<missing>"


_MISSING = _Missing()


def _step(container: Any, key: str) -> tuple[Any, bool]:
  """Take one step into ``container``. Returns (value, came_from_mapping)."""
  if isinstance(container, Mapping):
    actual = mapping_key(container, key)
    if actual is not _MISSING:
      return container[actual], True
  if hasattr(container, key):
    return getattr(container, key), False
  raise KeyError(key)


def resolve(root: Any, parts: Sequence[str]) -> Resolved:
  """Walk ``parts`` from ``root`` and return a handle on the final leaf.

  Raises ``KeyError`` naming the first segment that did not exist, so the
  message points at the typo rather than at the whole path.
  """
  if not parts:
    raise KeyError("(empty path)")
  container = root
  for index, part in enumerate(parts[:-1]):
    try:
      container, _ = _step(container, part)
    except KeyError:
      raise KeyError(join_path(parts[: index + 1])) from None
  last = parts[-1]
  if isinstance(container, Mapping):
    actual = mapping_key(container, last)
    if actual is not _MISSING:
      return Resolved(container, actual, is_item=True)
    if hasattr(container, last):
      return Resolved(container, last, is_item=False)
    raise KeyError(join_path(parts))
  if hasattr(container, last):
    return Resolved(container, last, is_item=False)
  raise KeyError(join_path(parts))


def _descendable(value: Any) -> bool:
  """True when a walk should look inside this value for more leaves."""
  if isinstance(value, MutableMapping):
    return True
  if dataclasses.is_dataclass(value) and not isinstance(value, type):
    return type(value).__name__ not in SKIP_TYPE_NAMES
  return False


def walk_leaves(
    obj: Any,
    prefix: Sequence[str] = (),
    depth: int = 0,
    skip_keys: frozenset = SKIP_KEYS,
) -> Iterator[tuple[list[str], Any]]:
  """Yield ``(path_parts, value)`` for every tunable leaf inside ``obj``.

  Descends through dicts and dataclasses only. Everything that is not a number,
  bool or numeric pair is ignored, which is what keeps entity selectors, regex
  name tuples and callables out of the parameter list without needing a rule
  for each of them.
  """
  if depth >= MAX_DEPTH:
    return

  if isinstance(obj, Mapping):
    items = list(obj.items())
  elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
    items = [(f.name, getattr(obj, f.name, None)) for f in dataclasses.fields(obj)]
  else:
    return

  for raw_key, value in items:
    key = str(raw_key)
    if key.startswith("_") or key in skip_keys:
      continue
    parts = [*prefix, key]
    if is_leaf(value):
      yield parts, value
    elif _descendable(value):
      yield from walk_leaves(value, parts, depth + 1, skip_keys)
