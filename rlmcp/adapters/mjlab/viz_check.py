"""Read a live mjlab scene's colours, and the colours its overlays draw in.

The colour arithmetic is backend-agnostic and lives in
:mod:`rlmcp.core.palette`; this is the part that knows where mjlab keeps
geometry and where command terms keep their visualisation settings.

Marker colours are found by convention: any field on a command term's config
whose name contains ``color`` and whose value looks like an RGB(A) tuple. That
is loose on purpose. The alternative is a registry every task has to remember
to populate, and a check nobody registers with is a check that never fires.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

from rlmcp.core.palette import DEFAULT_MIN_DISTANCE, find_collisions

_COLOR_HINT = "color"
_INVISIBLE_ALPHA = 0.05


def _looks_like_rgba(value: Any) -> bool:
  if not isinstance(value, (tuple, list)) or not 3 <= len(value) <= 4:
    return False
  return all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in value)


def collect_marker_colors(env: Any) -> dict[str, Sequence[float]]:
  """Colours the command terms' debug visualisations draw in."""
  out: dict[str, Sequence[float]] = {}
  manager = getattr(env, "command_manager", None)
  terms = getattr(manager, "_terms", None) if manager is not None else None
  if not terms:
    return out
  for term_name, term in terms.items():
    cfg = getattr(term, "cfg", None)
    if cfg is None or not getattr(cfg, "debug_vis", False):
      continue
    for field in dataclasses.fields(cfg) if dataclasses.is_dataclass(cfg) else ():
      if _COLOR_HINT not in field.name.lower():
        continue
      value = getattr(cfg, field.name, None)
      if _looks_like_rgba(value):
        out[f"{term_name}.{field.name}"] = value
      elif isinstance(value, (tuple, list)):
        for i, item in enumerate(value):
          if _looks_like_rgba(item):
            out[f"{term_name}.{field.name}[{i}]"] = item
  return out


def collect_scene_colors(env: Any) -> dict[str, Sequence[float]]:
  """Colours of the visible geoms in the scene, by geom name.

  Fully transparent geoms are skipped -- they annotate nothing and collide with
  nothing -- as are unnamed ones, which cannot be reported usefully anyway.
  """
  out: dict[str, Sequence[float]] = {}
  model = None
  for path in ("sim.mj_model", "unwrapped.sim.mj_model", "scene.model", "sim.model"):
    obj: Any = env
    for part in path.split("."):
      obj = getattr(obj, part, None)
      if obj is None:
        break
    if obj is not None and hasattr(obj, "ngeom"):
      model = obj
      break
  if model is None:
    return out
  for i in range(int(model.ngeom)):
    try:
      geom = model.geom(i)
      name = geom.name
      rgba = tuple(float(c) for c in geom.rgba)
    except Exception:
      continue
    if not name or rgba[3] <= _INVISIBLE_ALPHA:
      continue
    out[name] = rgba
  return out


def check_marker_colors(
  env: Any, min_distance: float = DEFAULT_MIN_DISTANCE
) -> list[dict]:
  """Debug markers a viewer would mistake for objects in the scene."""
  markers = collect_marker_colors(env)
  if not markers:
    return []
  scene = collect_scene_colors(env)
  if not scene:
    return []
  return find_collisions(markers, scene, min_distance=min_distance)
