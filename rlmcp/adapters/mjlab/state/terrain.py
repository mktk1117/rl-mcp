"""The terrain grid: reading it, and steering who stands where.

mjlab lays procedural terrain out as a grid -- one column per terrain type, one
row per difficulty level -- and every environment holds a
``(terrain_level, terrain_type)`` pair naming its spawn patch. Curriculum
control is therefore two independent knobs:

* which columns environments may occupy  -> which terrains they face,
* how far down the rows they may go      -> how hard those terrains get.

Both are applied by rewriting the per-env assignment tensors and recomputing
origins. Robots move to their new patch on their next episode reset, which is
also the only point at which moving them is safe.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from rlmcp.adapters.base import NotSupported


class TerrainControl:
  """Read and steer the terrain assignment of a live scene."""

  def __init__(self, env: Any):
    self.env = env

  @property
  def terrain(self) -> Any:
    return getattr(self.env.scene, "terrain", None)

  # Grid.

  def grid(self) -> Tuple[Any, List[str], int, int]:
    """(terrain entity, terrain names, num_rows, num_cols)."""
    terrain = self.terrain
    if terrain is None or getattr(terrain, "terrain_origins", None) is None:
      raise NotSupported(
          "This environment has no procedural terrain grid "
          "(terrain_type is not 'generator'), so terrain curriculum control "
          "does not apply."
      )
    generator_cfg = terrain.cfg.terrain_generator
    names = list(generator_cfg.sub_terrains.keys()) if generator_cfg else []
    num_rows, num_cols = terrain.terrain_origins.shape[:2]
    if len(names) != num_cols:
      # Random (non-curriculum) layouts do not map columns to terrain names.
      names = [f"col_{i}" for i in range(num_cols)]
    return terrain, names, int(num_rows), int(num_cols)

  def names(self) -> List[str]:
    _, names, _, _ = self.grid()
    return names

  # Reading.

  def status(self) -> Dict[str, Any]:
    terrain, names, num_rows, num_cols = self.grid()
    types = terrain.terrain_types.detach().cpu().numpy()
    levels = terrain.terrain_levels.detach().cpu().numpy()
    ceiling = int(getattr(terrain, "max_terrain_level", num_rows))

    per_terrain = []
    for i, name in enumerate(names):
      mask = types == i
      count = int(mask.sum())
      entry: Dict[str, Any] = {"terrain": name, "column": i, "num_envs": count}
      if count:
        entry["level_mean"] = round(float(levels[mask].mean()), 3)
        entry["level_max"] = int(levels[mask].max())
        entry["level_histogram"] = np.bincount(levels[mask], minlength=num_rows).tolist()
      per_terrain.append(entry)

    return {
        "num_rows_levels": num_rows,
        "num_cols_terrains": num_cols,
        "level_ceiling": ceiling,
        "all_terrains": names,
        "active_terrains": [e["terrain"] for e in per_terrain if e["num_envs"] > 0],
        "level_mean": round(float(levels.mean()), 3),
        "level_max": int(levels.max()),
        "level_frac": round(float(levels.mean()) / max(1, ceiling - 1), 3),
        "per_terrain": per_terrain,
    }

  def metrics(self) -> Dict[str, float]:
    """Terrain progress as ``rlmcp/`` scalars, for curriculum promotion rules."""
    try:
      terrain, _, num_rows, _ = self.grid()
    except NotSupported:
      return {}
    levels = terrain.terrain_levels.float()
    ceiling = int(getattr(terrain, "max_terrain_level", num_rows))
    mean = float(torch.mean(levels).item())
    return {
        "rlmcp/terrain_level_mean": round(mean, 4),
        "rlmcp/terrain_level_max": float(torch.max(levels).item()),
        "rlmcp/terrain_level_frac": round(mean / max(1, ceiling - 1), 4),
        "rlmcp/terrain_level_ceiling": float(ceiling),
    }

  def env_ids_on(
      self, terrain: Optional[str] = None, level: Optional[int] = None
  ) -> List[int]:
    """Environment indices currently spawned on a given terrain / level."""
    entity, names, _, _ = self.grid()
    types = entity.terrain_types.detach().cpu().numpy()
    levels = entity.terrain_levels.detach().cpu().numpy()
    mask = np.ones_like(types, dtype=bool)
    if terrain is not None:
      if terrain not in names:
        raise ValueError(f"Unknown terrain '{terrain}'. Available: {names}")
      mask &= types == names.index(terrain)
    if level is not None:
      mask &= levels == int(level)
    return [int(i) for i in np.nonzero(mask)[0]]

  # Steering.

  def configure(
      self,
      terrains: Optional[Sequence[str]] = None,
      weights: Optional[Dict[str, float]] = None,
      max_level: Optional[int] = None,
  ) -> Dict[str, Any]:
    """Restrict which terrains and difficulty levels environments may occupy."""
    terrain, names, num_rows, _ = self.grid()
    device = terrain.terrain_levels.device
    changed: Dict[str, Any] = {}

    if max_level is not None:
      ceiling = int(max(1, min(int(max_level), num_rows)))
      terrain.max_terrain_level = ceiling
      terrain.terrain_levels.clamp_(max=ceiling - 1)
      changed["level_ceiling"] = ceiling

    if terrains is not None or weights is not None:
      requested = list(terrains) if terrains is not None else list(names)
      unknown = [t for t in requested if t not in names]
      if unknown:
        raise ValueError(f"Unknown terrain(s) {unknown}. This scene has: {names}")
      columns = [names.index(t) for t in requested]
      if not columns:
        raise ValueError("At least one terrain must stay enabled.")

      counts = self._env_counts(columns, names, weights)
      assignment = np.repeat(np.array(columns, dtype=np.int64), counts)
      np.random.default_rng().shuffle(assignment)
      terrain.terrain_types = torch.from_numpy(assignment).to(
          device=device, dtype=terrain.terrain_types.dtype
      )
      changed["terrains"] = requested
      changed["envs_per_terrain"] = {
          names[c]: int(n) for c, n in zip(columns, counts)
      }

    # Recompute spawn origins so the new assignment takes effect on reset.
    terrain.terrain_levels.clamp_(
        min=0, max=int(getattr(terrain, "max_terrain_level", num_rows)) - 1
    )
    terrain.env_origins[:] = terrain.terrain_origins[
        terrain.terrain_levels, terrain.terrain_types
    ]
    changed["applied"] = True
    return changed

  def _env_counts(
      self,
      columns: Sequence[int],
      names: Sequence[str],
      weights: Optional[Dict[str, float]],
  ) -> np.ndarray:
    """Split the environments across ``columns`` by weight, exactly."""
    weight_values = np.array(
        [float((weights or {}).get(names[c], 1.0)) for c in columns], dtype=np.float64
    )
    if np.any(weight_values < 0):
      raise ValueError("Terrain weights must be non-negative.")
    if weight_values.sum() <= 0:
      weight_values = np.ones_like(weight_values)
    weight_values /= weight_values.sum()

    num_envs = int(self.env.num_envs)
    counts = np.floor(weight_values * num_envs).astype(int)
    if num_envs >= len(columns):
      counts = np.maximum(counts, 1)  # Every enabled terrain gets someone.
    while counts.sum() > num_envs:
      counts[int(np.argmax(counts))] -= 1
    leftover = num_envs - counts.sum()
    if leftover > 0:
      order = np.argsort(-weight_values)
      for i in range(leftover):
        counts[order[i % len(order)]] += 1
    return counts

  # Checkpointing.

  def snapshot(self) -> Dict[str, Any]:
    terrain = self.terrain
    if terrain is None or getattr(terrain, "terrain_origins", None) is None:
      return {}
    return {
        "terrain_levels": terrain.terrain_levels.detach().cpu().tolist(),
        "terrain_types": terrain.terrain_types.detach().cpu().tolist(),
        "max_terrain_level": int(getattr(terrain, "max_terrain_level", 0)),
    }

  def restore(self, state: Dict[str, Any]) -> None:
    terrain = self.terrain
    if terrain is None or getattr(terrain, "terrain_origins", None) is None:
      return
    if state.get("max_terrain_level"):
      terrain.max_terrain_level = int(state["max_terrain_level"])
    device = terrain.terrain_levels.device
    for field in ("terrain_levels", "terrain_types"):
      values = state.get(field)
      if values is None:
        continue
      tensor = torch.tensor(values, device=device, dtype=terrain.terrain_levels.dtype)
      if tensor.shape[0] != terrain.terrain_levels.shape[0]:
        continue  # Different num_envs; keep the current layout.
      getattr(terrain, field)[:] = tensor
    terrain.env_origins[:] = terrain.terrain_origins[
        terrain.terrain_levels, terrain.terrain_types
    ]
