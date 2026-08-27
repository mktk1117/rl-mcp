"""Terrain control, as an extension.

Everything here is specific to environments that have a procedural terrain grid
-- which is why it is not in the core. An environment without one (a
manipulation task, a cartpole) simply does not load it, and the terrain commands
are absent rather than broken.

This is the worked example for writing your own: one module, one decorated
class, and the capability reaches the CLI, MCP and curriculum stages with no
change anywhere else. See :mod:`rlmcp.extensions` for the three-step version.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from rlmcp.core.curriculum import Action, Condition, CurriculumStage, StageSchedule
from rlmcp.core.extensions import Extension
from rlmcp.extensions.registry import register

METRIC_TERRAIN_LEVEL_FRAC = "rlmcp/terrain_level_frac"
METRIC_TERRAIN_LEVEL_MEAN = "rlmcp/terrain_level_mean"


@register
class TerrainExtension(Extension):
  """Which terrains environments spawn on, and how hard they are allowed to get."""

  name = "terrain"

  def __init__(self, env: Any, plotter: Optional[Callable[..., bytes]] = None):
    """``plotter`` is the deprecated constructor-probed plot sink; when absent,
    plots go through the :class:`~rlmcp.core.extensions.ExtensionContext`
    artifact writer received in :meth:`~rlmcp.core.extensions.Extension.bind`.
    """
    super().__init__(env)
    # Imported here rather than at module scope to keep the module import cheap
    # and side-effect free; the plan builders below need none of it.
    from rlmcp.adapters.mjlab.state.terrain import TerrainControl

    self.control = TerrainControl(env)
    self._plotter = plotter

  def available(self) -> bool:
    try:
      self.control.grid()
    except Exception:
      return False
    return True

  # Hooks.

  def commands(self) -> Dict[str, Callable[..., Any]]:
    return {
        "terrain_status": self.cmd_terrain_status,
        "set_terrain": self.cmd_set_terrain,
        "plot_terrain": self.cmd_plot_terrain,
    }

  def metrics(self) -> Dict[str, float]:
    return self.control.metrics()

  def select_envs(self, **criteria: Any) -> Optional[List[int]]:
    """Understands ``terrain=<name>`` and ``level=<int>``."""
    terrain = criteria.pop("terrain", None)
    level = criteria.pop("level", None)
    if criteria or (terrain is None and level is None):
      return None  # Not our vocabulary.
    return self.control.env_ids_on(terrain=terrain, level=level)

  def selectors(self) -> Dict[str, Dict[str, Any]]:
    """Terrains that currently hold environments, and the levels in play.

    Only *active* terrains are offered: a terrain configured out spawns nothing,
    so asking for it returns an empty selection, and offering it as a choice is
    offering a dead end. Levels stop at the ceiling in force, which moves when
    the curriculum is capped.
    """
    try:
      status = self.control.status()
    except Exception:
      return {}
    ceiling = int(status.get("level_ceiling") or 0)
    return {
        "terrain": {
            "label": "terrain",
            "values": list(status.get("active_terrains") or []),
        },
        "level": {
            "label": "difficulty level",
            "values": list(range(max(ceiling, 1))),
        },
    }

  def describe(self) -> Dict[str, Any]:
    status = self.control.status()
    return {
        "active_terrains": status["active_terrains"],
        "level_ceiling": status["level_ceiling"],
        "level_mean": status["level_mean"],
        "num_rows_levels": status["num_rows_levels"],
    }

  def snapshot(self) -> Dict[str, Any]:
    return self.control.snapshot()

  def restore(self, state: Dict[str, Any]) -> None:
    self.control.restore(state)

  # Commands.

  def cmd_terrain_status(self) -> Dict[str, Any]:
    """Per-terrain environment counts and difficulty levels reached."""
    return self.control.status()

  def cmd_set_terrain(
      self,
      terrains: Optional[Sequence[str]] = None,
      weights: Optional[Dict[str, float]] = None,
      max_level: Optional[int] = None,
      rationale: str = "",
  ) -> Dict[str, Any]:
    """Choose which terrains spawn environments and cap how hard they get."""
    changed = self.control.configure(
        terrains=terrains, weights=weights, max_level=max_level
    )
    return {"changed": changed, "status": self.control.status(), "rationale": rationale}

  def cmd_plot_terrain(self) -> Dict[str, Any]:
    """Bar chart of environments per terrain and their mean difficulty."""
    sink = self._plotter or (self.context.write_artifact if self.context else None)
    if sink is None:
      raise RuntimeError("No plot sink was attached to the terrain extension.")
    from rlmcp.core.telemetry.plotter import plot_terrain_status

    png = plot_terrain_status(self.control.status())
    return {"image_path": str(sink("terrain", ".png", png))}

  # Plan construction.

  def terrain_names(self) -> List[str]:
    return self.control.names()

  def num_levels(self) -> int:
    return self.control.status()["num_rows_levels"]


DIFFICULTY_GROUPS: List[tuple] = [
    ("flat", ("flat",)),
    ("rough", ("random_rough", "wave_terrain", "perlin_noise")),
    (
        "slopes",
        ("hf_pyramid_slope", "hf_pyramid_slope_inv", "box_random_grid", "tilted_grid"),
    ),
    (
        "obstacles",
        ("discrete_obstacles", "random_spread_boxes", "stepping_stones",
         "narrow_beams", "nested_rings"),
    ),
    (
        "stairs",
        ("pyramid_stairs", "pyramid_stairs_inv", "random_stairs", "open_stairs",
         "easy_stairs", "moderate_stairs", "challenging_stairs"),
    ),
]
"""Terrain names grouped roughly by how much they punish a half-trained gait.

Order matters: earlier groups are unlocked first. Unknown terrain names are
appended as a final group rather than silently dropped.
"""


def group_terrains(
    available: Sequence[str],
    groups: Optional[Sequence[tuple]] = None,
) -> List[tuple]:
  """Bucket the terrains a scene actually contains into difficulty groups."""
  groups = groups or DIFFICULTY_GROUPS
  remaining = list(available)
  out: List[tuple] = []
  for label, names in groups:
    present = [n for n in remaining if n in names]
    if present:
      out.append((label, present))
      remaining = [n for n in remaining if n not in present]
  if remaining:
    out.append(("other", remaining))
  return out


def build_terrain_plan(
    available: Sequence[str],
    num_levels: int,
    level_frac_to_promote: float = 0.6,
    episode_length_frac_to_promote: float = 0.5,
    min_iterations: int = 150,
    hold_iterations: int = 20,
    auto_promote: bool = True,
) -> StageSchedule:
  """Build a flat-first curriculum that unlocks terrain groups one at a time.

  Each stage keeps everything unlocked so far and adds the next group, while the
  difficulty ceiling climbs toward ``num_levels``. Promotion needs both a
  survival signal (episodes running most of their length) and a progress signal
  (the terrain-level curriculum pushing robots up the rows).

  The stages this returns are ordinary :class:`CurriculumStage` objects whose
  ``apply`` lists call ``set_terrain`` -- the core runs them without knowing
  what terrain is.
  """
  from rlmcp.core.curriculum import METRIC_EPISODE_LENGTH_FRAC

  grouped = group_terrains(available)
  if not grouped:
    raise ValueError("No terrains available to build a plan from.")

  num_stages = len(grouped)
  stages: List[CurriculumStage] = []
  unlocked: List[str] = []

  for i, (label, names) in enumerate(grouped):
    unlocked = unlocked + names
    is_last = i == num_stages - 1
    # The ceiling grows across stages; the first stage stays on the easiest rows
    # so a fresh policy is not dropped onto tall stairs.
    ceiling = num_levels if is_last else max(1, round(num_levels * (i + 1) / num_stages))

    promote: List[Condition] = []
    if not is_last:
      promote = [
          Condition(METRIC_EPISODE_LENGTH_FRAC, ">=", episode_length_frac_to_promote),
          Condition(METRIC_TERRAIN_LEVEL_FRAC, ">=", level_frac_to_promote),
      ]

    stages.append(
        CurriculumStage(
            name=f"{i}_{label}",
            apply=[
                Action(
                    "set_terrain",
                    {"terrains": list(unlocked), "max_level": ceiling},
                )
            ],
            promote_when=promote,
            min_iterations=min_iterations,
            hold_iterations=hold_iterations,
            notes=(
                f"Adds {', '.join(names)}; difficulty ceiling {ceiling}/{num_levels}."
                if not is_last
                else f"Full terrain set at ceiling {ceiling}/{num_levels}."
            ),
        )
    )

  return StageSchedule(stages, auto_promote=auto_promote)
