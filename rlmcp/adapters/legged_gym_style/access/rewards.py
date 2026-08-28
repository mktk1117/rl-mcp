"""Reward weights, and the parameters the reward functions read.

Two different things live here, because the environment keeps them in two
different places:

* ``reward_scales`` -- one number per reward term, multiplied into that term's
  output every step. These are exposed as ``reward.<term>.weight``, the same
  key a manager-based backend uses, so an agent that has driven mjlab does not
  learn a second vocabulary.
* ``reward_cfg`` -- everything else the reward functions read (``tracking_sigma``,
  ``base_height_target``). Ordinary dict leaves, discovered by walking.

The ``dt`` problem
------------------

``__init__`` does ``reward_scales[name] *= dt`` before training starts, in
place, on the dict ``reward_cfg["reward_scales"]`` is bound to -- so the number
the task config was written with is *gone*, and no inspection recovers it. Left
alone, ``rlmcp params`` would show ``-0.0001`` for a config that said ``-0.005``
and ``rlmcp set`` would take values in units nobody writes.

So the weights are not exposed as dict leaves at all. Each is a synthetic whose
getter divides by ``dt`` and whose setter multiplies, which puts the key back in
the units the config used. When an environment did *not* pre-multiply,
``FlatEnvSpec.scales_premultiplied_by_dt=False`` turns the conversion off and
the same key means the same thing.

What cannot be done here is adding a term. ``reward_functions`` binds
``_reward_<name>`` for each key once, at construction; a name that was not in
the dict then has no function behind it and never will this run.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from rlmcp.adapters.access import paths
from rlmcp.adapters.access.base import AccessProvider, Synthetic, Term
from rlmcp.core.parameters.spec import ParameterCategory


class RewardAccess(AccessProvider):
  """``reward.<term>.weight`` plus the reward functions' own parameters."""

  domain = "reward"
  category = ParameterCategory.REWARD

  def __init__(self, env: Any, spec: Any):
    super().__init__(env)
    self.spec = spec

  # The scales dict, and the dt conversion applied to every value in it.

  @property
  def scales(self) -> Dict[str, float]:
    return self.spec.resolve(self.env, "reward_scales", {}) or {}

  @property
  def _dt(self) -> float:
    """The factor between a configured weight and the stored one.

    1.0 when the environment did not pre-multiply, which makes every
    conversion below a no-op rather than a special case.
    """
    if not self.spec.scales_premultiplied_by_dt:
      return 1.0
    dt = self.spec.resolve(self.env, "dt", None)
    return float(dt) if dt else 1.0

  def available(self) -> bool:
    return bool(self.scales)

  def terms(self) -> List[Term]:
    cfg = self.spec.resolve(self.env, "reward_cfg", None)
    if not isinstance(cfg, dict):
      return []
    return [Term(key="params", root=cfg, label="reward parameters")]

  @property
  def skip_keys(self) -> frozenset:
    """The scales are served as synthetics; walking them too would double them
    up under a second key, in the wrong units."""
    return paths.SKIP_KEYS | {self.spec.reward_scales}

  def synthetic(self) -> List[Synthetic]:
    out: List[Synthetic] = []
    for name in sorted(self.scales):
      out.append(
          Synthetic(
              key=f"{self.domain}.{name}.weight",
              getter=self._weight_getter(name),
              setter=self._weight_setter(name),
              default=self.scales[name] / self._dt,
              description=self._describe_weight(name),
          )
      )
    return out

  def _weight_getter(self, name: str):
    def get() -> float:
      return float(self.scales[name]) / self._dt
    return get

  def _weight_setter(self, name: str):
    def put(value: Any) -> bool:
      try:
        weight = float(value)
      except (TypeError, ValueError):
        raise ValueError(
            f"Reward weight '{name}' takes a number; got {value!r}."
        ) from None
      scales = self.scales
      if name not in scales:
        raise KeyError(
            f"No reward term '{name}'. Terms are bound once, at construction, "
            "from the keys reward_scales had then, so one that was not there "
            f"cannot be added now. Available: {sorted(scales)}."
        )
      scales[name] = weight * self._dt
      return True
    return put

  def _describe_weight(self, name: str) -> str:
    kind = "penalty" if self.scales.get(name, 0.0) < 0 else "reward"
    return (
        f"Weight of reward term '{name}' ({kind}), in the units the task "
        "config uses"
    )

  def miss_hint(self, key: str) -> str:
    if not key.endswith(".weight"):
      return ""
    return (
        "Reward terms are bound once, at construction, from the keys "
        "reward_scales had then -- a term that was not there cannot be added "
        "during this run, only re-weighted."
    )

  def describe(self, term: Term, parts: Sequence[str], value: Any) -> str:
    return (
        f"Parameter '{'.'.join(parts)}' read by the reward functions"
    )
