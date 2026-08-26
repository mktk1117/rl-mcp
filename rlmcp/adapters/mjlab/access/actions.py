"""Action scaling.

The live ``_scale`` may be a float or a per-joint tensor, so the useful knob is
not the field itself but a gain relative to whatever the task configured. The
baseline is captured the first time the provider sees the term, and the gain
always multiplies that baseline -- so setting 1.0 restores the original exactly,
however many times it has been changed.
"""

from __future__ import annotations

from typing import Any, Dict, List

from rlmcp.adapters.mjlab.access.base import AccessProvider, Synthetic, Term
from rlmcp.core.parameters.spec import ParameterCategory


class ActionAccess(AccessProvider):
  domain = "action"
  category = ParameterCategory.ACTION
  manager_attr = "action_manager"

  def __init__(self, env: Any):
    super().__init__(env)
    self._baseline: Dict[str, Any] = {}

  def terms(self) -> List[Term]:
    # Action term configs hold no independently tunable leaves worth exposing;
    # everything useful goes through the synthetic gain below.
    return []

  def synthetic(self) -> List[Synthetic]:
    manager = self.manager
    out: List[Synthetic] = []
    for name in getattr(manager, "active_terms", []) or []:
      if self._scale_of(name) is None:
        continue
      out.append(
          Synthetic(
              key=f"action.{name}.scale_gain",
              getter=lambda n=name: self.get_gain(n),
              setter=lambda value, n=name: self.set_gain(n, float(value)),
              default=1.0,
              description=(
                  f"Multiplier on the configured action scale of '{name}'. "
                  "Lowering it shrinks the reachable joint-target range, which is "
                  "a direct lever on jerkiness."
              ),
          )
      )
    return out

  # Internals.

  def _term(self, name: str) -> Any:
    try:
      return self.manager.get_term(name)
    except Exception:
      return None

  def _scale_of(self, name: str) -> Any:
    term = self._term(name)
    return getattr(term, "_scale", None) if term is not None else None

  def _baseline_of(self, name: str) -> Any:
    if name not in self._baseline:
      scale = self._scale_of(name)
      if scale is None:
        return None
      self._baseline[name] = scale.clone() if hasattr(scale, "clone") else float(scale)
    return self._baseline[name]

  def get_gain(self, name: str) -> float:
    baseline, scale = self._baseline_of(name), self._scale_of(name)
    if baseline is None or scale is None:
      return 1.0
    if hasattr(scale, "abs"):
      reference = float(baseline.abs().mean().item()) or 1.0
      return round(float(scale.abs().mean().item()) / reference, 6)
    return round(float(scale) / (float(baseline) or 1.0), 6)

  def set_gain(self, name: str, gain: float) -> bool:
    baseline, term = self._baseline_of(name), self._term(name)
    if baseline is None or term is None:
      return False
    term._scale = baseline * gain if hasattr(baseline, "clone") else float(baseline) * gain
    return True
