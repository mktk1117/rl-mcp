"""Reward term weights and their function parameters."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

from rlmcp.adapters.mjlab.access.base import AccessProvider, Term
from rlmcp.core.parameters.spec import ParameterCategory


class RewardAccess(AccessProvider):
  domain = "reward"
  category = ParameterCategory.REWARD
  manager_attr = "reward_manager"

  def terms(self) -> List[Term]:
    return [
        Term(key=name, root=cfg, label=f"reward '{name}'")
        for name, cfg in self._term_cfgs().items()
    ]

  def describe(self, term: Term, parts: Sequence[str], value: Any) -> str:
    if list(parts) == ["weight"]:
      kind = "penalty" if isinstance(value, (int, float)) and value < 0 else "reward"
      func = getattr(term.root, "func", None)
      func_name = getattr(func, "__name__", type(func).__name__ if func else "?")
      return f"Weight of reward term '{term.key}' ({kind}; function {func_name})"
    return f"Parameter '{'.'.join(parts)}' of reward term '{term.key}'"

  def bounds(
      self, term: Term, parts: Sequence[str], value: Any
  ) -> Tuple[Optional[float], Optional[float]]:
    if list(parts) != ["weight"]:
      return None, None
    # Wide enough to zero a term or push it an order of magnitude, narrow enough
    # that a fat-fingered exponent is refused rather than silently applied.
    limit = max(10.0, abs(float(value)) * 20.0)
    return -limit, limit
