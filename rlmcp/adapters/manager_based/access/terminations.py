"""Termination thresholds -- when an episode is cut short."""

from __future__ import annotations

from typing import Any, List, Sequence

from rlmcp.adapters.access import paths
from rlmcp.adapters.access.base import AccessProvider, Term
from rlmcp.core.parameters.spec import ParameterCategory


class TerminationAccess(AccessProvider):
  domain = "termination"
  category = ParameterCategory.TERMINATION
  manager_attr = "termination_manager"

  # `time_out` tells the algorithm whether ending here is a timeout (bootstrap
  # the value) or a failure (do not). Flipping it live would quietly corrupt
  # returns, and it is not a tuning knob in any case.
  skip_keys = paths.SKIP_KEYS | {"time_out"}

  def terms(self) -> List[Term]:
    return [
        Term(key=name, root=cfg, label=f"termination '{name}'")
        for name, cfg in self._term_cfgs().items()
    ]

  def describe(self, term: Term, parts: Sequence[str], value: Any) -> str:
    return f"Threshold '{'.'.join(parts)}' of termination '{term.key}'"
