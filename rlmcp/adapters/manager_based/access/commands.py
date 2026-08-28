"""Velocity command sampling ranges.

Writing here is not just a field assignment. A task can drive the same ranges
from its own curriculum term (mjlab's ``commands_vel`` re-applies a velocity
stage table on every environment reset), which would silently undo the edit
within a step. The write hook rewrites that table too, and reports that it did.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from rlmcp.adapters.access import paths
from rlmcp.adapters.access.base import (
    AccessProvider,
    Term,
    constructor_cfg_reads,
)
from rlmcp.core.parameters.spec import Liveness, ParameterCategory


class CommandAccess(AccessProvider):
  domain = "command"
  category = ParameterCategory.CURRICULUM
  manager_attr = "command_manager"

  # Debug drawing settings; they change the picture, not the task.
  skip_keys = paths.SKIP_KEYS | {"viz", "debug_vis"}

  def available(self) -> bool:
    manager = self.manager
    return manager is not None and bool(getattr(manager, "active_terms", None))

  def terms(self) -> List[Term]:
    manager = self.manager
    out: List[Term] = []
    for name in getattr(manager, "active_terms", []) or []:
      try:
        instance = manager.get_term(name)
        cfg = instance.cfg
      except Exception:
        continue
      out.append(
          Term(key=name, root=cfg, label=f"command '{name}'", instance=instance)
      )
    return out

  def describe(self, term: Term, parts: Sequence[str], value: Any) -> str:
    return f"Command '{term.key}' sampling range for {'.'.join(parts)}"

  def liveness(self, term: Term, parts: Sequence[str], value: Any) -> Liveness:
    """Live unless the term's constructor read this field.

    A command term is a class instance handed its cfg once. Fields it reads
    while running -- ``ranges`` at every resample, ``resampling_time_range``
    from the base class -- are live. Fields its ``__init__`` read were folded
    into whatever it built there and are never consulted again: mjlab's
    ``MotionCommand`` bakes ``adaptive_lambda`` and ``adaptive_kernel_size``
    into a sampling kernel at construction, so an edit to either reports
    success and leaves the kernel as it was (and, for the size, desynchronises
    the padding it *does* read from the kernel it does not rebuild). Those are
    inert, and the write is refused rather than faked.

    A term that can rebuild its cache says so by exposing ``update_params``,
    which keeps every field live; the write hook calls it with the updated
    top-level field.
    """
    instance = term.instance
    if instance is None:
      return Liveness.LIVE
    if callable(getattr(instance, "update_params", None)):
      return Liveness.LIVE
    reads = constructor_cfg_reads(type(instance))
    for depth in range(1, len(parts) + 1):
      if ".".join(parts[:depth]) in reads:
        return Liveness.INERT
    return Liveness.LIVE

  # Curriculum ownership.

  def curriculum_stages(self, command: str) -> List[tuple]:
    """Curriculum terms that rewrite this command's ranges on every reset."""
    manager = getattr(self.env, "curriculum_manager", None)
    out: List[tuple] = []
    for name in getattr(manager, "active_terms", []) or []:
      try:
        params = getattr(manager.get_term_cfg(name), "params", None) or {}
      except Exception:
        continue
      stages = params.get("velocity_stages")
      if not isinstance(stages, list) or not stages:
        continue
      owner = params.get("command_name")
      if owner is not None and owner != command:
        continue
      out.append((name, stages))
    return out

  def claim_from_curriculum(
      self, command: str, field: str, bounds: tuple
  ) -> List[str]:
    """Rewrite any curriculum stage table that owns ``field``. Returns term names."""
    conflicts: List[str] = []
    for name, stages in self.curriculum_stages(command):
      if not any(field in stage for stage in stages):
        continue
      conflicts.append(name)
      for stage in stages:
        stage[field] = bounds
    return conflicts

  def after_set(
      self, term: Term, parts: Sequence[str], value: Any
  ) -> Optional[Dict[str, Any]]:
    if len(parts) < 2 or parts[0] != "ranges":
      return None
    field = parts[-1]
    conflicts = self.claim_from_curriculum(term.key, field, tuple(value))
    if not conflicts:
      return None
    return {
        "curriculum_overridden": sorted(conflicts),
        "note": (
            f"The task's own curriculum term(s) {sorted(conflicts)} also drive these "
            "ranges; their stage table was rewritten so this change persists."
        ),
    }
