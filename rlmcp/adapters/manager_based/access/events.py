"""Domain randomization and scripted events.

Events are the one family whose terms are not a flat list: the manager groups
them by mode (``startup``, ``reset``, ``interval``), and the same term name can
in principle appear under more than one. Keys are therefore ``<mode>.<name>``,
which the longest-prefix term lookup handles without special casing.
"""

from __future__ import annotations

from typing import Any, List, Sequence

from rlmcp.adapters.manager_based.access import paths
from rlmcp.adapters.manager_based.access.base import AccessProvider, Term
from rlmcp.core.parameters.spec import Liveness, ParameterCategory


class EventAccess(AccessProvider):
  domain = "event"
  category = ParameterCategory.DOMAIN_RANDOMIZATION
  manager_attr = "event_manager"

  # How the interval clock is measured, not how strong the randomization is.
  skip_keys = paths.SKIP_KEYS | {"is_global_time"}

  def terms(self) -> List[Term]:
    manager = self.manager
    out: List[Term] = []
    for mode, names in (getattr(manager, "active_terms", {}) or {}).items():
      mode_name = getattr(mode, "value", str(mode))
      for name in names:
        try:
          cfg = manager.get_term_cfg(name)
        except Exception:
          continue
        out.append(
            Term(
                key=f"{mode_name}.{name}",
                root=cfg,
                label=f"event '{name}' ({mode_name})",
            )
        )
    return out

  def describe(self, term: Term, parts: Sequence[str], value: Any) -> str:
    path = ".".join(parts)
    hint = "; set with a two-element [min, max] list" if isinstance(value, (tuple, list)) else ""
    return f"Randomization '{path}' of {term.label}{hint}"

  def liveness(self, term: Term, parts: Sequence[str], value: Any) -> Liveness:
    """Liveness from the event's mode, which is the first key segment.

    Startup events fire exactly once at environment construction, so their cfg
    is never read again this run. Reset, interval and step events pass
    ``**cfg.params`` to their function at every firing, so those leaves take
    effect at the next reset or the next firing -- unless the term is a class
    instance that cached the param (the base check).
    """
    mode = term.key.split(".", 1)[0]
    if mode == "startup":
      return Liveness.AT_STARTUP
    inherited = super().liveness(term, parts, value)
    if inherited is not Liveness.LIVE:
      return inherited
    if mode == "reset":
      return Liveness.AT_RESET
    return Liveness.LIVE
