"""What an access provider has to supply.

A provider answers one question -- *which objects in this environment hold
tunable values, and what are they called* -- and then gets out of the way. The
reflective core in :mod:`.paths` does the reading, writing and discovery.

Adding a new family of parameters is therefore a small file: list the terms,
optionally describe them, and register the provider. Nothing in the adapter, the
controller or the MCP server needs to change.
"""

from __future__ import annotations

import functools
import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

from rlmcp.adapters.manager_based.access import paths
from rlmcp.core.parameters.spec import Liveness, ParameterCategory


def is_term_instance(func: Any) -> bool:
  """True for an instantiated class-based term, False for plain functions.

  mjlab's ``_resolve_common_term_cfg`` replaces a class ``func`` with an
  instance at manager construction, so by the time a provider sees a cfg the
  distinction is instance-vs-function.
  """
  if func is None or not callable(func):
    return False
  return not (
      inspect.isfunction(func)
      or inspect.ismethod(func)
      or inspect.isbuiltin(func)
      or inspect.isclass(func)
  )


_CFG_READ = re.compile(r"\bcfg\.((?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*)")


@functools.lru_cache(maxsize=None)
def constructor_cfg_reads(cls: type) -> FrozenSet[str]:
  """Dotted ``cfg`` paths the constructors of ``cls`` read, from their source.

  A class-based term is handed its cfg once, at construction, and whatever its
  ``__init__`` derives from a cfg field -- a kernel built from a decay rate, a
  buffer sized from a count -- keeps the value it was built with after the cfg
  is edited. The attribute-name check in :meth:`AccessProvider.liveness` cannot
  see that kind of cache: mjlab's ``MotionCommand`` stores its sampling kernel
  as ``kernel``, not as ``adaptive_lambda``. The constructor's source can.

  Every ``__init__`` along the MRO is scanned, since a base class may cache on
  a subclass's behalf. Paths are matched whole and dotted, so a constructor
  that only validates ``cfg.ranges.heading`` does not make ``ranges.lin_vel_x``
  inert. Source that cannot be read (built-ins, extension modules) contributes
  nothing, which leaves the default of live.
  """
  reads: set = set()
  for klass in cls.__mro__:
    init = klass.__dict__.get("__init__")
    if init is None:
      continue
    try:
      source = inspect.getsource(init)
    except (OSError, TypeError):
      continue
    reads.update(_CFG_READ.findall(source))
  return frozenset(reads)


@dataclass
class Term:
  """One addressable config object and the name it is addressed by.

  ``key`` may contain dots -- events are addressed as ``<mode>.<name>`` -- which
  is why term lookup matches the longest key rather than a single segment.

  ``instance`` is the live term object when the config belongs to one (command
  terms are always class instances holding their cfg), so a provider can ask
  it about liveness and hand it ``update_params`` writes. Function-based terms
  leave it ``None`` and are reached through ``root.func`` instead.
  """

  key: str
  root: Any
  label: str = ""
  instance: Any = None


@dataclass
class Synthetic:
  """A parameter with no config field behind it.

  Action scale is the motivating case: what an agent wants is "make all the
  action scales 30% smaller", and the useful knob is a gain relative to the
  configured value rather than the per-joint tensor itself.
  """

  key: str
  getter: Callable[[], Any]
  setter: Callable[[Any], bool]
  default: Any
  description: str
  data_type: str = "float"
  min_value: Optional[float] = None
  max_value: Optional[float] = None


class AccessProvider:
  """Base class for one family of parameters (rewards, events, ...)."""

  domain: str = ""
  category: ParameterCategory = ParameterCategory.OTHER
  manager_attr: str = ""
  """Attribute on the env holding this family's manager, if any."""

  skip_keys: frozenset = paths.SKIP_KEYS
  """Fields this family should not expose. Override to hide knobs that change
  *semantics* rather than magnitude -- a flag the RL algorithm interprets, or a
  visualisation setting -- since those are traps rather than tuning surface."""

  def __init__(self, env: Any):
    self.env = env

  # Availability.

  @property
  def manager(self) -> Any:
    return getattr(self.env, self.manager_attr, None) if self.manager_attr else None

  def available(self) -> bool:
    """False when this environment has no such manager; the provider is skipped."""
    return self.manager is not None

  # The one method a provider must implement.

  def terms(self) -> List[Term]:
    raise NotImplementedError

  # Optional annotation, so discovered keys arrive with useful metadata.

  def describe(self, term: Term, parts: Sequence[str], value: Any) -> str:
    path = ".".join(parts)
    subject = term.label or f"'{term.key}'"
    return f"{path} of {self.domain} term {subject}"

  def bounds(
      self, term: Term, parts: Sequence[str], value: Any
  ) -> Tuple[Optional[float], Optional[float]]:
    return None, None

  def liveness(self, term: Term, parts: Sequence[str], value: Any) -> Liveness:
    """When a write to this leaf actually reaches the running simulation.

    mjlab managers re-read ``cfg.params`` (and ``weight``) on every call, so
    leaves default to live. The exception is a class-based term: mjlab
    instantiates a class ``func`` once with its cfg, and the documented pattern
    is to cache a param on the instance at construction and ignore the config
    copy from then on -- such a leaf is inert, unless the instance exposes an
    ``update_params`` hook that lets a write rebuild the cache.

    Caching is detected by the instance holding an attribute named after the
    top-level param (or its underscore twin): every class-based term in mjlab
    follows that convention (``posture.std``, ``variable_posture.std_*``,
    ``apply_body_impulse._cooldown_s``), and the params those terms *do* read
    from each call have no matching attribute.
    """
    if len(parts) < 2 or parts[0] != "params":
      return Liveness.LIVE
    func = getattr(term.root, "func", None)
    if not is_term_instance(func):
      return Liveness.LIVE
    if callable(getattr(func, "update_params", None)):
      return Liveness.LIVE
    name = parts[1]
    if hasattr(func, name) or hasattr(func, f"_{name}"):
      return Liveness.INERT
    return Liveness.LIVE

  def synthetic(self) -> List[Synthetic]:
    """Parameters this family exposes that are not plain config fields."""
    return []

  # Optional write hook, for families where setting a field is not the whole job.

  def after_set(
      self, term: Term, parts: Sequence[str], value: Any
  ) -> Optional[Dict[str, Any]]:
    """Run after a successful write. Return notes to surface to the caller."""
    return None

  # Helpers shared by several providers.

  def _term_cfgs(self) -> Dict[str, Any]:
    """``{term_name: cfg}`` for managers with the usual mjlab shape."""
    manager = self.manager
    if manager is None:
      return {}
    out: Dict[str, Any] = {}
    for name in getattr(manager, "active_terms", []) or []:
      try:
        out[name] = manager.get_term_cfg(name)
      except Exception:
        continue
    return out


@dataclass
class ProviderRegistry:
  """The providers active for one environment."""

  providers: Dict[str, AccessProvider] = field(default_factory=dict)

  def add(self, provider: AccessProvider) -> None:
    if provider.available():
      self.providers[provider.domain] = provider

  def get(self, domain: str) -> Optional[AccessProvider]:
    return self.providers.get(domain)

  def domains(self) -> List[str]:
    return sorted(self.providers)

  def __iter__(self):
    return iter(self.providers.values())
