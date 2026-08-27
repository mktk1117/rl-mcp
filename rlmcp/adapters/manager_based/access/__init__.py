"""Parameter access: discovery, reads and writes, assembled from providers.

    key = "<domain>.<term>.<path...>"

:class:`ParameterAccess` splits a key into those three parts, hands the term
lookup to the provider that owns the domain, and lets :mod:`.paths` do the rest
by reflection. No branch anywhere in this package knows what a reward weight or
a push velocity is.

To add a family: write a provider (see :mod:`.base`), append it to
:data:`PROVIDER_TYPES`. Discovery, get, set, bounds, the MCP tools and the CLI
all pick it up with no further change.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from rlmcp.adapters.manager_based.access import paths
from rlmcp.adapters.manager_based.access.actions import ActionAccess
from rlmcp.adapters.manager_based.access.base import (
    AccessProvider,
    ProviderRegistry,
    Synthetic,
    Term,
)
from rlmcp.adapters.manager_based.access.commands import CommandAccess
from rlmcp.adapters.manager_based.access.events import EventAccess
from rlmcp.adapters.manager_based.access.rewards import RewardAccess
from rlmcp.adapters.manager_based.access.terminations import TerminationAccess
from rlmcp.core.parameters.spec import Liveness, ParameterSpec

PROVIDER_TYPES: List[type] = [
    RewardAccess,
    TerminationAccess,
    EventAccess,
    CommandAccess,
    ActionAccess,
]

__all__ = [
    "AccessProvider",
    "ActionAccess",
    "CommandAccess",
    "EventAccess",
    "ParameterAccess",
    "PROVIDER_TYPES",
    "RewardAccess",
    "Synthetic",
    "TerminationAccess",
    "Term",
    "paths",
]


class ParameterAccess:
  """The environment's tunable surface, discovered rather than declared."""

  def __init__(self, env: Any, provider_types: Optional[List[type]] = None):
    self.env = env
    self.registry = ProviderRegistry()
    for provider_type in provider_types or PROVIDER_TYPES:
      self.registry.add(provider_type(env))
    self._synthetic: Dict[str, Synthetic] = {}
    for provider in self.registry:
      for item in provider.synthetic():
        self._synthetic[item.key] = item

  # Discovery.

  def discover(self) -> List[ParameterSpec]:
    """Every tunable leaf reachable from every provider's terms."""
    specs: List[ParameterSpec] = []
    for provider in self.registry:
      for term in provider.terms():
        for parts, value in paths.walk_leaves(term.root, skip_keys=provider.skip_keys):
          low, high = provider.bounds(term, parts, value)
          specs.append(
              ParameterSpec(
                  key=(
                      f"{provider.domain}.{term.key}." + paths.join_path(parts)
                  ),
                  data_type=paths.leaf_kind(value),
                  current_value=list(value) if paths.is_range(value) else value,
                  default_value=list(value) if paths.is_range(value) else value,
                  min_value=low,
                  max_value=high,
                  description=provider.describe(term, parts, value),
                  category=provider.category,
                  liveness=provider.liveness(term, parts, value),
              )
          )
    for item in self._synthetic.values():
      specs.append(
          ParameterSpec(
              key=item.key,
              data_type=item.data_type,
              current_value=item.getter(),
              default_value=item.default,
              min_value=item.min_value,
              max_value=item.max_value,
              description=item.description,
              category=self._synthetic_category(item.key),
          )
      )
    return specs

  def _synthetic_category(self, key: str):
    provider = self.registry.get(key.split(".", 1)[0])
    return provider.category if provider else None

  # Resolution.

  def _split(self, key: str) -> Tuple[AccessProvider, Term, List[str]]:
    """Key -> (provider, term, remaining path).

    Matching happens on split segments rather than on the raw string, because a
    segment may legitimately contain an escaped dot. Term keys can also span
    several segments -- events are ``<mode>.<name>`` -- so the longest matching
    term wins.
    """
    segments = paths.split_path(key)
    domain, rest = segments[0], segments[1:]
    provider = self.registry.get(domain)
    if provider is None:
      raise KeyError(
          f"No parameter domain '{domain}'. Available: {self.registry.domains()}"
      )
    candidates = provider.terms()
    matches = [
        (t, t.key.split("."))
        for t in candidates
        if rest[: len(t.key.split("."))] == t.key.split(".")
    ]
    if not matches:
      raise KeyError(
          f"No {domain} term in '{key}'. Available: "
          f"{sorted(t.key for t in candidates)}"
      )
    term, term_parts = max(matches, key=lambda pair: len(pair[1]))
    leaf = rest[len(term_parts) :]
    if not leaf:
      raise KeyError(f"'{key}' names a term but no field inside it.")
    return provider, term, leaf

  # Read and write.

  def get(self, key: str) -> Any:
    item = self._synthetic.get(key)
    if item is not None:
      return item.getter()
    _, term, parts = self._split(key)
    value = paths.resolve(term.root, parts).read()
    return list(value) if paths.is_range(value) else value

  def set(self, key: str, value: Any) -> Dict[str, Any]:
    """Write a parameter. Returns notes from the provider's write hook, if any.

    Refuses writes that cannot take effect this run (liveness 'at_startup' or
    'inert') and values that do not match the leaf's shape -- both before
    anything is mutated, so a rejected call leaves the config exactly as it
    was. A write that lands but only applies later ('at_reset') says so in the
    returned notes.
    """
    item = self._synthetic.get(key)
    if item is not None:
      if not item.setter(value):
        raise RuntimeError(f"Parameter '{key}' could not be applied.")
      return {}

    provider, term, parts = self._split(key)
    resolved = paths.resolve(term.root, parts)
    liveness = provider.liveness(term, parts, resolved.read())
    if liveness is Liveness.AT_STARTUP:
      raise ValueError(
          f"Cannot set '{key}': liveness is 'at_startup'. mjlab applies "
          "startup events exactly once when the environment is constructed, "
          "so this write would report success and then change nothing for the "
          "rest of the run. Change the task config and restart training "
          "instead."
      )
    if liveness is Liveness.INERT:
      raise ValueError(
          f"Cannot set '{key}': liveness is 'inert'. The term's class "
          "instance cached this value when it was constructed and never "
          "re-reads the config, so no write can take effect during this run. "
          "Change the task config and restart training instead."
      )
    try:
      written = resolved.write(value)
    except (TypeError, ValueError) as exc:
      raise type(exc)(f"Cannot set '{key}': {exc}") from exc
    try:
      self._notify_class_term(term.root, parts, written)
    except Exception as exc:
      raise RuntimeError(
          f"Set '{key}': the config value was updated, but rebuilding the "
          f"term's cached state failed ({type(exc).__name__}: {exc}), so the "
          "term may still be using the old value until reset or restart."
      ) from exc
    notes = provider.after_set(term, parts, written) or {}
    if liveness is Liveness.AT_RESET:
      notes.setdefault("liveness", Liveness.AT_RESET.value)
      notes.setdefault(
          "note",
          "Applied to the live config; mjlab re-reads it on episode reset, so "
          "the change takes effect from each environment's next reset.",
      )
    return notes

  @staticmethod
  def _notify_class_term(root: Any, parts: List[str], value: Any) -> None:
    """Give class-based terms a chance to rebuild anything they cached.

    Most mjlab terms are plain functions reading ``params`` every step and need
    nothing. A class-based term whose instance exposes ``update_params`` gets
    it called with the updated top-level param -- the whole ``std`` dict when
    one regex-keyed entry inside it changed, since that is the shape the term
    cached from. A failure here must surface rather than vanish: the config
    write has already landed, and an agent told nothing would keep trusting a
    cache the term failed to rebuild.
    """
    func = getattr(root, "func", None)
    hook = getattr(func, "update_params", None)
    if not callable(hook):
      return
    if len(parts) >= 2 and parts[0] == "params":
      name = parts[1]
      payload = getattr(root, "params", {}).get(name, value)
    else:
      name, payload = parts[-1], value
    hook(**{name: payload})

  # Introspection helpers used by the adapter.

  def domains(self) -> List[str]:
    return self.registry.domains()

  def provider(self, domain: str) -> Optional[AccessProvider]:
    return self.registry.get(domain)
