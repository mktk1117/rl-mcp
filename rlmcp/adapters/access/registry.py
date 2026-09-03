"""The environment's tunable surface, assembled from providers.

    key = "<domain>.<term>.<path...>"

:class:`ParameterAccess` splits a key into those three parts, hands the term
lookup to the provider that owns the domain, and lets :mod:`.paths` do the rest
by reflection. No branch anywhere in here knows what a reward weight or a push
velocity is -- which is why it serves any family of environment, not just the
manager-based ones it was first written for. What a family supplies is a list
of providers; see :mod:`rlmcp.adapters.access.base` for what one looks like.
"""

from __future__ import annotations

from typing import Any, ClassVar

from rlmcp.adapters.access import paths
from rlmcp.adapters.access.base import (
    AccessProvider,
    ProviderRegistry,
    Synthetic,
    Term,
)
from rlmcp.core.parameters.spec import Liveness, ParameterSpec

__all__ = ["ParameterAccess"]


class ParameterAccess:
  """The environment's tunable surface, discovered rather than declared.

  Family-neutral: what makes this a *manager-based* or a *legged-gym-shaped*
  view of an environment is only which providers are bound to it. A family
  subclasses and sets :attr:`PROVIDER_TYPES`; a caller with an unusual
  environment passes ``provider_types`` directly.
  """

  PROVIDER_TYPES: ClassVar[list[type]] = []
  """The providers this family offers. Subclasses set it; see
  :mod:`rlmcp.adapters.manager_based.access` for the worked example."""

  def __init__(self, env: Any, provider_types: list[type] | None = None):
    self.env = env
    self.registry = ProviderRegistry()
    for provider_type in provider_types or self.PROVIDER_TYPES:
      self.registry.add(provider_type(env))
    self._synthetic: dict[str, Synthetic] = {}
    self._refresh_synthetic()

  def _refresh_synthetic(self) -> None:
    """Pick up synthetics that appeared since the last look.

    A reward term added mid-run is the case: its weight is served as a
    synthetic by the same provider as the task's own, but the provider was
    asked once, at wrap time, before the term existed. Keys already served
    keep the object they had -- its recorded default is the value at wrap
    time, which is what a reset goes back to -- and keys that went away are
    dropped.
    """
    seen: set[str] = set()
    for provider in self.registry:
      for item in provider.synthetic():
        seen.add(item.key)
        self._synthetic.setdefault(item.key, item)
    for key in list(self._synthetic):
      if key not in seen:
        del self._synthetic[key]

  def _lookup_synthetic(self, key: str) -> Synthetic | None:
    """The synthetic behind ``key``, looking again once if it is new."""
    item = self._synthetic.get(key)
    if item is None:
      self._refresh_synthetic()
      item = self._synthetic.get(key)
    return item

  # Discovery.

  def discover(self) -> list[ParameterSpec]:
    """Every tunable leaf reachable from every provider's terms."""
    self._refresh_synthetic()
    specs: list[ParameterSpec] = []
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
              liveness=item.liveness,
          )
      )
    return specs

  def _synthetic_category(self, key: str):
    provider = self.registry.get(key.split(".", 1)[0])
    return provider.category if provider else None

  # Resolution.

  def _split(self, key: str) -> tuple[AccessProvider, Term, list[str]]:
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
      known = sorted(t.key for t in candidates)
      served = sorted(k for k in self._synthetic if k.startswith(f"{domain}."))
      hint = provider.miss_hint(key)
      raise KeyError(
          f"No {domain} term in '{key}'. Available: {known}"
          + (f"; and these keys, served directly rather than as term fields: "
             f"{served}" if served else "")
          + (f". {hint}" if hint else "")
      )
    term, term_parts = max(matches, key=lambda pair: len(pair[1]))
    leaf = rest[len(term_parts) :]
    if not leaf:
      raise KeyError(f"'{key}' names a term but no field inside it.")
    return provider, term, leaf

  # Read and write.

  def get(self, key: str) -> Any:
    item = self._lookup_synthetic(key)
    if item is not None:
      return item.getter()
    _, term, parts = self._split(key)
    value = paths.resolve(term.root, parts).read()
    return list(value) if paths.is_range(value) else value

  def set(self, key: str, value: Any) -> dict[str, Any]:
    """Write a parameter. Returns notes from the provider's write hook, if any.

    Refuses writes that cannot take effect this run (liveness 'at_startup' or
    'inert') and values that do not match the leaf's shape -- both before
    anything is mutated, so a rejected call leaves the config exactly as it
    was. A write that lands but only applies later ('at_reset') says so in the
    returned notes.
    """
    item = self._lookup_synthetic(key)
    if item is not None:
      self._refuse_if_not_live(key, item.liveness)
      if not item.setter(value):
        raise RuntimeError(f"Parameter '{key}' could not be applied.")
      return {}

    provider, term, parts = self._split(key)
    resolved = paths.resolve(term.root, parts)
    liveness = provider.liveness(term, parts, resolved.read())
    self._refuse_if_not_live(key, liveness)
    try:
      written = resolved.write(value)
    except (TypeError, ValueError) as exc:
      raise type(exc)(f"Cannot set '{key}': {exc}") from exc
    try:
      self._notify_class_term(term, parts, written)
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
  def _refuse_if_not_live(key: str, liveness: Liveness) -> None:
    """Reject a write that would report success and then change nothing.

    Checked before anything is mutated, so a refused call leaves the config
    exactly as it was. Synthetics go through here too: a parameter served by a
    getter and a setter can still stand in front of a value the environment
    copied into a tensor at construction and never reads again.
    """
    if liveness is Liveness.AT_STARTUP:
      raise ValueError(
          f"Cannot set '{key}': liveness is 'at_startup'. This value is read "
          "exactly once, when the environment is constructed, so the write "
          "would report success and then change nothing for the rest of the "
          "run. Change the task config and restart training instead."
      )
    if liveness is Liveness.INERT:
      raise ValueError(
          f"Cannot set '{key}': liveness is 'inert'. The value was cached when "
          "the term or the environment was constructed and is never re-read, "
          "so no write can take effect during this run. Change the task config "
          "and restart training instead."
      )

  @staticmethod
  def _notify_class_term(term: Term, parts: list[str], value: Any) -> None:
    """Give class-based terms a chance to rebuild anything they cached.

    Most mjlab terms are plain functions reading ``params`` every step and need
    nothing. A class-based term whose instance exposes ``update_params`` gets
    it called with the updated top-level field -- the whole ``std`` dict when
    one regex-keyed entry inside it changed, the whole ``ranges`` object when
    one range did, since that is the shape the term cached from. The instance
    is the term itself for a command term and ``root.func`` for a function
    slot that mjlab filled with an instance. A failure here must surface
    rather than vanish: the config write has already landed, and an agent told
    nothing would keep trusting a cache the term failed to rebuild.
    """
    root = term.root
    instance = term.instance
    if instance is None:
      instance = getattr(root, "func", None)
    hook = getattr(instance, "update_params", None)
    if not callable(hook):
      return
    if len(parts) >= 2 and parts[0] == "params":
      name = parts[1]
      payload = getattr(root, "params", {}).get(name, value)
    elif term.instance is not None:
      name = parts[0]
      payload = paths.resolve(root, [name]).read() if len(parts) > 1 else value
    else:
      name, payload = parts[-1], value
    hook(**{name: payload})

  # Introspection helpers used by the adapter.

  def domains(self) -> list[str]:
    return self.registry.domains()

  def provider(self, domain: str) -> AccessProvider | None:
    return self.registry.get(domain)
