"""The parameter providers for a single-file environment.

Same machinery as every other family -- :mod:`rlmcp.adapters.access` does the
reading and writing -- and the same key vocabulary, so
``reward.tracking_lin_vel.weight`` means what it means on mjlab. What differs
is only where the values are found: fields of one dataclass tree on
``env.cfg`` rather than manager term configs.

The tree decides the domains. Each nested dataclass directly under ``cfg`` is
a domain named after its field (``command.lin_vel_x``, ``termination.max_tilt``,
``noise.dof_pos``); the reward group is served as terms
(``reward.<name>.weight``, ``reward.<name>.params.<p>``); and the scalars left
at the top level are ``env.<name>``. Nothing here is hand-listed by parameter
name, so a knob added to the config is tunable the moment it is declared.

Liveness comes from the declaration. A field marked ``Static`` -- or anything
under a nested dataclass marked ``Static`` -- is ``at_startup``, and a write to
it is refused with that reason. Everything else is live, because a
single-file environment reads its config on every step or resample by
construction; that is the whole point of the shape.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

from rlmcp import declare
from rlmcp.adapters.access import paths
from rlmcp.adapters.access.base import AccessProvider, Synthetic, Term
from rlmcp.adapters.access.registry import ParameterAccess as _ParameterAccess
from rlmcp.adapters.single_file.spec import SingleFileSpec, detect
from rlmcp.core.parameters.spec import Liveness, ParameterCategory

CATEGORY_BY_GROUP = {
    "reward": ParameterCategory.REWARD,
    "command": ParameterCategory.CURRICULUM,
    "termination": ParameterCategory.TERMINATION,
    "action": ParameterCategory.ACTION,
    "noise": ParameterCategory.DOMAIN_RANDOMIZATION,
    "randomization": ParameterCategory.DOMAIN_RANDOMIZATION,
    "domain_randomization": ParameterCategory.DOMAIN_RANDOMIZATION,
    "dr": ParameterCategory.DOMAIN_RANDOMIZATION,
    "sim": ParameterCategory.PHYSICS,
    "physics": ParameterCategory.PHYSICS,
}
"""How a group's name decides the category an agent filters by. A group
named anything else is ``OTHER``, and still fully tunable."""


def _liveness(static: bool) -> Liveness:
  return Liveness.AT_STARTUP if static else Liveness.LIVE


def _is_nested(value: Any) -> bool:
  return (
      dataclasses.is_dataclass(value)
      and not isinstance(value, type)
      and not declare.is_term(value)
  )


def _field_synthetic(
    owner: Any, name: str, key: str, static: bool, description: str
) -> Synthetic | None:
  """A synthetic over one leaf field of a dataclass, or None if not a leaf."""
  value = getattr(owner, name, None)
  if not paths.is_leaf(value):
    return None

  def get() -> Any:
    current = getattr(owner, name)
    return list(current) if paths.is_range(current) else current

  def put(new: Any) -> bool:
    current = getattr(owner, name)
    coerced = paths.coerce_like(current, new)
    if isinstance(current, tuple) and isinstance(coerced, list):
      coerced = tuple(coerced)
    setattr(owner, name, coerced)
    return True

  return Synthetic(
      key=key,
      getter=get,
      setter=put,
      default=list(value) if paths.is_range(value) else value,
      description=description,
      data_type=paths.leaf_kind(value),
      liveness=_liveness(static),
  )


class GroupAccess(AccessProvider):
  """One nested dataclass under ``cfg``, served under its own field name.

  Leaf fields are synthetics (``command.lin_vel_x``); a dataclass nested one
  level further is a term, so its leaves are walked (``noise.imu.gyro``).
  """

  domain = ""
  category = ParameterCategory.OTHER

  def __init__(self, env: Any, spec: SingleFileSpec, name: str, static: bool = False):
    super().__init__(env)
    self.spec = spec
    self.domain = name
    self.category = CATEGORY_BY_GROUP.get(name, ParameterCategory.OTHER)
    self._static = static

  @property
  def group(self) -> Any:
    return getattr(self.spec.resolve(self.env, "cfg", None), self.domain, None)

  def available(self) -> bool:
    return _is_nested(self.group)

  def _statics(self) -> set[str]:
    return declare.static_fields(self.group) if self.available() else set()

  def terms(self) -> list[Term]:
    group = self.group
    if not _is_nested(group):
      return []
    out: list[Term] = []
    for f in dataclasses.fields(group):
      value = getattr(group, f.name, None)
      if _is_nested(value):
        out.append(Term(key=f.name, root=value, label=f"{self.domain}.{f.name}"))
    return out

  def synthetic(self) -> list[Synthetic]:
    group = self.group
    if not _is_nested(group):
      return []
    statics = self._statics()
    out: list[Synthetic] = []
    for f in dataclasses.fields(group):
      item = _field_synthetic(
          group, f.name, f"{self.domain}.{f.name}",
          self._static or f.name in statics,
          self.describe_field(f.name),
      )
      if item is not None:
        out.append(item)
    return out

  def describe_field(self, name: str) -> str:
    return f"'{name}' in the {self.domain} section of the task config"

  def liveness(self, term: Term, parts: Sequence[str], value: Any) -> Liveness:
    if self._static or term.key in self._statics():
      return Liveness.AT_STARTUP
    owner = term.root
    for part in parts[:-1]:
      if declare.is_static(owner, part):
        return Liveness.AT_STARTUP
      owner = getattr(owner, part, None)
      if owner is None:
        return Liveness.LIVE
    return _liveness(declare.is_static(owner, parts[-1]))

  def describe(self, term: Term, parts: Sequence[str], value: Any) -> str:
    return f"'{'.'.join(parts)}' under {term.label} in the task config"

  # Command groups: which channel is what.

  def channel_names(self) -> list[str]:
    """The range fields, in declaration order -- the ``commands`` columns."""
    group = self.group
    if not _is_nested(group):
      return []
    return [
        f.name for f in dataclasses.fields(group)
        if paths.is_range(getattr(group, f.name, None))
    ]


class RewardAccess(AccessProvider):
  """``reward.<term>.weight`` and ``reward.<term>.params.<p>`` off the table."""

  domain = "reward"
  category = ParameterCategory.REWARD

  def __init__(self, env: Any, spec: SingleFileSpec):
    super().__init__(env)
    self.spec = spec

  @property
  def table(self) -> dict[str, declare.Term]:
    cfg = self.spec.resolve(self.env, "cfg", None)
    return declare.terms(getattr(cfg, self.spec.reward_group, None))

  def available(self) -> bool:
    return bool(self.table)

  def terms(self) -> list[Term]:
    return []

  def synthetic(self) -> list[Synthetic]:
    out: list[Synthetic] = []
    for name, term in self.table.items():
      out.append(
          Synthetic(
              key=f"{self.domain}.{name}.weight",
              getter=self._weight_getter(name),
              setter=self._weight_setter(name),
              default=float(term.weight),
              description=self._describe_weight(name, term),
          )
      )
      for param, value in term.params.items():
        if not paths.is_leaf(value):
          continue
        out.append(
            Synthetic(
                key=f"{self.domain}.{name}.params.{param}",
                getter=self._param_getter(name, param),
                setter=self._param_setter(name, param),
                default=list(value) if paths.is_range(value) else value,
                description=f"Parameter '{param}' read by reward term '{name}'",
                data_type=paths.leaf_kind(value),
            )
        )
    return out

  def _term(self, name: str) -> declare.Term:
    table = self.table
    if name not in table:
      raise KeyError(
          f"No reward term '{name}'. Available: {sorted(table)}. "
          "`rlmcp add-reward` can add one, since it brings the function."
      )
    return table[name]

  def _weight_getter(self, name: str):
    def get() -> float:
      return float(self._term(name).weight)
    return get

  def _weight_setter(self, name: str):
    def put(value: Any) -> bool:
      try:
        weight = float(value)
      except (TypeError, ValueError):
        raise ValueError(
            f"Reward weight '{name}' takes a number; got {value!r}."
        ) from None
      self._term(name).weight = weight
      return True
    return put

  def _param_getter(self, name: str, param: str):
    def get() -> Any:
      current = self._term(name).params[param]
      return list(current) if paths.is_range(current) else current
    return get

  def _param_setter(self, name: str, param: str):
    def put(value: Any) -> bool:
      term = self._term(name)
      current = term.params[param]
      coerced = paths.coerce_like(current, value)
      if isinstance(current, tuple) and isinstance(coerced, list):
        coerced = tuple(coerced)
      term.params[param] = coerced
      return True
    return put

  @staticmethod
  def _describe_weight(name: str, term: declare.Term) -> str:
    kind = "penalty" if float(term.weight) < 0 else "reward"
    added = " (added at runtime)" if term.func is not None else ""
    return f"Weight of reward term '{name}' ({kind}){added}"

  def miss_hint(self, key: str) -> str:
    return (
        "Reward terms are the Term fields of the config's reward section. "
        "`rlmcp add-reward` appends one, since it brings the function."
    )

  def describe(self, term: Term, parts: Sequence[str], value: Any) -> str:
    return f"Parameter '{'.'.join(parts)}' of reward term '{term.key}'"


class EnvAccess(AccessProvider):
  """The scalars left at the top level of ``cfg``, as ``env.<name>``."""

  domain = "env"
  category = ParameterCategory.OTHER

  def __init__(self, env: Any, spec: SingleFileSpec):
    super().__init__(env)
    self.spec = spec

  @property
  def cfg(self) -> Any:
    return self.spec.resolve(self.env, "cfg", None)

  def available(self) -> bool:
    return _is_nested(self.cfg)

  def terms(self) -> list[Term]:
    return []

  def synthetic(self) -> list[Synthetic]:
    cfg = self.cfg
    if not _is_nested(cfg):
      return []
    statics = declare.static_fields(cfg)
    out: list[Synthetic] = []
    for f in dataclasses.fields(cfg):
      item = _field_synthetic(
          cfg, f.name, f"{self.domain}.{f.name}", f.name in statics,
          f"'{f.name}' in the task config",
      )
      if item is not None:
        out.append(item)
    return out


class ParameterAccess(_ParameterAccess):
  """The tunable surface of one single-file environment.

  Providers are built from the config tree rather than listed: one for the
  reward table, one per nested dataclass, one for the top-level scalars.
  """

  def __init__(self, env: Any, spec: SingleFileSpec | None = None):
    self.spec = detect(env, spec)
    cfg = self.spec.resolve(env, "cfg")
    statics = declare.static_fields(cfg)
    providers: list[Any] = []
    for f in dataclasses.fields(cfg):
      value = getattr(cfg, f.name, None)
      if f.name == self.spec.reward_group:
        providers.append(lambda e, s=self.spec: RewardAccess(e, s))
      elif _is_nested(value):
        providers.append(
            lambda e, s=self.spec, n=f.name, st=(f.name in statics): GroupAccess(e, s, n, st)
        )
    providers.append(lambda e, s=self.spec: EnvAccess(e, s))
    super().__init__(env, providers)

  def provider(self, domain: str) -> AccessProvider | None:
    return self.registry.get(domain)

  def command_channels(self) -> list[str]:
    provider = self.provider(self.spec.command_group)
    if isinstance(provider, GroupAccess):
      return provider.channel_names()
    return []


__all__ = [
    "CATEGORY_BY_GROUP",
    "EnvAccess",
    "GroupAccess",
    "ParameterAccess",
    "RewardAccess",
]
