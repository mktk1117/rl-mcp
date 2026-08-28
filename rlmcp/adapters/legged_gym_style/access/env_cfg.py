"""Three domains that all read one dict.

``env_cfg`` is a flat bag: termination limits, the action scale, the PD gains
and the initial pose sit in it side by side, with nothing in the dict saying
which is which. So the split into domains is made here, by name, and it is the
one place in this package that knows a convention rather than discovering a
structure -- because there is no structure to discover.

What is *not* hand-listed is the contents. Anything in ``env_cfg`` that no
domain claims still shows up under ``env.<key>``, so a fork's extra knob is
tunable the day it is added rather than the day somebody remembers to list it.
Only numbers, bools and ``[min, max]`` pairs are offered, which is what keeps
joint-name lists and initial quaternions out of the parameter list without a
rule for each.

These are single scalars rather than objects with fields, so they are served as
synthetics under their own names -- ``action.scale``, not ``action.scale.value``.
Inventing a field name to hang a scalar off would be describing rlmcp's
plumbing rather than the environment.

Liveness is the part that matters. Several of these values are copied into
tensors at construction and never read again: the PD gains go through
``set_dofs_kp``, the default joint angles become ``default_dof_pos``,
``episode_length_s`` becomes ``max_episode_length``. A write to any of them
would report success and change nothing, so they are declared ``at_startup``
and refused with a reason.
"""

from __future__ import annotations

from typing import Any, Dict, List

from rlmcp.adapters.access import paths
from rlmcp.adapters.access.base import AccessProvider, Synthetic, Term
from rlmcp.core.parameters.spec import Liveness, ParameterCategory

TERMINATION_PREFIX = "termination_if_"

ACTION_KEYS = {"action_scale": "scale", "clip_actions": "clip"}

BAKED_AT_STARTUP = frozenset(
    {
        "kp",
        "kd",
        "num_actions",
        "num_obs",
        "num_commands",
        "episode_length_s",
        "default_joint_angles",
        "base_init_pos",
        "base_init_quat",
        "joint_names",
    }
)
"""Read once, when the environment is built. A write cannot reach the run."""


class _EnvCfgProvider(AccessProvider):
  """Shared plumbing: every provider here serves scalars out of one dict."""

  def __init__(self, env: Any, spec: Any):
    super().__init__(env)
    self.spec = spec

  @property
  def cfg(self) -> dict:
    cfg = self.spec.resolve(self.env, "env_cfg", None)
    return cfg if isinstance(cfg, dict) else {}

  def claimed(self) -> Dict[str, str]:
    """``{env_cfg key: the name it is exposed under}`` for this domain."""
    raise NotImplementedError

  def available(self) -> bool:
    return bool(self.claimed())

  def terms(self) -> List[Term]:
    return []

  def synthetic(self) -> List[Synthetic]:
    return [
        Synthetic(
            key=f"{self.domain}.{name}",
            getter=self._getter(cfg_key),
            setter=self._setter(cfg_key),
            default=self.cfg.get(cfg_key),
            description=self.explain(cfg_key, name),
            data_type=paths.leaf_kind(self.cfg.get(cfg_key)),
            liveness=self.entry_liveness(cfg_key),
        )
        for cfg_key, name in self.claimed().items()
        if paths.is_leaf(self.cfg.get(cfg_key))
    ]

  def entry_liveness(self, cfg_key: str) -> Liveness:
    return Liveness.LIVE

  def explain(self, cfg_key: str, name: str) -> str:
    return f"env_cfg['{cfg_key}']"

  def _getter(self, cfg_key: str):
    def get() -> Any:
      value = self.cfg[cfg_key]
      return list(value) if paths.is_range(value) else value
    return get

  def _setter(self, cfg_key: str):
    def put(value: Any) -> bool:
      current = self.cfg[cfg_key]
      try:
        if paths.is_range(current):
          if not paths.is_range(value):
            raise ValueError("expected a [min, max] pair")
          self.cfg[cfg_key] = type(current)(float(v) for v in value)
        elif isinstance(current, bool):
          if not isinstance(value, bool):
            raise ValueError("expected a bool")
          self.cfg[cfg_key] = value
        elif isinstance(current, int):
          self.cfg[cfg_key] = int(value)
        else:
          self.cfg[cfg_key] = float(value)
      except (TypeError, ValueError) as exc:
        raise ValueError(f"Cannot set env_cfg['{cfg_key}']: {exc}") from None
      return True
    return put


class TerminationAccess(_EnvCfgProvider):
  """``termination.<name>``, one per ``termination_if_*`` limit."""

  domain = "termination"
  category = ParameterCategory.TERMINATION

  def claimed(self) -> Dict[str, str]:
    return {
        key: key[len(TERMINATION_PREFIX):]
        for key in self.cfg
        if key.startswith(TERMINATION_PREFIX)
    }

  def explain(self, cfg_key: str, name: str) -> str:
    return f"Episode ends when {name.replace('_', ' ')} this value"


class ActionAccess(_EnvCfgProvider):
  """``action.scale`` and ``action.clip``: how hard the policy may push, both
  read on every step."""

  domain = "action"
  category = ParameterCategory.ACTION

  def claimed(self) -> Dict[str, str]:
    return {key: name for key, name in ACTION_KEYS.items() if key in self.cfg}

  def explain(self, cfg_key: str, name: str) -> str:
    if name == "scale":
      return "Multiplier from policy output to target joint position"
    return "Symmetric limit the policy's raw output is clipped to"


class EnvAccess(_EnvCfgProvider):
  """``env.<key>``: everything in ``env_cfg`` no other domain claimed."""

  domain = "env"
  category = ParameterCategory.OTHER

  def claimed(self) -> Dict[str, str]:
    return {
        key: key
        for key in self.cfg
        if key not in ACTION_KEYS and not key.startswith(TERMINATION_PREFIX)
    }

  def entry_liveness(self, cfg_key: str) -> Liveness:
    return Liveness.AT_STARTUP if cfg_key in BAKED_AT_STARTUP else Liveness.LIVE

  def explain(self, cfg_key: str, name: str) -> str:
    if cfg_key in BAKED_AT_STARTUP:
      return f"env_cfg['{cfg_key}'] -- read once, when the environment is built"
    return f"env_cfg['{cfg_key}']"
