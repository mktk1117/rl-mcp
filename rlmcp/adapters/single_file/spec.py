"""What a single-file environment is, and how rlmcp tells one apart.

There is no base class to inherit from. An environment is single-file-shaped
when it keeps its configuration in one dataclass on ``env.cfg`` (declared
with the markers in :mod:`rlmcp.declare`), its state in the conventional
buffers (``dof_pos``, ``base_lin_vel``, ``commands`` and so on -- the same
names legged_gym uses, which is why the sampler is shared), and its physics
behind ``env.sim``, whatever simulator is underneath.

:class:`SingleFileSpec` names those attributes so an environment that spells
them differently can say so. :func:`detect` checks the shape at wrap time and
refuses, naming what was missing, rather than discovering zero parameters and
letting somebody find out hours into a run.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any


class NotASingleFileEnv(TypeError):
  """Raised when an environment has none of the attributes this family needs."""


@dataclass(frozen=True)
class SingleFileSpec:
  """Attribute names on the environment, one per thing rlmcp needs to find."""

  cfg: str = "cfg"
  """The config dataclass. Every tunable parameter is a field of it, or of a
  nested dataclass under it."""

  reward_group: str = "reward"
  """The field of ``cfg`` holding the reward table: a dataclass whose fields
  are :class:`~rlmcp.declare.Term` values."""

  command_group: str = "command"
  """The field of ``cfg`` holding command ranges. Its ``[low, high]`` fields,
  in declaration order, are the columns of the ``commands`` buffer -- which is
  how the sampler knows whether the buffer holds a plane velocity."""

  sim: str = "sim"
  """The physics backend. Rendering asks it for ``render(env_id)``; the live
  view asks it for ``mj_model`` when it has one."""

  reset_fn: str = "reset"
  """``reset(env_ids)``: restart the given environments, ``None`` for all."""

  control_dt: str = "control_dt"
  """Seconds per policy step. Falls back to ``cfg.dt * cfg.decimation``."""

  max_episode_steps: str = "max_episode_steps"
  reward_buffer: str = "rew_buf"
  """Where the wrapper parks the last step's reward so the trace can read it."""

  def resolve(self, env: Any, name: str, default: Any = None) -> Any:
    """The object ``name`` points at on ``env``, or ``default``."""
    return getattr(env, getattr(self, name), default)


def detect(env: Any, spec: SingleFileSpec | None = None) -> SingleFileSpec:
  """Confirm ``env`` is this shape, or say exactly what was missing."""
  spec = spec or SingleFileSpec()
  problems: list[str] = []

  cfg = spec.resolve(env, "cfg", None)
  if cfg is None:
    problems.append(f"env.{spec.cfg} (the config dataclass) is missing")
  elif not dataclasses.is_dataclass(cfg) or isinstance(cfg, type):
    problems.append(
        f"env.{spec.cfg} is a {type(cfg).__name__}, not a dataclass instance"
    )

  if getattr(env, "num_envs", None) is None:
    problems.append("env.num_envs is missing")
  if not callable(getattr(env, spec.reset_fn, None)):
    problems.append(f"env.{spec.reset_fn}(env_ids) is missing")

  if problems:
    have = sorted(k for k in vars(env) if not k.startswith("_")) if hasattr(env, "__dict__") else []
    raise NotASingleFileEnv(
        f"{type(env).__name__} does not look like a single-file environment: "
        + "; ".join(problems)
        + ". rlmcp reads the config dataclass on env.cfg (declared with "
        "rlmcp.declare's Static and term markers) and the conventional state "
        "buffers. If this environment keeps them under other names, say so "
        "with wrap(spec=SingleFileSpec(...)). Attributes it does have: "
        f"{', '.join(have) or '(none)'}."
    )
  return spec


__all__ = ["NotASingleFileEnv", "SingleFileSpec", "detect"]
