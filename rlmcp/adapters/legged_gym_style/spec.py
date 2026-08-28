"""Where a legged-gym-shaped environment keeps the things rlmcp tunes.

Manager-based environments describe themselves: ask the reward manager for its
active terms and it answers. This family does not. The tunables are plain dicts
on the env instance, bound to attribute names by convention rather than by an
interface, so something has to say which attribute is which.

:class:`FlatEnvSpec` is that something, and it is *detected* before it is
declared: the conventional names are tried first, and an explicit spec is only
needed for an environment that spells them differently. What is never done is
guessing silently -- an env that matches nothing is refused at wrap time, with
the names it was looking for, rather than coming up with zero parameters and
letting somebody discover that four hours into a run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


class NotAFlatEnv(TypeError):
  """Raised when an environment has none of the attributes this family needs."""


@dataclass(frozen=True)
class FlatEnvSpec:
  """Attribute names on the environment, one per thing rlmcp needs to find."""

  reward_scales: str = "reward_scales"
  """``{term_name: scale}``. The scales are live: the env multiplies each
  reward function's output by its entry every step."""

  reward_cfg: str = "reward_cfg"
  """Parameters the reward functions read, ``tracking_sigma`` and friends."""

  command_cfg: str = "command_cfg"
  env_cfg: str = "env_cfg"
  obs_cfg: str = "obs_cfg"

  command_limits: str = "commands_limits"
  """The ``(lower, upper)`` tensors the command sampler actually reads. Genesis
  builds these once from ``command_cfg``, which is why writing ``command_cfg``
  alone changes nothing -- see :mod:`.access.commands`."""

  commands: str = "commands"
  robot: str = "robot"
  scene: str = "scene"
  dt: str = "dt"
  num_envs: str = "num_envs"
  reset_fn: str = "_reset_idx"
  episode_length: str = "episode_length_buf"
  max_episode_length: str = "max_episode_length"

  scales_premultiplied_by_dt: bool = True
  """Whether ``__init__`` already multiplied every reward scale by ``dt``.

  True for Genesis's own examples and for legged_gym, and unrecoverable by
  inspection: the multiplication happens in place on the dict ``reward_cfg``
  holds, so the configured number is gone by the time rlmcp sees it. The
  wrapper says which way it assumed at startup rather than leaving it implicit.
  """

  def resolve(self, env: Any, name: str, default: Any = None) -> Any:
    """The object ``name`` points at on ``env``, or ``default``."""
    return getattr(env, getattr(self, name), default)


REQUIRED = ("reward_scales", "env_cfg")
"""Without these there is nothing to tune, so there is nothing to wrap."""


def detect(env: Any, spec: Optional[FlatEnvSpec] = None) -> FlatEnvSpec:
  """Confirm ``env`` is this shape, or say exactly what was missing.

  A spec passed in is honoured as given -- somebody who names their attributes
  is not second-guessed -- but it is still checked, because a typo in a spec
  and an environment of the wrong shape fail the same way and deserve the same
  message.
  """
  spec = spec or FlatEnvSpec()
  missing: Dict[str, str] = {}
  for field in REQUIRED:
    attr = getattr(spec, field)
    if getattr(env, attr, None) is None:
      missing[field] = attr
  if missing:
    have = sorted(k for k in vars(env) if not k.startswith("_"))
    raise NotAFlatEnv(
        f"{type(env).__name__} does not look like a legged-gym-shaped "
        "environment: rlmcp needs "
        + ", ".join(f"env.{attr} ({field})" for field, attr in missing.items())
        + " and found neither. If this environment keeps them under other "
        "names, say so with wrap(spec=FlatEnvSpec(...)). Attributes it does "
        f"have: {', '.join(have) or '(none)'}."
    )
  return spec
