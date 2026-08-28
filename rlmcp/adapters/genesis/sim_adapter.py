"""SimAdapter for a Genesis environment.

Thin, and thinner than it looks: parameter discovery, trace sampling and
summary metrics come from :mod:`rlmcp.adapters.legged_gym_style`, which is
written against the shape Genesis's examples share with legged_gym and its
forks -- dicts on the environment instance, buffers as attributes, reward
functions bound at construction. What is genuinely Genesis's own is small: how
a frame is asked for, and how episodes are restarted.

Everything an agent changes lands on an object the environment re-reads on its
next step, so nothing here needs a restart or a checkpoint round-trip. The
exceptions are declared rather than discovered the hard way -- see
:mod:`rlmcp.adapters.legged_gym_style.access.env_cfg` for what is refused and
why.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from rlmcp.adapters.base import NotSupported, SimAdapter
from rlmcp.adapters.genesis import rendering
from rlmcp.adapters.legged_gym_style import metrics as flat_metrics
from rlmcp.adapters.legged_gym_style.access import ParameterAccess
from rlmcp.adapters.legged_gym_style.sampling import StateSampler
from rlmcp.adapters.legged_gym_style.spec import FlatEnvSpec, detect
from rlmcp.core.parameters.spec import ParameterSpec


class GenesisSimAdapter(SimAdapter):
  """Live control surface over one Genesis environment."""

  def __init__(self, env: Any, spec: Optional[FlatEnvSpec] = None):
    self.env = env
    self.spec = detect(env, spec)
    self.parameters = ParameterAccess(env, self.spec)
    self.sampler = StateSampler(env, command_names=self._command_names())
    self._last_set_notes: Dict[str, Any] = {}

  def _command_names(self) -> List[str]:
    """What each command channel is, asked of the provider that knows.

    The sampler needs these to decide whether the command buffer is a plane
    velocity, which decides whether it may be published under the channel the
    tracking diagnostics read.
    """
    provider = self.parameters.provider("command")
    return provider.channel_names() if provider is not None else []

  # Required.

  def discover_parameters(self) -> List[ParameterSpec]:
    return self.parameters.discover()

  def get_parameter(self, key: str) -> Any:
    return self.parameters.get(key)

  def set_parameter(self, key: str, value: Any) -> bool:
    self._last_set_notes = self.parameters.set(key, value)
    return True

  def last_set_notes(self) -> Dict[str, Any]:
    return dict(self._last_set_notes)

  # Introspection.

  def num_envs(self) -> int:
    return int(self.env.num_envs)

  def step_dt(self) -> float:
    dt = self.spec.resolve(self.env, "dt", None)
    if dt is None:
      raise NotSupported("step_dt")
    return float(dt)

  def joint_names(self) -> List[str]:
    names = self.sampler.joint_names()
    if not names:
      raise NotSupported("joint_names")
    return names

  def max_episode_length(self) -> Optional[float]:
    length = self.spec.resolve(self.env, "max_episode_length", None)
    return float(length) if length else None

  # State.

  def sample_state(self, env_id: int = 0) -> Dict[str, np.ndarray]:
    sample = self.sampler.sample(env_id)
    if not sample:
      raise NotSupported("sample_state")
    return sample

  def trace_labels(self) -> Dict[str, List[str]]:
    return self.sampler.labels()

  def summary_metrics(self) -> Dict[str, float]:
    return flat_metrics.summary_metrics(self.env, self.sampler)

  def reset_envs(self, env_ids: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Start fresh episodes, through the path a termination already takes.

    Genesis's reset takes a boolean mask over environments rather than a list
    of indices, and ``None`` means all of them -- so the ids the controller
    resolved (including whatever ``--where`` matched) are converted here rather
    than anywhere that would have to know about masks.
    """
    reset = getattr(self.env, self.spec.reset_fn, None)
    if not callable(reset):
      raise NotSupported(
          f"reset_envs: this environment has no {self.spec.reset_fn}()."
      )
    total = self.num_envs()
    if env_ids is None:
      reset(None)
      return {"num_reset": total, "env_ids": None}

    ids = sorted({int(i) for i in env_ids})
    out_of_range = [i for i in ids if not 0 <= i < total]
    if out_of_range:
      raise ValueError(
          f"No environment {out_of_range[0]}: this run has {total} "
          f"(0 to {total - 1})."
      )
    if not ids:
      return {"num_reset": 0, "env_ids": []}
    reset(self._mask(ids, total))
    return {"num_reset": len(ids), "env_ids": ids}

  def _mask(self, ids: List[int], total: int) -> Any:
    import torch

    device = getattr(self.env, "device", None)
    mask = torch.zeros(total, dtype=torch.bool, device=device)
    mask[torch.tensor(ids, dtype=torch.long, device=device)] = True
    return mask

  # Rendering.

  def render(self, env_id: int = 0) -> np.ndarray:
    return rendering.render(self.env, env_id)

  def renderer_ready(self) -> bool:
    """True when a frame costs nothing to set up.

    Either there is a camera or there never will be one this run, so this is
    also the answer to "will a screenshot work" -- unlike a backend that builds
    a renderer on demand and might fail doing it.
    """
    return rendering.observing_camera(self.env) is not None
