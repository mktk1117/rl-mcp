"""The single-file half of the wrapper: its adapter, and the loop's two lines.

Everything else -- servicing, telemetry, curricula, records, progress clips --
comes from :mod:`rlmcp.adapters.env_wrapper`, unchanged. What this family
adds is how a hand-written training loop tells rlmcp where its iteration
boundary is, since there is no runner to hook::

    env = MyEnv(cfg)                             # a SingleFileEnv
    env = rlmcp.adapters.single_file.wrap(env, session_dir=log_dir / "rlmcp")
    env.attach_algorithm(ppo)                    # knobs + checkpoints

    for iteration in range(1, max_iterations + 1):
      ...rollout with env.step(), then ppo.update()...
      env.service(iteration, metrics=losses)     # the iteration boundary

``service`` is where parameter edits land, commands are answered and a pause
blocks: after the update, before the next rollout, which is the one point
where nothing is mid-step. It raises :class:`TrainingStopped` when an agent
asks the run to stop, so the loop can save and exit.
"""

from __future__ import annotations

from typing import Any

import torch

from rlmcp.adapters.env_wrapper import RlMcpEnvWrapper as _BaseWrapper
from rlmcp.adapters.env_wrapper import TrainingStopped
from rlmcp.adapters.single_file.algorithm import AlgorithmAdapter
from rlmcp.adapters.single_file.sim_adapter import SingleFileSimAdapter
from rlmcp.adapters.single_file.spec import SingleFileSpec


class RlMcpEnvWrapper(_BaseWrapper):
  """Transparent wrapper over a single-file environment."""

  def __init__(self, env: Any, spec: SingleFileSpec | None = None, **kwargs: Any):
    self._spec = spec
    self._algorithm: AlgorithmAdapter | None = None
    super().__init__(env, **kwargs)

  def build_sim_adapter(self, env: Any, robot_name: str | None) -> Any:
    return SingleFileSimAdapter(env, spec=self._spec)

  def startup_checks(self) -> None:
    sim = self.rlmcp.sim
    backend = getattr(sim.sim, "name", None) or type(sim.sim).__name__
    print(
        f"[rlmcp] single-file environment on the {backend} backend; "
        f"{len(sim.discover_parameters())} parameters declared by the config. "
        "Call env.service(iteration) once per learning iteration, after the "
        "update, so edits land between rollouts.",
        flush=True,
    )
    if not sim.renderer_ready():
      print(
          "[rlmcp] this backend has no render(env_id), so `shot`, `video` and "
          "progress clips are unavailable on it. Everything else works.",
          flush=True,
      )

  # The step: park the reward where the trace reads it, log the terms.

  def collect_step_logs(self, out: Any) -> None:
    if not isinstance(out, tuple) or len(out) < 2:
      return
    rewards = out[1]
    if torch.is_tensor(rewards):
      spec = self.rlmcp.sim.spec
      state = spec.resolve(self.unwrapped, "state", None)
      if state is not None and "reward" in state:
        state.reward.copy_(rewards)
      else:
        setattr(self.unwrapped, spec.reward_buffer, rewards)
    info = out[-1] if isinstance(out[-1], dict) else {}
    log: dict[str, Any] = {}
    terms = info.get("reward_terms")
    if isinstance(terms, dict):
      for name, value in terms.items():
        log[f"rewards/{name}"] = value
    for key, tag in (("episode_rewards", "episode/reward"),
                     ("episode_lengths", "episode/length")):
      value = info.get(key)
      if torch.is_tensor(value) and value.numel() > 0:
        log[tag] = value.float().mean()
    self._accumulate_log(log)

  # The loop's two lines.

  def attach_algorithm(self, algorithm: Any, log_dir: str | None = None) -> AlgorithmAdapter:
    """Hand over the PPO object: hyperparameters and checkpoints.

    Also marks the iteration boundary as the caller's job, so the wrapper
    stops servicing on a step cadence and waits for :meth:`service`.
    """
    adapter = AlgorithmAdapter(algorithm, log_dir=log_dir)
    self._algorithm = adapter
    self.rlmcp.attach_runner(adapter)
    self._runner_hooked = True
    clips = self.rlmcp.progress_video
    if clips.active:
      print(
          f"[rlmcp] progress clips: {clips.seconds:g}s of env {clips.env_id}, "
          f"{clips.cadence.prose()}; each one is filed in the run record "
          f"(budget {clips.budget_mb:g} MB). Change it with "
          "`rlmcp video --every <cadence>`.",
          flush=True,
      )
    return adapter

  def service(self, iteration: int, metrics: dict[str, Any] | None = None) -> None:
    """The iteration boundary: apply edits, answer commands, honour pause.

    ``metrics`` are scalars the loop already has -- the update's losses, say
    -- and are published alongside rlmcp's own. Raises
    :class:`TrainingStopped` when an agent asked the run to stop.
    """
    self._runner_hooked = True
    if self._algorithm is not None:
      self._algorithm.iteration = int(iteration)
    extra: dict[str, float] = {}
    for key, value in (metrics or {}).items():
      try:
        extra[str(key)] = float(value)
      except (TypeError, ValueError):
        continue
    self._service(iteration=int(iteration), extra_metrics=extra)


def wrap(env: Any, **kwargs: Any) -> RlMcpEnvWrapper:
  """Wrap a single-file environment so rlmcp can watch and steer the run.

  See :class:`~rlmcp.adapters.env_wrapper.RlMcpEnvWrapper` for the keyword
  arguments, and ``spec=`` for an environment whose attributes are not the
  conventional ones.
  """
  return RlMcpEnvWrapper(env, **kwargs)


__all__ = ["RlMcpEnvWrapper", "TrainingStopped", "wrap"]
