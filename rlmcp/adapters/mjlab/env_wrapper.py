"""The one line you add to a training script.

``RlMcpEnvWrapper`` sits between the task environment and the RL runner. It
forwards every attribute it does not handle, so anything that worked on the bare
environment keeps working -- including mjlab's own ``RslRlVecEnvWrapper``, which
reaches through ``.unwrapped``.

What it adds:

* per-step hooks that feed traces and video capture,
* per-iteration servicing of agent commands,
* accumulation of mjlab's episode logs into rlmcp's telemetry,
* an optional terrain curriculum that advances itself.

Usage::

    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")
    env = rlmcp.wrap(env, session_dir=log_dir / "rlmcp", curriculum="terrain")
    vec_env = RslRlVecEnvWrapper(env)
    runner = VelocityOnPolicyRunner(vec_env, agent_cfg, str(log_dir), device)
    env.attach_runner(runner)                 # PPO knobs + checkpoints + exact
    runner.learn(num_learning_iterations=...) # iteration boundaries

Keep a name bound to the rlmcp wrapper: ``attach_runner`` lives on it, and
``RslRlVecEnvWrapper`` does not forward attribute lookups.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import torch

from rlmcp.adapters.mjlab.runner_adapter import MjlabRunnerAdapter
from rlmcp.adapters.mjlab.sim_adapter import MjlabSimAdapter
from rlmcp.core.controller import RlMcp
from rlmcp.core.curriculum import StageSchedule
from rlmcp.adapters.mjlab.viz_check import check_marker_colors
from rlmcp.core.palette import format_report
from rlmcp.core.extensions import Extension
from rlmcp.extensions import discover as discover_extensions

CurriculumArg = Union[None, str, StageSchedule, Sequence[Any]]


class TrainingStopped(RuntimeError):
  """Raised inside the training loop when an agent asks training to stop.

  The training entrypoint is expected to catch this, save a final checkpoint and
  exit cleanly -- rsl_rl's ``learn()`` has no stop hook of its own.
  """


class RlMcpEnvWrapper:
  """Transparent wrapper that exposes a training run to rlmcp."""

  def __init__(
      self,
      env: Any,
      session_dir: Optional[Path | str] = None,
      curriculum: CurriculumArg = None,
      service_every_steps: int = 24,
      robot_name: Optional[str] = None,
      task_id: str = "",
      trace_capacity: int = 6000,
      curriculum_kwargs: Optional[Dict[str, Any]] = None,
      extensions: Optional[Sequence[str]] = None,
      exclude_extensions: Optional[Sequence[str]] = None,
      record_run: Optional[str] = None,
      records_root: Optional[str] = None,
      record_slot: str = "",
      record_strict: bool = False,
  ):
    self.env = env
    self.service_every_steps = max(1, int(service_every_steps))

    sim_adapter = MjlabSimAdapter(self.unwrapped, robot_name=robot_name)
    session_dir = Path(session_dir) if session_dir else Path.cwd() / "rlmcp"

    from rlmcp.records.link import open_link

    records = open_link(record_run, root=records_root, slot=record_slot, strict=record_strict)

    self.rlmcp = RlMcp(
        sim_adapter=sim_adapter,
        runner_adapter=None,
        session_dir=session_dir,
        curriculum=None,  # Set below, once extensions can inform the plan.
        trace_capacity=trace_capacity,
        records=records,
        session_info={
            "task": task_id,
            "num_envs": sim_adapter.num_envs(),
            "device": str(getattr(self.unwrapped, "device", "")),
            "step_dt": sim_adapter.step_dt(),
            "render_mode": getattr(self.unwrapped, "render_mode", None),
        },
    )

    # Extensions are built after the controller so they can write artifacts
    # through it, and register only if this environment supports them.
    for extension in discover_extensions(
        self.unwrapped,
        plot_sink=self.rlmcp.write_artifact,
        include=extensions,
        exclude=exclude_extensions,
    ):
      self.rlmcp.add_extension(extension)

    lab.start(
        str(self.rlmcp.session.dir), self.rlmcp.parameters.get_snapshot()
    )

    self.rlmcp.curriculum = self._resolve_curriculum(
        curriculum, self.rlmcp, curriculum_kwargs or {}
    )

    self._steps = 0
    self._runner_hooked = False
    self._log_sums: Dict[str, torch.Tensor] = {}
    self._log_counts: Dict[str, int] = {}

    self._warn_about_marker_colors()

    active = self.rlmcp.extensions.names()
    print(
        f"[rlmcp] session ready: {self.rlmcp.session.dir}\n"
        f"[rlmcp] extensions: {', '.join(active) if active else 'none'}\n"
        f"[rlmcp] inspect it with: rlmcp status --session {self.rlmcp.session.dir}",
        flush=True,
    )

  def _warn_about_marker_colors(self) -> None:
    """Say so at startup if a debug overlay is the colour of a scene object.

    Twice in this project a visualisation has made the system look like it was
    doing something it was not -- a landing marker the same yellow as a ball,
    and neighbouring envs composited into one frame -- and both times the search
    went after a physics bug that did not exist. The check is cheap, the failure
    is silent, and nobody thinks to look for it, so it runs unasked. It only
    ever warns: a colour clash is a legibility problem, not a reason to refuse
    to train.
    """
    try:
      collisions = check_marker_colors(self.unwrapped)
    except Exception as exc:  # a diagnostic must never take the run down
      self.rlmcp.session.append_event(
          "viz_check_failed", {"error": f"{type(exc).__name__}: {exc}"}
      )
      return
    if not collisions:
      return
    print(f"[rlmcp] {format_report(collisions)}", flush=True)
    self.rlmcp.session.append_event("viz_color_collision", {"collisions": collisions})

  # Construction helpers.

  @staticmethod
  def _resolve_curriculum(
      curriculum: CurriculumArg,
      lab: RlMcp,
      kwargs: Dict[str, Any],
  ) -> Optional[StageSchedule]:
    """Turn the ``curriculum=`` argument into a schedule, or None."""
    if curriculum is None:
      return None
    if isinstance(curriculum, StageSchedule):
      return curriculum
    if isinstance(curriculum, (list, tuple)):
      return StageSchedule(list(curriculum))
    if isinstance(curriculum, str) and curriculum in ("terrain", "auto"):
      terrain = next(
          (e for e in lab.extensions if e.name == "terrain"), None
      )
      if terrain is None:
        print(
            "[rlmcp] this environment has no terrain grid; skipping the "
            "automatic terrain curriculum.",
            flush=True,
        )
        return None
      from rlmcp.extensions.terrain import build_terrain_plan

      return build_terrain_plan(
          terrain.terrain_names(), terrain.num_levels(), **kwargs
      )
    raise ValueError(
        f"Unsupported curriculum argument {curriculum!r}. Use None, 'terrain', "
        "a StageSchedule, or a list of CurriculumStage."
    )

  # Transparent forwarding.

  @property
  def unwrapped(self) -> Any:
    return getattr(self.env, "unwrapped", self.env)

  def __getattr__(self, name: str) -> Any:
    # Only called when normal lookup fails, so wrapper attributes win.
    return getattr(self.__dict__["env"], name)

  def __repr__(self) -> str:
    return f"RlMcpEnvWrapper({self.env!r})"

  # Gym-ish surface.

  def reset(self, *args: Any, **kwargs: Any) -> Any:
    return self.env.reset(*args, **kwargs)

  def step(self, action: Any) -> Any:
    out = self.env.step(action)
    self._steps += 1
    self.rlmcp.on_step()
    extras = out[-1] if isinstance(out, tuple) and out else None
    if isinstance(extras, dict):
      self._accumulate_log(extras.get("log"))
    if not self._runner_hooked and self._steps % self.service_every_steps == 0:
      self._service()
    return out

  def render(self, *args: Any, **kwargs: Any) -> Any:
    return self.env.render(*args, **kwargs)

  def close(self) -> Any:
    self.rlmcp.close()
    return self.env.close()

  # Telemetry accumulation.

  def _accumulate_log(self, log: Optional[Dict[str, Any]]) -> None:
    """Sum mjlab's episode logs on-device; convert once per iteration."""
    if not log:
      return
    for key, value in log.items():
      if torch.is_tensor(value):
        value = value.detach().reshape(-1)
        if value.numel() == 0:
          continue
        value = value.mean() if value.numel() > 1 else value.reshape(())
      elif isinstance(value, (int, float)):
        value = torch.tensor(float(value))
      else:
        continue
      if key in self._log_sums:
        self._log_sums[key] = self._log_sums[key] + value
      else:
        self._log_sums[key] = value.clone() if torch.is_tensor(value) else value
      self._log_counts[key] = self._log_counts.get(key, 0) + 1

  def _flush_log(self) -> Dict[str, float]:
    if not self._log_sums:
      return {}
    out: Dict[str, float] = {}
    for key, total in self._log_sums.items():
      count = max(1, self._log_counts.get(key, 1))
      try:
        out[key] = float(total.item()) / count
      except Exception:
        continue
    self._log_sums.clear()
    self._log_counts.clear()
    return out

  # Servicing.

  def _service(self, iteration: Optional[int] = None) -> None:
    self.rlmcp.service(iteration=iteration, metrics=self._flush_log())
    if self.rlmcp.should_stop():
      raise TrainingStopped(
          self.rlmcp.stop_reason or "Training stop requested through rlmcp."
      )

  # Runner integration.

  def attach_runner(self, runner: Any) -> MjlabRunnerAdapter:
    """Hook an rsl_rl runner: PPO knobs, checkpoints, exact iteration boundaries.

    The runner's logger is called exactly once per learning iteration, after the
    policy update, which is the safest point to apply parameter changes and to
    block on a pause.
    """
    adapter = MjlabRunnerAdapter(runner)
    self.rlmcp.attach_runner(adapter)

    logger = getattr(runner, "logger", None)
    if logger is None or not hasattr(logger, "log"):
      print(
          "[rlmcp] runner has no logger; falling back to servicing every "
          f"{self.service_every_steps} steps.",
          flush=True,
      )
      return adapter

    original_log = logger.log
    wrapper = self

    def logged(*args: Any, **kwargs: Any) -> Any:
      result = original_log(*args, **kwargs)
      iteration = kwargs.get("it", args[0] if args else None)
      wrapper._service(iteration=iteration)
      return result

    logger.log = logged
    self._runner_hooked = True
    return adapter


def wrap(env: Any, **kwargs: Any) -> RlMcpEnvWrapper:
  """Wrap a task environment so rlmcp can watch and steer the run.

  See :class:`RlMcpEnvWrapper` for the keyword arguments.
  """
  return RlMcpEnvWrapper(env, **kwargs)
