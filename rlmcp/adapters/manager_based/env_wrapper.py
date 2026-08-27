"""The one line you add to a training script, for any manager-based backend.

``RlMcpEnvWrapper`` sits between the task environment and the RL runner. It
forwards every attribute it does not handle, so anything that worked on the bare
environment keeps working -- including mjlab's own ``RslRlVecEnvWrapper``, which
reaches through ``.unwrapped``.

What it adds:

* per-step hooks that feed traces and video capture,
* a clip of the policy every so many iterations, filed in the run record,
* an optional live view of the run in a browser, over viser,
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

from rlmcp.adapters.rsl_rl_runner import RslRlRunnerAdapter
from rlmcp.core.controller import RlMcp, SessionStopped
from rlmcp.core.curriculum import StageSchedule
from rlmcp.extensions import discover as discover_extensions

CurriculumArg = Union[None, str, StageSchedule, Sequence[Any]]


class TrainingStopped(SessionStopped):
  """Raised inside the training loop when an agent asks training to stop.

  The training entrypoint is expected to catch this, save a final checkpoint and
  exit cleanly -- rsl_rl's ``learn()`` has no stop hook of its own.

  It is a :class:`~rlmcp.core.controller.SessionStopped`, which is the same
  signal named for what it is rather than for the loop it usually interrupts:
  the wrapper services a play session too, where there is no training to stop
  and no checkpoint to save, and ``rlmcp play`` catches the base class. Nothing
  that catches ``TrainingStopped`` needs to change.
  """


class RlMcpEnvWrapper:
  """Transparent wrapper that exposes a training run to rlmcp.

  Backends subclass this and answer two questions: which
  :class:`~rlmcp.adapters.base.SimAdapter` speaks to their environment
  (:meth:`build_sim_adapter`), and whether they want anything checked at
  startup (:meth:`startup_checks`). Everything else -- servicing, telemetry,
  curricula, records, progress clips -- is the same work whichever simulator is
  underneath, which is why it is written once.

  This class is never wrapped around an environment directly -- it has no
  simulator to talk to. The ``wrap()`` a training script calls lives in the
  backend package, next to the subclass it builds: :func:`rlmcp.wrap` and
  :func:`rlmcp.adapters.mjlab.wrap` for mjlab,
  :func:`rlmcp.adapters.isaaclab.wrap` for IsaacLab.
  """

  def build_sim_adapter(self, env: Any, robot_name: Optional[str]) -> Any:
    raise NotImplementedError(
        f"{type(self).__name__} does not say which SimAdapter speaks to its "
        "environment. A backend subclasses RlMcpEnvWrapper, overrides "
        "build_sim_adapter(), and exports its own wrap() -- see "
        "rlmcp/adapters/mjlab/env_wrapper.py, which is 60 lines.")

  def startup_checks(self) -> None:
    """Cheap diagnostics run once, at wrap time. Never fatal."""

  def __init__(
      self,
      env: Any,
      session_dir: Optional[Path | str] = None,
      curriculum: CurriculumArg = None,
      service_every_steps: int = 24,
      robot_name: Optional[str] = None,
      task_id: str = "",
      session_kind: str = "",
      trace_capacity: int = 6000,
      curriculum_kwargs: Optional[Dict[str, Any]] = None,
      extensions: Optional[Sequence[str]] = None,
      exclude_extensions: Optional[Sequence[str]] = None,
      record_run: Optional[str] = None,
      records_root: Optional[str] = None,
      record_slot: str = "",
      record_strict: bool = False,
      code_root: Optional[str] = None,
      video_every: Any = None,
      video_seconds: float = 4.0,
      video_env_id: int = 0,
      video_budget_mb: Optional[float] = None,
      viser: bool = False,
      viser_port: Optional[int] = None,
      viser_host: Optional[str] = None,
      viser_fps: Optional[float] = None,
      viser_env_id: int = 0,
      viser_realtime: bool = False,
      viser_buffer_seconds: Optional[float] = None,
  ):
    self.env = env
    self.service_every_steps = max(1, int(service_every_steps))

    sim_adapter = self.build_sim_adapter(self.unwrapped, robot_name)
    session_dir = Path(session_dir) if session_dir else Path.cwd() / "rlmcp_session"

    from rlmcp.records.link import open_link

    records = open_link(record_run, root=records_root, slot=record_slot,
                        strict=record_strict, code_root=code_root)

    self.rlmcp = RlMcp(
        sim_adapter=sim_adapter,
        runner_adapter=None,
        session_dir=session_dir,
        curriculum=None,  # Set below, once extensions can inform the plan.
        trace_capacity=trace_capacity,
        records=records,
        video_every=video_every,
        video_seconds=video_seconds,
        video_env_id=video_env_id,
        **({} if video_budget_mb is None else {"video_budget_mb": video_budget_mb}),
        viser=viser,
        viser_env_id=viser_env_id,
        viser_realtime=viser_realtime,
        **({} if viser_buffer_seconds is None
           else {"viser_buffer_seconds": viser_buffer_seconds}),
        **({} if viser_port is None else {"viser_port": viser_port}),
        **({} if viser_host is None else {"viser_host": viser_host}),
        **({} if viser_fps is None else {"viser_fps": viser_fps}),
        session_info={
            # Empty means "a training run", which is what the controller
            # already writes; only a play session needs to say otherwise.
            **({"kind": session_kind} if session_kind else {}),
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

    records.start(
        str(self.rlmcp.session.dir), self.rlmcp.parameters.get_snapshot()
    )

    self.rlmcp.curriculum = self._resolve_curriculum(
        curriculum, self.rlmcp, curriculum_kwargs or {}
    )

    self._steps = 0
    self._runner_hooked = False
    self._log_sums: Dict[str, torch.Tensor] = {}
    self._log_counts: Dict[str, int] = {}

    self.startup_checks()

    active = self.rlmcp.extensions.names()
    view = self.rlmcp.live_view
    print(
        f"[rlmcp] session ready: {self.rlmcp.session.dir}\n"
        f"[rlmcp] extensions: {', '.join(active) if active else 'none'}\n"
        f"[rlmcp] inspect it with: rlmcp status --session {self.rlmcp.session.dir}",
        flush=True,
    )
    if view.running:
      # The URL is the whole point of having asked for it, so it is said once
      # here rather than only in a status payload somebody has to go and read.
      print(
          f"[rlmcp] watch it live: {view.url}  (also {view.host_url})\n"
          f"[rlmcp] {view.prose()}; the view costs nothing while no browser is "
          "open. Detach it with `rlmcp view --off`",
          flush=True,
      )
    elif view.last_error:
      print(f"[rlmcp] the live view could not start: {view.last_error}", flush=True)

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

  def attach_runner(self, runner: Any) -> RslRlRunnerAdapter:
    """Hook an rsl_rl runner: PPO knobs, checkpoints, exact iteration boundaries.

    The runner's logger is called exactly once per learning iteration, after the
    policy update, which is the safest point to apply parameter changes and to
    block on a pause.
    """
    adapter = RslRlRunnerAdapter(runner)
    self.rlmcp.attach_runner(adapter)

    # The clip schedule is only decided once the runner says how long the run
    # is, and a cadence nobody was told about is one nobody trusts.
    clips = self.rlmcp.progress_video
    if clips.active:
      print(
          f"[rlmcp] progress clips: {clips.seconds:g}s of env {clips.env_id}, "
          f"{clips.cadence.prose()}; each one is filed in the run record "
          f"(budget {clips.budget_mb:g} MB). Change it with "
          "`rlmcp video --every <cadence>`.",
          flush=True,
      )

    # Where "once per learning iteration" lives depends on the runner. mjlab
    # gives its runner a logger object; plain rsl_rl -- what IsaacLab uses --
    # logs from a method on the runner itself. Both are called once per
    # iteration after the policy update, which is the point where a parameter
    # edit cannot race the simulator and a pause can block safely.
    for owner, name in ((getattr(runner, "logger", None), "log"), (runner, "log")):
      if owner is not None and callable(getattr(owner, name, None)):
        self._hook_iteration_boundary(owner, name)
        return adapter

    print(
        "[rlmcp] this runner reports no per-iteration hook; falling back to "
        f"servicing every {self.service_every_steps} steps. Commands still "
        "work -- they are just answered on a step boundary rather than an "
        "iteration one.",
        flush=True,
    )
    return adapter

  def _hook_iteration_boundary(self, owner: Any, name: str) -> None:
    """Wrap one per-iteration call so rlmcp services the run right after it."""
    original = getattr(owner, name)
    wrapper = self

    def logged(*args: Any, **kwargs: Any) -> Any:
      result = original(*args, **kwargs)
      wrapper._service(iteration=wrapper._iteration_from(args, kwargs))
      return result

    setattr(owner, name, logged)
    self._runner_hooked = True

  def _iteration_from(self, args: Any, kwargs: Any) -> Optional[int]:
    """The iteration a log call is about, however that runner spells it.

    mjlab passes ``it=``; rsl_rl passes a ``locs`` dict carrying the loop
    variable. Neither is guessed at: an unreadable call returns None, and the
    controller then asks the runner adapter, which knows.
    """
    if "it" in kwargs:
      return kwargs["it"]
    for value in list(args) + list(kwargs.values()):
      if isinstance(value, int):
        return value
      if isinstance(value, dict):
        for key in ("it", "current_learning_iteration", "iteration"):
          if isinstance(value.get(key), int):
            return value[key]
    return None

