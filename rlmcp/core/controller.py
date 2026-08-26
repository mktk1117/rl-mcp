"""The controller: one object that owns the agent-facing command surface.

The training loop calls exactly two methods -- :meth:`RlMcp.on_step` once per
environment step and :meth:`RlMcp.service` once per learning iteration. Every
agent command is executed inside ``service``, i.e. between rollout batches,
which is why parameter edits never race the simulator.

Commands that need to watch the robot move (video, traces, diagnosis) cannot
finish inside a single service tick. Those return a *deferred* job: the request
stays open, ``on_step`` feeds it frames or samples during the next rollout, and
the response is written once the job completes. The protocol is open --
:class:`DeferredJob` -- so an extension command can defer the same way the
built-ins do ("watch the robot for N steps and report").
"""

from __future__ import annotations

import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from rlmcp.adapters.base import NotSupported, RunnerAdapter, SimAdapter
from rlmcp.core import diagnostics as diag
from rlmcp.core.curriculum import StageSchedule
from rlmcp.core.extensions import Extension, ExtensionContext, ExtensionRegistry
from rlmcp.core.parameters.registry import ParameterRegistry
from rlmcp.core.telemetry.buffer import TelemetryBuffer
from rlmcp.core.telemetry import plotter
from rlmcp.core.telemetry.trace import TraceRecorder
from rlmcp.session import Request, Response, Session

_MAX_VIDEO_SECONDS = 20.0
_MAX_TRACE_SECONDS = 60.0

DEFAULT_JOB_TIMEOUT_S = 90.0
"""Wall-clock budget a deferred job gets before it is failed truthfully.

Checked at every service boundary (and inside the pause loop), so a job whose
steps never come -- a paused run, a stalled loop -- answers its waiting client
with an error instead of hanging it. Per-job overridable via the
:class:`DeferredJob` constructor.
"""


MAX_CONCURRENT_JOBS = 4
"""In-flight deferred jobs allowed at once.

Each job costs a render or a state sample per environment step while it
collects; beyond a few of them "the training is being watched" becomes "the
training is busy being watched". Memory bounds the cap too: a video job holds
its raw frames in RAM until it encodes, and a max-length clip is ~1000 frames
(:data:`_MAX_VIDEO_SECONDS` at a 50 Hz step) at roughly a megabyte per decoded
frame -- so four in flight is already a multi-GB worst case. Requests past the
cap are refused with a busy error naming what is in flight.
"""


class SessionStopped(RuntimeError):
  """Raised inside a stepping loop when an agent asks the session to stop.

  Stopping is delivered by unwinding, not by a return value: the loop polls
  :meth:`RlMcp.should_stop` at each service point and raises, because the
  backend integration is usually inside somebody else's ``learn()`` or viewer
  loop with no place to return to (see :meth:`RunnerAdapter.request_stop`).

  Whoever owns the loop catches this and finishes cleanly. A trainer saves a
  final checkpoint and exits; a play session closes its viewer and reports the
  stop as a result. Neither should let it reach a user as a traceback -- an
  asked-for stop is not a crash.

  :class:`~rlmcp.adapters.mjlab.env_wrapper.TrainingStopped` is this exception
  under the name the training entrypoints already catch.
  """


class DeferredJob:
  """A command that needs simulation steps before it can answer.

  Protocol: a command handler -- built-in or extension -- returns a
  ``DeferredJob`` instead of a result. The controller schedules it (refusing
  while paused, and beyond :data:`MAX_CONCURRENT_JOBS`), calls :meth:`feed`
  once per environment step until :attr:`ready`, then :meth:`complete` at the
  next service boundary, and writes the return value as the command's
  response. Each job keeps its own state (frame list, trace recorder), which
  is what makes concurrent jobs legal.

  Failure semantics: a raise inside :meth:`feed` stops collection and records
  the error (:meth:`fail`); :meth:`complete` still runs and decides whether
  partial data plus the error as a note is an answer, or raises to fail the
  request. The wall-clock timeout arrives the same way. A raise inside
  :meth:`complete` fails the request with that error.
  """

  kind: str = "job"

  def __init__(
      self,
      env_id: int = 0,
      steps_needed: int = 1,
      timeout_s: float = DEFAULT_JOB_TIMEOUT_S,
  ):
    self.req_id = ""  # Assigned when the controller schedules the job.
    self.env_id = int(env_id)
    self.steps_remaining = max(1, int(steps_needed))
    self.timeout_s = float(timeout_s)
    self.started_at = time.time()
    self.error: Optional[str] = None
    self.cancelled: Optional[str] = None
    # True for jobs submitted through the ExtensionContext with no request
    # waiting: their outcome goes to the session event log, not the outbox.
    self.session_event_only = False

  @property
  def ready(self) -> bool:
    """Whether the job should complete at the next service boundary."""
    return self.steps_remaining <= 0 or self.error is not None

  @property
  def timed_out(self) -> bool:
    return time.time() - self.started_at > self.timeout_s

  def fail(self, message: str) -> None:
    """Stop collecting; :meth:`complete` will see ``message`` as the error."""
    self.error = message

  def cancel(self, reason: str) -> None:
    """Mark cancelled; the controller answers the requester and drops the job."""
    self.cancelled = reason

  def feed(self, lab: "RlMcp") -> None:
    """Collect one step's worth of data; called once per environment step.

    Implementations pull what they need from ``lab`` -- ``lab.sim.render`` for
    frames, ``lab.sim.sample_state`` for signals -- and decrement
    :attr:`steps_remaining` for each step they consume. Raising fails the job
    truthfully: the controller records the error via :meth:`fail`.
    """
    raise NotImplementedError

  def complete(self, lab: "RlMcp") -> Dict[str, Any]:
    """Produce the response payload; called once at a service boundary."""
    raise NotImplementedError

  def describe(self) -> Dict[str, Any]:
    """Status-payload description of this in-flight job."""
    return {
        "req_id": self.req_id,
        "kind": self.kind,
        "env_id": self.env_id,
        "steps_remaining": self.steps_remaining,
        "elapsed_s": round(time.time() - self.started_at, 1),
        "timeout_s": self.timeout_s,
    }


class _VideoJob(DeferredJob):
  """Collect rendered frames of one env, then encode them as a clip."""

  kind = "video"

  def __init__(
      self,
      env_id: int,
      steps_needed: int,
      fps: int,
      seconds: float,
      where: Optional[Dict[str, Any]],
  ):
    super().__init__(env_id=env_id, steps_needed=steps_needed)
    self.fps = int(fps)
    self.seconds = float(seconds)
    self.where = where
    self.frames: List[np.ndarray] = []

  def feed(self, lab: "RlMcp") -> None:
    self.frames.append(lab.sim.render(self.env_id))
    self.steps_remaining -= 1

  def complete(self, lab: "RlMcp") -> Dict[str, Any]:
    if self.error and not self.frames:
      raise RuntimeError(self.error)
    if not self.frames:
      raise RuntimeError("No frames captured; is render_mode='rgb_array' enabled?")
    import imageio.v2 as imageio

    path = lab._artifact(f"clip_env{self.env_id}", ".mp4")
    frames = [np.asarray(f).astype(np.uint8) for f in self.frames]
    fps = self.fps
    faststart = True
    try:
      # +faststart moves the moov atom in front of the media data. Without it a
      # player has to fetch the entire file before showing the first frame,
      # which means a clip silently fails to start in a preview pane or a chat
      # client -- the file is fine, it just never plays where it is looked at.
      imageio.mimsave(
          path, frames, fps=fps, macro_block_size=1,
          output_params=["-movflags", "+faststart"],
      )
    except TypeError:
      faststart = False
      imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
    except Exception:
      faststart = False
      path = path.with_suffix(".gif")
      imageio.mimsave(path, frames, fps=min(fps, 30))
    return {
        "video_path": str(path),
        "env_id": self.env_id,
        "num_frames": len(frames),
        "fps": fps,
        "seconds": round(len(frames) / max(fps, 1), 2),
        "size_mb": round(path.stat().st_size / 1e6, 2),
        "faststart": faststart,
        "where": self.where,
        "note": self.error,
    }


class _TraceJob(DeferredJob):
  """Record per-step signals of one env into a job-private recorder.

  Every trace/diagnose job owns its :class:`TraceRecorder`, so concurrent
  traces -- two envs at once, or the same env twice -- stay env-pure instead
  of interleaving samples in a shared buffer.
  """

  def __init__(
      self,
      kind: str,
      env_id: int,
      steps_needed: int,
      recorder: TraceRecorder,
      plot: bool,
      where: Optional[Dict[str, Any]],
      seconds: float,
  ):
    super().__init__(env_id=env_id, steps_needed=steps_needed)
    self.kind = kind  # "trace" or "diagnose".
    self.recorder = recorder
    self.plot = plot
    self.where = where
    self.seconds = float(seconds)

  def feed(self, lab: "RlMcp") -> None:
    # Contract (rlmcp.adapters.base): sample_state with nothing to offer
    # RAISES NotSupported, which surfaces truthfully as this job's error. A
    # falsy-but-present sample means "nothing this step": skip the recording,
    # consume the step.
    sample = lab.sim.sample_state(self.env_id)
    if sample:
      self.recorder.record(sample)
    self.steps_remaining -= 1

  def complete(self, lab: "RlMcp") -> Dict[str, Any]:
    self.recorder.disarm()
    data = self.recorder.snapshot()
    if not data:
      raise RuntimeError(self.error or "Trace captured no samples.")
    labels = {ch: self.recorder.labels(ch) for ch in data if ch != "time"}
    lab.last_trace = data
    lab.last_trace_labels = labels
    dt = lab._safe(lab.sim.step_dt, 0.02)
    report = diag.analyze_trace(data, labels, dt=dt)
    lab.last_trace_report = report

    npz_path = self.recorder.save_npz(
        lab._artifact(f"trace_env{self.env_id}", ".npz"))
    result: Dict[str, Any] = {
        "trace_path": str(npz_path),
        "env_id": self.env_id,
        "where": self.where,
        "report": report,
    }
    if self.plot:
      png = plotter.plot_trace(
          data,
          labels,
          title=(
              f"env {self.env_id}"
              + (f" ({self.where})" if self.where else "")
              + f" @ iteration {lab.iteration}"
          ),
      )
      result["image_path"] = str(
          lab._write_artifact(f"trace_env{self.env_id}", ".png", png))
    if self.error:
      result["note"] = self.error
    lab.session.append_event(
        "trace",
        {"iteration": lab.iteration, "env_id": self.env_id,
         "verdict": report.get("verdict", [])},
    )
    return result

  def describe(self) -> Dict[str, Any]:
    out = super().describe()
    out["records"] = self.recorder.num_records
    return out


class RlMcp:
  """Agent-facing control surface over a live training run."""

  def __init__(
      self,
      sim_adapter: SimAdapter,
      runner_adapter: Optional[RunnerAdapter] = None,
      session_dir: Path | str = "./rlmcp_session",
      curriculum: Optional[StageSchedule] = None,
      session_info: Optional[Dict[str, Any]] = None,
      trace_capacity: int = 6000,
      extensions: Optional[Sequence[Extension]] = None,
      records: Optional[Any] = None,
  ):
    self.sim = sim_adapter
    self.runner = runner_adapter
    self.curriculum = curriculum
    self.extensions = ExtensionRegistry(list(extensions or []))
    self.records = records  # A RecordLink, or None for an unrecorded run.

    self.session = Session(session_dir).create(
        {
            "kind": "rlmcp-training-session",
            **(session_info or {}),
        }
    )
    self.parameters = ParameterRegistry()
    self.telemetry = TelemetryBuffer(maxlen=20000, on_drop=self._on_telemetry_drop)

    # Ceiling on steps per trace job; each job allocates its own recorder.
    self._trace_capacity = int(trace_capacity)
    self.last_trace: Dict[str, np.ndarray] = {}
    self.last_trace_labels: Dict[str, List[str]] = {}
    self.last_trace_report: Dict[str, Any] = {}

    self.paused = False
    self.step_once_requested = False
    self.stop_requested = False
    self.stop_reason = ""
    self.iteration = 0
    self.total_env_steps = 0

    self._jobs: List[DeferredJob] = []
    self._falsifier_fired: Optional[int] = None
    self._handlers: Dict[str, Callable[..., Any]] = {}
    self._handler_owner: Dict[str, str] = {}
    # Extension hook failures become session events, once per (extension, hook).
    self.extensions.set_error_sink(self._on_extension_error)
    self._register_handlers()
    self._discover_parameters()
    self._defaults = self.parameters.get_snapshot()

    self._extension_context = ExtensionContext(
        write_artifact=self.write_artifact,
        telemetry=self.telemetry,
        append_event=self.session.append_event,
        submit_job=self.submit_job,
        pending_jobs=lambda: [j.describe() for j in self._jobs],
    )
    self.extensions.bind_all(self._extension_context)

    self.session.publish_params(self.parameters.export_schema_json())
    self.session.append_event(
        "session_start",
        {
            "session_dir": str(self.session.dir),
            "commands": sorted(self._handlers),
            "extensions": self.extensions.names(),
        },
    )

  # Setup.

  @staticmethod
  def _safe(fn: Callable[[], Any], default: Any) -> Any:
    try:
      return fn()
    except Exception:
      return default

  def _on_extension_error(self, extension_name: str, hook: str, message: str) -> None:
    """Registry error sink: a failing (extension, hook) pair, reported once."""
    self.session.append_event(
        "extension_error",
        {"iteration": self.iteration, "extension": extension_name,
         "hook": hook, "error": message},
    )

  def _on_telemetry_drop(self, key: str, value: Any) -> None:
    """First sight of a non-scalar metric value: say so instead of losing it."""
    self.session.append_event(
        "telemetry_drop",
        {"iteration": self.iteration, "key": key, "value": repr(value),
         "note": "value is not coercible to float; dropped from telemetry"},
    )

  def _discover_parameters(self) -> None:
    # Which keys write through the sim adapter (vs the runner); consulted when
    # deciding whether the sim's last_set_notes() belong to a given write.
    self._sim_param_keys: set = set()
    for spec in self.sim.discover_parameters():
      self._sim_param_keys.add(spec.key)
      self.parameters.register(
          spec=spec,
          setter=lambda value, k=spec.key: self.sim.set_parameter(k, value),
          getter=lambda k=spec.key: self.sim.get_parameter(k),
      )
    if self.runner is not None:
      for spec in self.runner.discover_hyperparameters():
        self.parameters.register(
            spec=spec,
            setter=lambda value, k=spec.key: self.runner.set_hyperparameter(k, value),
            getter=lambda k=spec.key: self.runner.get_hyperparameter(k),
        )

  def add_extension(self, extension: Extension) -> bool:
    """Register an extension after construction, adding its commands.

    Extensions are usually built by the wrapper once the controller exists, so
    that they can write artifacts through it. Command names are first-wins --
    the same policy as construction-time registration -- with a clash logged
    naming both sides; the extension then receives the
    :class:`~rlmcp.core.extensions.ExtensionContext` via ``bind``.
    """
    if not self.extensions.add(extension):
      return False
    self._merge_extension_commands(extension)
    self.extensions.bind(extension, self._extension_context)
    return True

  def _merge_extension_commands(self, extension: Extension) -> None:
    """Merge one extension's verbs into the dispatch table, first-wins.

    Both registration paths (the construction batch and :meth:`add_extension`)
    come through here, so the conflict policy cannot drift between them: the
    earlier owner keeps the verb, and the clash is logged naming both sides.
    """
    try:
      contributed = extension.commands()
    except Exception as exc:
      self._on_extension_error(extension.name, "commands", str(exc))
      return
    for name, handler in contributed.items():
      owner = self._handler_owner.get(name)
      if owner is not None:
        self.session.append_event(
            "command_conflict",
            {"iteration": self.iteration, "command": name,
             "kept": owner, "ignored": extension.name},
        )
        continue
      self._handlers[name] = handler
      self._handler_owner[name] = extension.name

  def attach_runner(self, runner_adapter: RunnerAdapter) -> None:
    """Wire in the RL runner after construction (hyperparameters, checkpoints)."""
    self.runner = runner_adapter
    for spec in runner_adapter.discover_hyperparameters():
      self.parameters.register(
          spec=spec,
          setter=lambda value, k=spec.key: self.runner.set_hyperparameter(k, value),
          getter=lambda k=spec.key: self.runner.get_hyperparameter(k),
      )
    self._defaults = self.parameters.get_snapshot()
    self.session.publish_params(self.parameters.export_schema_json())

  # Training-loop hooks.

  def on_step(self) -> None:
    """Call once per environment step. Feeds any in-flight recording jobs.

    Each job collects into its own state, so jobs never contend; a job whose
    ``feed`` raises is failed with the error and answers truthfully at the
    next service boundary.
    """
    self.total_env_steps += 1
    if not self._jobs:
      return
    for job in self._jobs:
      if job.ready:
        continue
      try:
        job.feed(self)
      except Exception as exc:
        job.fail(f"{type(exc).__name__}: {exc}")

  def service(
      self,
      iteration: Optional[int] = None,
      metrics: Optional[Dict[str, float]] = None,
  ) -> None:
    """Call once per learning iteration: run commands, publish state, honour pause."""
    if iteration is not None:
      self.iteration = int(iteration)
    elif self.runner is not None:
      self.iteration = self.runner.current_iteration()

    merged = dict(metrics or {})
    merged.update(self._safe(self.sim.summary_metrics, {}))
    merged.update(self.extensions.metrics())
    if self.runner is not None:
      merged.update(self._safe(self.runner.runner_metrics, {}))
    max_len = self._safe(self.sim.max_episode_length, None)
    mean_len = merged.get("Train/mean_episode_length")
    if max_len and isinstance(mean_len, (int, float)):
      merged["rlmcp/episode_length_frac"] = round(float(mean_len) / float(max_len), 4)

    if merged:
      self.telemetry.push(self.iteration, merged)
      self.session.append_metrics(self.iteration, merged)
    self.extensions.on_iteration(self.iteration, merged)

    if self.records is not None:
      self.records.snapshot_config(self.parameters.get_snapshot())
      self.records.heartbeat()
      self._watch_falsifier(merged)

    self._finish_jobs()
    # Curriculum first, then agent commands: an explicit instruction issued this
    # iteration must not be silently reverted by a stage that was applied in the
    # same tick.
    self._advance_curriculum(merged)
    self._drain_inbox()
    self._publish_status()

    # Pause loop: keep answering commands so the agent can inspect and resume.
    while self.paused and not self.stop_requested:
      if self.step_once_requested:
        self.step_once_requested = False
        break
      time.sleep(0.15)
      self._finish_jobs()
      self._drain_inbox()
      self._publish_status()

  def should_stop(self) -> bool:
    return self.stop_requested

  # Command plumbing.

  def _drain_inbox(self) -> None:
    for request in self.session.pop_requests():
      self._execute(request)

  def _execute(self, request: Request) -> None:
    handler = self._handlers.get(request.cmd)
    if handler is None:
      self.session.respond(
          Response(
              req_id=request.req_id,
              ok=False,
              error=(
                  f"Unknown command '{request.cmd}'. "
                  f"Available: {', '.join(sorted(self._handlers))}"
              ),
          )
      )
      return
    try:
      result = handler(**(request.args or {}))
    except TypeError as exc:
      self.session.respond(
          Response(req_id=request.req_id, ok=False, error=f"Bad arguments: {exc}")
      )
      return
    except Exception as exc:
      self.session.respond(
          Response(
              req_id=request.req_id,
              ok=False,
              error=f"{type(exc).__name__}: {exc}",
          )
      )
      self.session.append_event(
          "command_error",
          {"cmd": request.cmd, "error": str(exc), "traceback": traceback.format_exc()},
      )
      return

    if isinstance(result, DeferredJob):
      result.req_id = request.req_id
      try:
        self._schedule_job(result)
      except Exception as exc:
        self.session.respond(
            Response(
                req_id=request.req_id, ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        )
      return

    self.session.respond(Response(req_id=request.req_id, ok=True, result=result))

  def run_command(self, cmd: str, /, **args: Any) -> Any:
    """Run one command in this process, right now, and return its result.

    Every other route into a command goes through the session inbox, because
    every other caller is in another process. An in-process caller -- a play
    session restoring the conditions a checkpoint was trained under -- has no
    inbox to post to and no service loop to wait for, so it needs the handler
    table directly. Errors raise rather than becoming a Response: there is a
    caller here with a stack to unwind into.

    A deferred command is returned as its :class:`DeferredJob` rather than
    submitted; whoever asked for it owns the decision to queue it.
    """
    handler = self._handlers.get(cmd)
    if handler is None:
      raise KeyError(
          f"Unknown command '{cmd}'. This run has: "
          f"{', '.join(sorted(self._handlers))}"
      )
    return handler(**args)

  def _schedule_job(self, job: DeferredJob) -> None:
    """Admit a deferred job for step feeding, or refuse truthfully.

    Refused while paused -- no environment steps happen, so the job could
    never progress and its client would wait on nothing -- and beyond
    :data:`MAX_CONCURRENT_JOBS`. Jobs already in flight when a pause lands
    are not dropped: ``step_once`` feeds them, and the wall-clock timeout
    answers them truthfully if steps never come.
    """
    if self.paused:
      raise RuntimeError(
          f"Cannot start '{job.kind}' while training is paused: deferred "
          "commands collect data during rollout steps, which do not happen "
          "while paused. Resume training, then request it again."
      )
    if len(self._jobs) >= MAX_CONCURRENT_JOBS:
      pending = ", ".join(
          f"'{j.kind}' env {j.env_id} ({j.steps_remaining} steps left)"
          for j in self._jobs
      )
      raise RuntimeError(
          f"Deferred-job limit reached ({MAX_CONCURRENT_JOBS} in flight: "
          f"{pending}). Wait for one to complete, or cancel_job it."
      )
    self._jobs.append(job)

  def submit_job(self, job: DeferredJob) -> Dict[str, Any]:
    """Schedule a deferred job with no request waiting on it.

    This is the :class:`~rlmcp.core.extensions.ExtensionContext` surface for
    jobs started outside a command handler (a handler simply *returns* its
    job). Admission rules are the same -- pause refusal and the concurrency
    cap raise -- and the outcome is appended to the session event log
    (``job_complete`` / ``job_failed``) instead of the outbox.
    """
    if not job.req_id:
      job.req_id = f"ctx-{uuid.uuid4().hex[:12]}"
    job.session_event_only = True
    self._schedule_job(job)
    return job.describe()

  def _respond_job(
      self,
      job: DeferredJob,
      ok: bool,
      result: Any = None,
      error: Optional[str] = None,
  ) -> None:
    """Deliver a job's outcome: the request's response, or a session event."""
    if job.session_event_only:
      self.session.append_event(
          "job_complete" if ok else "job_failed",
          {"req_id": job.req_id, "job_kind": job.kind, "env_id": job.env_id,
           "result": result, "error": error},
      )
      return
    self.session.respond(
        Response(req_id=job.req_id, ok=ok, result=result, error=error))

  @staticmethod
  def _job_event_detail(job: DeferredJob) -> Dict[str, Any]:
    """A job description safe to splat into an event (``kind`` is the event's)."""
    detail = job.describe()
    detail["job_kind"] = detail.pop("kind", job.kind)
    return detail

  def _finish_jobs(self) -> None:
    if not self._jobs:
      return
    remaining: List[DeferredJob] = []
    for job in self._jobs:
      if not job.ready and job.timed_out:
        # Runs in the pause loop too, so a job starved of steps by a pause
        # answers its client instead of hanging until resume.
        self.session.append_event("job_timeout", self._job_event_detail(job))
        job.fail(
            f"Timed out after {job.timeout_s:.0f}s wall-clock with "
            f"{job.steps_remaining} steps still to collect. Deferred jobs "
            "progress only while training steps; if the run is paused or "
            "stalled, resume it and retry."
        )
      if not job.ready:
        remaining.append(job)
        continue
      try:
        result = job.complete(self)
        self._respond_job(job, ok=True, result=result)
      except Exception as exc:
        self._respond_job(job, ok=False, error=f"{type(exc).__name__}: {exc}")
    self._jobs = remaining

  # Artifacts.

  def _artifact(self, stem: str, suffix: str) -> Path:
    return self.session.artifact_path(f"{stem}_it{self.iteration:06d}{suffix}")

  def write_artifact(self, stem: str, suffix: str, payload: bytes) -> Path:
    """Save bytes as a session artifact. Extensions use this for their plots."""
    path = self._artifact(stem, suffix)
    path.write_bytes(payload)
    return path

  _write_artifact = write_artifact

  def _resolve_env_id(
      self,
      env_id: Optional[int] = None,
      where: Optional[Dict[str, Any]] = None,
  ) -> int:
    """Pick an environment: an explicit id, or the first one matching ``where``.

    ``where`` is handed to the extensions, so its vocabulary is whatever this
    environment supports -- ``{"terrain": "pyramid_stairs"}`` on a locomotion
    task. The core does not interpret it.
    """
    if env_id is not None:
      return int(env_id)
    if not where:
      return 0
    return (self._resolve_env_ids(None, where) or [0])[0]

  def _resolve_env_ids(
      self,
      env_ids: Optional[Sequence[int]] = None,
      where: Optional[Dict[str, Any]] = None,
  ) -> Optional[List[int]]:
    """Pick a set of environments: explicit ids, a ``where`` query, or all.

    The plural of :meth:`_resolve_env_id`, and it answers a different question:
    that one picks *an* environment to look at and falls back to env 0, while
    this one returns ``None`` for "all of them" -- because a command that acts
    on environments must act on every one unless somebody narrowed it, and
    silently acting on env 0 alone would be a lie.

    ``where`` is the extensions' vocabulary, exactly as in ``shot --where``, so
    narrowing by a property of the task costs the core no new concepts.
    """
    if env_ids is not None:
      chosen = [int(i) for i in env_ids]
      if not chosen:
        raise ValueError("env_ids was empty; omit it to mean every environment.")
      return chosen
    if not where:
      return None
    candidates = self.extensions.select_envs(**where)
    if candidates is None:
      raise ValueError(
          f"No extension understands {where!r}. This run has: "
          f"{self.extensions.names() or 'no extensions'}."
      )
    if not candidates:
      raise ValueError(
          f"No environment currently matches {where!r}. Check the status payload "
          "for what is actually populated."
      )
    return [int(i) for i in candidates]

  # Handlers.

  def _register_handlers(self) -> None:
    self._handlers = {
        "help": self.cmd_help,
        "status": self.cmd_status,
        "list_parameters": self.cmd_list_parameters,
        "get_parameter": self.cmd_get_parameter,
        "set_parameter": self.cmd_set_parameter,
        "reset_parameters": self.cmd_reset_parameters,
        "reset_envs": self.cmd_reset_envs,
        "list_metrics": self.cmd_list_metrics,
        "get_metrics": self.cmd_get_metrics,
        "plot_metrics": self.cmd_plot_metrics,
        "screenshot": self.cmd_screenshot,
        "record_video": self.cmd_record_video,
        "record_trace": self.cmd_record_trace,
        "plot_trace": self.cmd_plot_trace,
        "diagnose": self.cmd_diagnose,
        "curriculum_status": self.cmd_curriculum_status,
        "curriculum_advance": self.cmd_curriculum_advance,
        "curriculum_goto": self.cmd_curriculum_goto,
        "curriculum_auto": self.cmd_curriculum_auto,
        "pause": self.cmd_pause,
        "resume": self.cmd_resume,
        "step_once": self.cmd_step_once,
        "cancel_job": self.cmd_cancel_job,
        "save_checkpoint": self.cmd_save_checkpoint,
        "list_checkpoints": self.cmd_list_checkpoints,
        "load_checkpoint": self.cmd_load_checkpoint,
        "note": self.cmd_note,
        "feedback": self.cmd_feedback,
        "stop_training": self.cmd_stop_training,
    }
    self._handler_owner = {name: "built-in" for name in self._handlers}
    # Extensions contribute verbs on equal footing with the built-ins: reachable
    # from the CLI, from MCP, and from a curriculum stage's `apply` list.
    for extension in self.extensions:
      self._merge_extension_commands(extension)

  def cmd_help(self) -> Dict[str, Any]:
    """List every command with its one-line docstring."""
    return {
        "commands": {
            name: (fn.__doc__ or "").strip().splitlines()[0]
            for name, fn in sorted(self._handlers.items())
        }
    }

  def cmd_status(self) -> Dict[str, Any]:
    """Current iteration, pause state, curriculum stage and headline metrics."""
    return self._status_payload()

  def _status_payload(self) -> Dict[str, Any]:
    latest = self.telemetry.get_latest_metrics()
    # Core keys plus every rlmcp/ metric, so an extension's metrics show up in
    # the headline without the core naming them.
    headline = {
        k: round(v, 5)
        for k, v in latest.items()
        if k.startswith("rlmcp/")
        or k in ("Train/mean_reward", "Train/mean_episode_length", "Loss/learning_rate")
    }
    payload: Dict[str, Any] = {
        "iteration": self.iteration,
        "total_env_steps": self.total_env_steps,
        "num_envs": self._safe(self.sim.num_envs, None),
        "paused": self.paused,
        "stop_requested": self.stop_requested,
        "headline_metrics": headline,
        "num_parameters": len(self.parameters.get_all_specs()),
        "renderer_built": bool(self._safe(
            getattr(self.sim, "renderer_ready", lambda: False), False)),
        "pending_jobs": [j.describe() for j in self._jobs],
    }
    if self.curriculum is not None:
      payload["curriculum"] = self.curriculum.status(self.iteration)
    described = self.extensions.describe()
    if described:
      payload["extensions"] = described
    if self.records is not None:
      payload["records"] = self.records.status()
      if self._falsifier_fired:
        payload["records"]["falsifier_fired_at"] = self._falsifier_fired
    return payload

  def _watch_falsifier(self, metrics: Dict[str, float]) -> None:
    """Say so, once, the moment the run disproves its own hypothesis.

    Not a stop: a falsified hypothesis is a result, and whether to spend more
    GPU on it is the operator's call. But it should never be discovered days
    later while writing the report.
    """
    if self._falsifier_fired is not None:
      return
    result = self.records.check_falsifier(metrics, iteration=self.iteration)
    if not result.get("fired"):
      return
    self._falsifier_fired = self.iteration
    fired = [c["condition"] for c in result.get("checks", []) if c["fired"]]
    self.session.append_event(
        "falsifier_fired",
        {"iteration": self.iteration, "conditions": fired,
         "prose": result.get("prose", "")},
    )
    print(
        f"[rlmcp] FALSIFIER FIRED at iteration {self.iteration}: "
        f"{', '.join(fired)}\n"
        f"[rlmcp]   {result.get('prose', '')}\n"
        f"[rlmcp] That is a result. Close the run rather than waiting it out.",
        flush=True,
    )

  def _publish_status(self) -> None:
    self.session.publish_status(self._status_payload())

  def cmd_list_parameters(
      self, category: Optional[str] = None, contains: Optional[str] = None
  ) -> Dict[str, Any]:
    """List tunable parameters with current values, bounds and descriptions."""
    schema = self.parameters.export_schema_json()
    items = {
        key: spec
        for key, spec in schema.items()
        if (category is None or spec.get("category") == category)
        and (contains is None or contains.lower() in key.lower())
    }
    return {"count": len(items), "parameters": items}

  def cmd_get_parameter(self, key: str) -> Dict[str, Any]:
    """Read one parameter's live value."""
    return {"key": key, "value": self.parameters.get_value(key)}

  def cmd_set_parameter(
      self, key: str, value: Any, rationale: str = ""
  ) -> Dict[str, Any]:
    """Change a reward weight, randomization range or PPO hyperparameter live."""
    old = self.parameters.get_value(key)
    ok = self.parameters.set_value(key, value)
    new = self.parameters.get_value(key)
    self.session.append_event(
        "set_parameter",
        {"iteration": self.iteration, "key": key, "old": old, "new": new,
         "applied": bool(ok), "rationale": rationale},
    )
    if ok:
      # A refused write changed nothing; do not republish an unchanged schema.
      self.session.publish_params(self.parameters.export_schema_json())

    result: Dict[str, Any] = {
        "key": key, "old_value": old, "new_value": new, "applied": bool(ok)}
    spec = self.parameters.get_spec(key)
    liveness = getattr(spec, "liveness", None)
    if liveness is not None:
      result["liveness"] = getattr(liveness, "value", liveness)
    # Adapter notes: e.g. "takes effect from each env's next reset", or a side
    # effect like a rewritten curriculum table. The sim adapter keeps notes
    # only for its own most recent write, so consult it only when this write
    # both applied and routed through the sim -- a runner hyperparameter write
    # must never pick up a stale note left by an earlier sim write.
    if ok and key in self._sim_param_keys:
      reader = getattr(self.sim, "last_set_notes", None)
      if callable(reader):
        for name, note in (self._safe(reader, {}) or {}).items():
          result.setdefault(name, note)
    return result

  def cmd_reset_parameters(
      self, keys: Optional[Sequence[str]] = None, rationale: str = ""
  ) -> Dict[str, Any]:
    """Restore parameters to the values they had when training started."""
    targets = list(keys) if keys else list(self._defaults)
    restored = {}
    for key in targets:
      if key not in self._defaults:
        continue
      old = self.parameters.get_value(key)
      if old == self._defaults[key]:
        continue
      ok = self.parameters.set_value(key, self._defaults[key])
      entry: Dict[str, Any] = {
          "old": old, "new": self._defaults[key], "applied": bool(ok)}
      # Same surfacing as cmd_set_parameter: liveness, and -- for a write that
      # both applied and routed through the sim -- the adapter's per-write
      # notes ("takes effect from each env's next reset", curriculum rewrites).
      spec = self.parameters.get_spec(key)
      liveness = getattr(spec, "liveness", None)
      if liveness is not None:
        entry["liveness"] = getattr(liveness, "value", liveness)
      if ok and key in self._sim_param_keys:
        reader = getattr(self.sim, "last_set_notes", None)
        if callable(reader):
          for name, note in (self._safe(reader, {}) or {}).items():
            entry.setdefault(name, note)
      restored[key] = entry
    if restored:
      self.session.append_event(
          "reset_parameters",
          {"iteration": self.iteration, "restored": restored, "rationale": rationale},
      )
      self.session.publish_params(self.parameters.export_schema_json())
    return {"restored_count": len(restored), "restored": restored}

  def cmd_reset_envs(
      self,
      env_ids: Optional[Sequence[int]] = None,
      where: Optional[Dict[str, Any]] = None,
      rationale: str = "",
  ) -> Dict[str, Any]:
    """Start fresh episodes in some or all environments.

    Episodes, not parameter values: ``reset_parameters`` puts the *knobs* back
    where they started and leaves the robot mid-fall, this puts the robot back
    on its feet and leaves the knobs alone. They are different verbs on purpose
    and neither implies the other.

    Narrow it with ``env_ids``, or with ``where`` in whatever vocabulary this
    run's extensions provide -- ``where={"terrain": "pyramid_stairs"}``
    restarts only the environments on that part of the task. Both omitted means
    every environment.

    Lands at a service boundary like every other command, so it cannot race the
    simulator, and is refused truthfully on a backend with no reset path.
    """
    selected = self._resolve_env_ids(env_ids, where)
    try:
      detail = self.sim.reset_envs(selected) or {}
    except NotSupported as exc:
      raise RuntimeError(
          f"This backend cannot reset episodes on demand ({exc}). Nothing was "
          "reset. An environment that can be restarted implements "
          "SimAdapter.reset_envs; one that cannot is left alone rather than "
          "pretending."
      ) from exc
    counted = (
        len(selected) if selected is not None
        else (self._safe(self.sim.num_envs, None) or 0)
    )
    result: Dict[str, Any] = {
        "scope": "all" if selected is None else "selection",
        "env_ids": selected,
        "num_reset": int(detail.get("num_reset", counted)),
        **{k: v for k, v in detail.items() if k != "num_reset"},
    }
    self.session.append_event(
        "reset_envs",
        {"iteration": self.iteration, "rationale": rationale, **result},
    )
    return result

  def _default_metric_names(self, limit: int = 4) -> List[str]:
    """A sensible default selection, drawn from what this run actually records."""
    available = set(self.telemetry.list_metrics())
    chosen = [
        k for k in ("Train/mean_reward", "Train/mean_episode_length") if k in available
    ]
    for key in sorted(k for k in available if k.startswith("rlmcp/")):
      if len(chosen) >= limit:
        break
      chosen.append(key)
    return chosen or sorted(available)[:limit]

  def cmd_list_metrics(self, contains: Optional[str] = None) -> Dict[str, Any]:
    """List every metric name recorded so far."""
    names = self.telemetry.list_metrics()
    if contains:
      names = [n for n in names if contains.lower() in n.lower()]
    return {"count": len(names), "metrics": sorted(names)}

  def cmd_get_metrics(
      self,
      names: Optional[Sequence[str]] = None,
      last_n: int = 30,
      summarize: bool = True,
  ) -> Dict[str, Any]:
    """Fetch recent values (and a trend summary) for selected metrics."""
    names = list(names) if names else self._default_metric_names()
    out: Dict[str, Any] = {"metrics": {}}
    rows: List[Dict[str, Any]] = []
    for name in names:
      series = self.telemetry.get_series(name, last_n=last_n)
      out["metrics"][name] = [[int(i), round(float(v), 6)] for i, v in series]
    if summarize:
      history = self.telemetry.as_rows(last_n=max(last_n, 60))
      rows = history
      out["summary"] = diag.summarize_metric_history(rows, list(names))
    return out

  def cmd_plot_metrics(
      self,
      names: Optional[Sequence[str]] = None,
      last_n: int = 400,
      smooth: int = 5,
      title: Optional[str] = None,
  ) -> Dict[str, Any]:
    """Render selected metric curves to a PNG and return its path."""
    names = list(names) if names else self._default_metric_names()
    series = {n: self.telemetry.get_series(n, last_n=last_n) for n in names}
    markers = []
    if self.curriculum is not None:
      markers = [
          (float(t["iteration"]), str(t["to"]))
          for t in self.curriculum.history
          if isinstance(t.get("iteration"), (int, float))
      ]
    png = plotter.plot_metric_series(
        series, title=title or f"metrics @ iteration {self.iteration}",
        smooth_window=max(1, int(smooth)), markers=markers,
    )
    path = self._write_artifact("metrics", ".png", png)
    return {"image_path": str(path), "metrics": list(names)}

  def cmd_screenshot(
      self,
      env_id: Optional[int] = None,
      where: Optional[Dict[str, Any]] = None,
  ) -> Dict[str, Any]:
    """Render one frame of an environment, chosen by id or by description."""
    resolved = self._resolve_env_id(env_id, where)
    frame = self.sim.render(resolved)
    from PIL import Image

    path = self._artifact(f"shot_env{resolved}", ".png")
    Image.fromarray(np.asarray(frame).astype(np.uint8)).save(path)
    return {
        "image_path": str(path),
        "env_id": resolved,
        "where": where,
        "size": [int(frame.shape[1]), int(frame.shape[0])],
    }

  def cmd_record_video(
      self,
      seconds: float = 4.0,
      env_id: Optional[int] = None,
      where: Optional[Dict[str, Any]] = None,
      fps: Optional[int] = None,
  ) -> DeferredJob:
    """Record a clip of training as it happens and return the video path."""
    resolved = self._resolve_env_id(env_id, where)
    dt = self._safe(self.sim.step_dt, 0.02)
    seconds = float(min(max(seconds, 0.2), _MAX_VIDEO_SECONDS))
    steps = max(1, int(round(seconds / dt)))
    return _VideoJob(
        env_id=resolved,
        steps_needed=steps,
        fps=int(fps or round(1.0 / dt)),
        seconds=seconds,
        where=where,
    )

  def cmd_record_trace(
      self,
      seconds: float = 4.0,
      env_id: Optional[int] = None,
      where: Optional[Dict[str, Any]] = None,
  ) -> DeferredJob:
    """Record per-step joint/base signals for one env and summarise them."""
    return self._start_trace_job("trace", seconds, env_id, where, plot=False)

  def cmd_diagnose(
      self,
      seconds: float = 4.0,
      env_id: Optional[int] = None,
      where: Optional[Dict[str, Any]] = None,
  ) -> DeferredJob:
    """Record a trace, analyse smoothness/tracking/gait, and plot it."""
    return self._start_trace_job("diagnose", seconds, env_id, where, plot=True)

  def _start_trace_job(
      self,
      kind: str,
      seconds: float,
      env_id: Optional[int],
      where: Optional[Dict[str, Any]],
      plot: bool,
  ) -> DeferredJob:
    dt = self._safe(self.sim.step_dt, 0.02)
    resolved = self._resolve_env_id(env_id, where)
    seconds = float(min(max(seconds, 0.2), _MAX_TRACE_SECONDS))
    steps = min(self._trace_capacity, max(8, int(round(seconds / dt))))
    labels = self._safe(getattr(self.sim, "trace_labels", lambda: {}), {})
    # The recorder belongs to this job alone, so concurrent trace/diagnose
    # jobs (any mix of envs) are legal -- each records only its own env.
    recorder = TraceRecorder(capacity=steps, dt=dt)
    recorder.arm(
        env_id=resolved,
        num_steps=steps,
        labels=labels,
        meta={"iteration": self.iteration, "where": str(where), "env_id": resolved},
    )
    return _TraceJob(
        kind=kind,
        env_id=resolved,
        steps_needed=steps,
        recorder=recorder,
        plot=plot,
        where=where,
        seconds=seconds,
    )

  def cmd_cancel_job(self, req_id: str, reason: str = "") -> Dict[str, Any]:
    """Cancel an in-flight deferred job; its requester gets a truthful error."""
    job = next((j for j in self._jobs if j.req_id == req_id), None)
    if job is None:
      pending = ", ".join(j.req_id for j in self._jobs) or "none"
      raise ValueError(
          f"No in-flight job with req_id '{req_id}'. In flight: {pending}.")
    reason = reason or "cancelled by request"
    job.cancel(reason)
    self._jobs.remove(job)
    described = job.describe()
    self._respond_job(
        job, ok=False,
        error=f"Cancelled after {described['elapsed_s']}s: {reason}",
    )
    self.session.append_event(
        "job_cancelled", {**self._job_event_detail(job), "reason": reason})
    return {"cancelled": True, **described}

  def cmd_plot_trace(
      self,
      channels: Optional[Sequence[str]] = None,
      components: Optional[Sequence[str]] = None,
      title: Optional[str] = None,
  ) -> Dict[str, Any]:
    """Re-plot the last recorded trace, optionally filtered to some joints."""
    if not self.last_trace:
      raise RuntimeError("No trace recorded yet. Run record_trace or diagnose first.")
    png = plotter.plot_trace(
        self.last_trace,
        self.last_trace_labels,
        channels=channels,
        components=components,
        title=title or f"trace @ iteration {self.iteration}",
    )
    return {"image_path": str(self._write_artifact("trace", ".png", png))}

  def cmd_curriculum_status(self) -> Dict[str, Any]:
    """Current stage, promotion conditions, and how close they are to met."""
    if self.curriculum is None:
      return {"enabled": False, "note": "No curriculum schedule attached to this run."}
    latest = self.telemetry.get_latest_metrics()
    return {
        "enabled": True,
        **self.curriculum.status(self.iteration),
        "progress": self.curriculum.check(self.iteration, latest),
    }

  def cmd_curriculum_advance(self, reason: str = "manual") -> Dict[str, Any]:
    """Promote to the next curriculum stage now."""
    if self.curriculum is None:
      raise RuntimeError("No curriculum schedule attached to this run.")
    transition = self.curriculum.advance(self.iteration, reason=reason)
    if transition is None:
      return {"transitioned": False, "note": "Already at the final stage."}
    self._apply_stage(transition)
    return {"transitioned": True, **transition, "stage": self.curriculum.current.to_dict()}

  def cmd_curriculum_goto(self, stage: str, reason: str = "manual") -> Dict[str, Any]:
    """Jump to a named curriculum stage (forward or backward)."""
    if self.curriculum is None:
      raise RuntimeError("No curriculum schedule attached to this run.")
    transition = self.curriculum.goto_named(stage, self.iteration, reason=reason)
    self._apply_stage(transition)
    return {"transitioned": True, **transition, "stage": self.curriculum.current.to_dict()}

  def cmd_curriculum_auto(self, enabled: bool = True) -> Dict[str, Any]:
    """Turn automatic stage promotion on or off."""
    if self.curriculum is None:
      raise RuntimeError("No curriculum schedule attached to this run.")
    self.curriculum.auto_promote = bool(enabled)
    self.session.append_event(
        "curriculum_auto", {"iteration": self.iteration, "enabled": bool(enabled)}
    )
    return {"auto_promote": self.curriculum.auto_promote}

  def cmd_pause(self) -> Dict[str, Any]:
    """Pause training between iterations (commands keep working)."""
    self.paused = True
    self.session.append_event("pause", {"iteration": self.iteration})
    return {"paused": True, "iteration": self.iteration}

  def cmd_resume(self) -> Dict[str, Any]:
    """Resume training after a pause."""
    self.paused = False
    self.session.append_event("resume", {"iteration": self.iteration})
    return {"paused": False, "iteration": self.iteration}

  def cmd_step_once(self) -> Dict[str, Any]:
    """Run exactly one more iteration while paused, then pause again."""
    self.step_once_requested = True
    return {"stepping": True, "iteration": self.iteration}

  def cmd_save_checkpoint(self, tag: str = "", note: str = "") -> Dict[str, Any]:
    """Save policy weights plus curriculum, parameters and extension state."""
    if self.runner is None:
      raise RuntimeError("No runner attached; cannot save a checkpoint.")
    tag = tag or f"it{self.iteration:06d}"
    path = self.session.dir / "checkpoints" / f"{tag}.pt"
    infos = {
        # The runner owns "env_state": mjlab's save replaces it wholesale with
        # its own payload (infos = {**infos, "env_state": {...}}), so nothing
        # rlmcp needs back may live under that key. We still offer the sim's
        # state there -- that slot is the runner's to manage -- and everything
        # rlmcp must get back rides under "rlmcp", which the runner preserves.
        "env_state": self._safe(self.sim.get_env_state, {}),
        "rlmcp": {
            "iteration": self.iteration,
            "note": note,
            "parameters": self.parameters.get_snapshot(),
            "curriculum": self.curriculum.to_dict() if self.curriculum else None,
            "extensions": self.extensions.snapshot(),
        },
    }
    saved = self.runner.save_checkpoint(str(path), infos)
    self.session.append_event(
        "checkpoint", {"iteration": self.iteration, "tag": tag, "path": saved,
                       "note": note}
    )
    return {"tag": tag, "path": saved, "iteration": self.iteration}

  def cmd_list_checkpoints(self) -> Dict[str, Any]:
    """List checkpoints saved through rlmcp in this session."""
    directory = self.session.dir / "checkpoints"
    if not directory.exists():
      return {"checkpoints": []}
    items = [
        {"tag": p.stem, "path": str(p), "size_mb": round(p.stat().st_size / 1e6, 2)}
        for p in sorted(directory.glob("*.pt"))
    ]
    return {"checkpoints": items}

  def cmd_load_checkpoint(self, path: str, restore_parameters: bool = True) -> Dict[str, Any]:
    """Roll back policy weights (and optionally parameters/curriculum) to a checkpoint."""
    if self.runner is None:
      raise RuntimeError("No runner attached; cannot load a checkpoint.")
    candidate = Path(path)
    if not candidate.exists():
      candidate = self.session.dir / "checkpoints" / f"{path}.pt"
    infos = self.runner.load_checkpoint(str(candidate))
    restored: Dict[str, Any] = {"weights": True, "path": str(candidate)}

    env_state = (infos or {}).get("env_state")
    if env_state:
      self.sim.set_env_state(env_state)
      restored["env_state"] = True

    payload = (infos or {}).get("rlmcp") or {}
    # Ordering constraint: apply the curriculum stage FIRST, then restore the
    # checkpointed snapshots on top. _apply_stage re-applies the stage's entry
    # parameters and actions, so running it afterwards would clobber the
    # snapshot -- hand-tuned parameters and extension state changed after stage
    # entry must win on rollback.
    if restore_parameters and payload.get("curriculum") and self.curriculum is not None:
      self.curriculum = StageSchedule.from_dict(payload["curriculum"])
      self._apply_stage({"from": "?", "to": self.curriculum.current.name,
                         "reason": "checkpoint restore"}, log=False)
      restored["curriculum_stage"] = self.curriculum.current.name
    if restore_parameters and payload.get("parameters"):
      applied = 0
      for key, value in payload["parameters"].items():
        try:
          self.parameters.set_value(key, value)
          applied += 1
        except Exception:
          continue
      restored["parameters_restored"] = applied

    # Extension state rides under "rlmcp" (the runner overwrites "env_state"
    # on save); fall back to the legacy env_state location for checkpoints
    # written before the move. The registry reports per-extension success, and
    # only actual successes are counted -- a payload whose restore raised, or
    # one addressed to an extension this run does not have, was not restored.
    ext_state = payload.get("extensions") or (env_state or {}).get("extensions")
    if ext_state:
      results = self.extensions.restore(ext_state)
      restored["extensions_restored"] = sum(1 for ok in results.values() if ok)

    if restored.get("parameters_restored"):
      # Values moved; republish so params.json reflects the rolled-back config.
      self.session.publish_params(self.parameters.export_schema_json())

    self.session.append_event(
        "load_checkpoint", {"iteration": self.iteration, **restored}
    )
    return restored

  def cmd_note(self, text: str) -> Dict[str, Any]:
    """Write a free-form note into the session's event log."""
    self.session.append_event("note", {"iteration": self.iteration, "text": text})
    return {"logged": True, "iteration": self.iteration}

  def cmd_feedback(
      self,
      text: str,
      kind: str = "steer",
      author: str = "user",
      interpretation: str = "",
  ) -> Dict[str, Any]:
    """Record something a human said about this run, at the current iteration.

    Separate from ``note`` because it is a different kind of fact: a note is the
    agent talking to itself, feedback is a human steering the run. It is stamped
    with the iteration here -- the only place that knows it -- and folded into
    the run record at close-out by ``rlmcp record close``.
    """
    self.session.append_event(
        "feedback",
        {
            "iteration": self.iteration,
            "text": text,
            "feedback_kind": kind,
            "author": author,
            "interpretation": interpretation,
        },
    )
    return {"logged": True, "iteration": self.iteration, "feedback_kind": kind}

  def cmd_stop_training(self, reason: str = "") -> Dict[str, Any]:
    """Ask the training loop to stop cleanly at the next iteration boundary."""
    self.stop_requested = True
    self.stop_reason = reason
    self.paused = False
    if self.runner is not None:
      try:
        self.runner.request_stop()
      except NotSupported:
        pass
    self.session.append_event(
        "stop_requested", {"iteration": self.iteration, "reason": reason}
    )
    return {"stop_requested": True, "iteration": self.iteration, "reason": reason}

  # Curriculum application.

  def _advance_curriculum(self, metrics: Dict[str, float]) -> None:
    if self.curriculum is None:
      return
    if not hasattr(self, "_curriculum_applied"):
      # Apply the opening stage exactly once, so a fresh run starts where the
      # plan says it should rather than wherever the env config happened to be.
      self._curriculum_applied = True
      self._apply_stage(
          {"from": "-", "to": self.curriculum.current.name, "reason": "initial stage"}
      )
    transition = self.curriculum.evaluate(self.iteration, metrics)
    if transition is not None:
      self._apply_stage(transition)

  def _apply_stage(self, transition: Dict[str, Any], log: bool = True) -> None:
    """Push the current stage's intent into the run.

    A stage speaks only in parameters and commands, so it works the same on a
    locomotion run that unlocks terrain and on any other task -- the commands it
    names are whatever this environment's extensions provide.
    """
    stage = self.curriculum.current
    applied: Dict[str, Any] = {}

    for key, value in (stage.parameters or {}).items():
      try:
        self.parameters.set_value(key, value)
        applied.setdefault("parameters", {})[key] = value
      except Exception as exc:
        applied.setdefault("parameter_errors", {})[key] = str(exc)

    for action in stage.apply or []:
      handler = self._handlers.get(action.cmd)
      if handler is None:
        applied.setdefault("action_errors", {})[action.cmd] = (
            f"Unknown command '{action.cmd}'. This run has: "
            f"{', '.join(sorted(self._handlers))}"
        )
        continue
      try:
        handler(**(action.args or {}))
        applied.setdefault("actions", []).append(action.describe())
        # The same call as data, for whoever reads this log back. `describe()`
        # is prose meant for a person and only parses back when every argument
        # has a literal repr; `to_dict()` always survives the round trip. See
        # rlmcp.core.replay, which prefers this and falls back to the prose.
        applied.setdefault("calls", []).append(action.to_dict())
      except Exception as exc:
        applied.setdefault("action_errors", {})[action.cmd] = str(exc)

    if log:
      self.session.append_event(
          "curriculum_stage",
          {"iteration": self.iteration, **transition, "applied": applied,
           "notes": stage.notes},
      )
      print(
          f"[rlmcp] curriculum stage -> '{stage.name}' at iteration "
          f"{self.iteration}: {stage.notes}",
          flush=True,
      )

  # Shutdown.

  def close(self) -> None:
    """Flush final state; safe to call more than once."""
    if self.records is not None:
      self.records.finish(self.stop_reason or "closed")
    # Extensions release what they hold; a raise is logged, never propagated.
    self.extensions.close()
    try:
      self._publish_status()
      self.session.append_event(
          "session_end",
          {"iteration": self.iteration, "stop_reason": self.stop_reason,
           "total_env_steps": self.total_env_steps},
      )
    except Exception:
      pass

