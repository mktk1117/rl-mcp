"""The live browser view of an mjlab environment, served over viser.

mjlab already knows how to put one of its scenes in a browser -- that is what
``rlmcp play --mode viser`` opens on a checkpoint. The pieces underneath are
reusable without the viewer that owns the simulation loop, and that distinction
is the whole feature: :class:`~mjlab.viewer.viser.MjlabViserScene` holds the
geometry and takes state, :class:`~mjlab.viewer.BaseViewer` holds the playback
clock, and ``ViserPlayViewer`` is the thing that would step the environment and
block. This module takes the first two and leaves the third, so the scene can
be fed from inside a training loop that belongs to rsl_rl.

Two ways to feed it, and they are genuinely different:

* :class:`MjlabLiveScene` -- push the current state, whenever asked. The
  cheapest thing that shows a robot, and the motion runs at the run's own pace.
* :class:`MjlabReplayView` -- record a window of the run and play it back at
  1x through mjlab's viewer, with its player controls in the tab. Costs a
  handful of bytes per step and shows a gait somebody can actually read.

Neither needs a render context: geometry crosses to the browser once and only
state follows, so this works on a headless machine and never touches the GPU
memory a training run is already using.

Why the replay does not subclass ``ViserPlayViewer``
---------------------------------------------------

It is tempting -- it is the finished viewer, with reward bars, term plots and
camera feeds. But every one of those panels reads the live environment from the
viewer's own thread, and its Commands folder *writes* to it. In a play session
that is fine: the viewer owns the environment and nothing else is running. Here
a trainer owns it, so a second thread reading manager buffers races the steps
being taken, and a slider that quietly retunes a command term would be a change
to the run with nobody's reason attached to it -- the one thing rlmcp exists to
prevent. So the replay takes mjlab's :class:`BaseViewer` (the accumulator that
makes 1x mean 1x, the speed ladder, pause and single-step) and mjlab's scene,
and builds a player that only ever reads a buffer.
"""

from __future__ import annotations

import copy
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

from rlmcp.adapters.base import NotSupported

DEFAULT_BUFFER_SECONDS = 4.0
MAX_BUFFER_SECONDS = 30.0
"""A ceiling on the window, so a mistyped `--buffer-seconds 3000` cannot ask a
training loop to record for an hour before anything appears in the tab."""

#: What one frame of a replay is. Everything else the scene draws is derived
#: from these by forward kinematics on the viewer's own copy of the model.
_CAPTURED_FIELDS = ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat")

GAP_S = 0.5
"""A pause in recording longer than this starts the window over.

A window is worth watching because its frames are *adjacent* steps of one
rollout. If the browser closed for a minute, or the run paused, the frames
either side of that gap are a minute apart, and playing them in sequence would
show a robot that teleports -- a lie about the motion, told at 1x.
"""

PLAYBACK_FPS = 30.0
"""Frames a second the player draws at. The buffer holds every control step
(usually 50 Hz); drawing every one of them would cost more than it shows."""


def open_live_view(env: Any, server: Any, realtime: bool = False,
                   buffer_seconds: float = DEFAULT_BUFFER_SECONDS) -> Any:
  """Build the scene for ``env`` on ``server`` and return a handle to feed it.

  Construction is the expensive half -- meshes are converted and sent to the
  browser once -- and it is paid here, at the service boundary where the view
  was asked for, rather than in the middle of a rollout.
  """
  scene = build_scene(env, server)
  if not realtime:
    scene.create_scene_gui()
    return MjlabLiveScene(env, server, scene)
  return MjlabReplayView(env, server, scene, seconds=buffer_seconds)


def build_scene(env: Any, server: Any) -> Any:
  """A viser scene for this environment, on its own copy of the host model.

  The copy matters. :class:`MjlabViserScene` edits the model it is handed --
  it clears MuJoCo's same-frame shortcuts and rewrites per-world fields on
  every frame -- and the model it would otherwise be handed is the one rlmcp's
  screenshot renderer draws from. Two viewers sharing one model means each one
  can change what the other sees, and a copy costs a few megabytes once.
  """
  try:
    from mjlab.viewer.viser import MjlabViserScene
  except ImportError as exc:  # pragma: no cover - depends on the install.
    raise NotSupported(
        f"mjlab's viser scene is unavailable ({exc}), so this build cannot "
        "serve a live view. `rlmcp video` still records clips."
    ) from exc

  sim = getattr(env, "sim", None)
  if sim is None or not hasattr(sim, "mj_model"):
    raise NotSupported(
        "This environment exposes no mjlab Simulation, so there is no scene "
        "to mirror into a browser."
    )

  try:
    scene = MjlabViserScene(
        server=server,
        mj_model=copy.deepcopy(sim.mj_model),
        num_envs=int(env.num_envs),
        sim_model=sim.model,
        expanded_fields=sim.expanded_fields,
    )
  except Exception as exc:
    raise NotSupported(
        f"Could not build a viser scene for this environment: {exc}"
    ) from exc

  # One environment, not all of them. A training run has thousands; drawing
  # every one of them would send the browser thousands of transforms per frame
  # to show a crowd nobody can read. The checkbox in the tab still says so.
  scene.show_only_selected = True
  return scene


class MjlabLiveScene:
  """The live handle: ``update(env_id)`` pushes the current state, now.

  This is the object :meth:`~rlmcp.adapters.base.SimAdapter.open_live_view`
  promises. It holds no state of its own beyond which environment was last
  selected, which exists so the slider in the browser keeps working -- see
  :meth:`update`.
  """

  def __init__(self, env: Any, server: Any, scene: Any):
    self.env = env
    self.server = server
    self.scene = scene
    self._selected = int(getattr(scene, "env_idx", 0))

  def update(self, env_id: int = 0) -> None:
    """Push the current state of one environment to every connected browser.

    ``env_id`` is applied only when it *changed*, and that is deliberate: the
    tab has an environment slider of its own, and writing rlmcp's setting back
    every frame would fight whoever is dragging it. So the last person to say
    which environment to watch wins, whether they said it in a browser or in
    `rlmcp view --env-id`.
    """
    wanted = max(0, min(int(env_id), int(self.env.num_envs) - 1))
    if wanted != self._selected:
      self.scene.env_idx = wanted
      self._selected = wanted
    # atomic() sends the frame as one update, so a browser never paints half a
    # robot from this frame and half from the last one.
    with self.server.atomic():
      self.scene.update(self.env.sim.data)

  def close(self) -> None:
    """Drop what the handle holds.

    The viser server -- and with it every scene node -- is closed by the caller
    that opened it, so there is nothing to remove here; letting go of the scene
    is what frees the converted meshes.
    """
    self.scene = None
    self.server = None


class _Window:
  """One contiguous stretch of the run, on the CPU, ready to play.

  Immutable except for where the player has got to. Frames are dicts of plain
  numpy rows, which is all the player needs and all that survives the training
  loop moving on.
  """

  def __init__(self, arrays: Dict[str, np.ndarray], env_id: int,
               frames: int, recorded_at: float):
    self.arrays = arrays
    self.env_id = int(env_id)
    self.frames = int(frames)
    self.recorded_at = float(recorded_at)
    self.index = 0

  @property
  def done(self) -> bool:
    return self.index >= self.frames

  @property
  def remaining(self) -> int:
    return max(0, self.frames - self.index)

  def advance(self) -> Dict[str, np.ndarray]:
    """The next frame, as a dict of rows. Repeats the last one at the end."""
    at = min(self.index, self.frames - 1)
    self.index += 1
    return {name: array[at] for name, array in self.arrays.items()}


class MjlabReplayView:
  """Record a window of the run on the training thread; play it back on another.

  The split is the point. Recording is a handful of small device-to-device
  copies per step with no synchronisation, which a training loop can afford;
  everything expensive -- forward kinematics, the transfer to the browser,
  waiting out real time -- happens on the player's thread, where being slow
  costs nobody anything.

  Recording is *not* continuous. A window is filled, handed over, and then
  nothing is recorded until the player asks for the next one, so the cost falls
  to zero for most of the time a run is being watched. What that buys is a
  window whose frames are adjacent steps of one rollout: real motion, at real
  speed, a few seconds old.
  """

  def __init__(self, env: Any, server: Any, scene: Any,
               seconds: float = DEFAULT_BUFFER_SECONDS,
               player_factory: Optional[Any] = None,
               clock: Any = time.monotonic):
    self.env = env
    self.server = server
    self.scene = scene

    self.dt = float(getattr(env, "step_dt", 0.02)) or 0.02
    seconds = min(max(float(seconds), self.dt * 2), MAX_BUFFER_SECONDS)
    self.capacity = max(2, int(round(seconds / self.dt)))
    self.seconds = round(self.capacity * self.dt, 3)

    self.env_id = int(getattr(scene, "env_idx", 0))
    self._requested_env = self.env_id
    self._buffers: Dict[str, Any] = {}
    self._fill = 0
    self._recording = True
    self._windows = 0
    self._dropped = 0
    self._gaps = 0
    self._lock = threading.Lock()
    self._pending: Optional[_Window] = None
    self._clock = clock
    self._last_capture: Optional[float] = None

    self.viewer = (player_factory or player_class())(env, server, scene, self)
    self._error = ""
    self._thread = threading.Thread(
        target=self._play, name="rlmcp-live-view", daemon=True)
    self._thread.start()

  # The training thread.

  def update(self, env_id: int = 0) -> None:
    """Record one step, if a window is being filled. Called every env step."""
    wanted = max(0, min(int(env_id), int(self.env.num_envs) - 1))
    if wanted != self._requested_env:
      # rlmcp was told to watch somewhere else; the half-filled window is of
      # the wrong robot, so it goes.
      self._requested_env = wanted
      self.set_env(wanted)
    if not self._recording:
      return

    now = self._clock()
    if (self._fill and self._last_capture is not None
        and now - self._last_capture > GAP_S):
      self._fill = 0  # Not one stretch of the run any more; start again.
      self._gaps += 1
    self._last_capture = now

    data = self.env.sim.data
    for name in _CAPTURED_FIELDS:
      row = self._read(data, name)
      if row is None:
        continue
      buffer = self._buffers.get(name)
      if buffer is None:
        buffer = self._allocate(name, row)
      buffer[self._fill].copy_(row)
    self._fill += 1
    if self._fill >= self.capacity:
      self._hand_over()

  def _read(self, data: Any, name: str) -> Any:
    """One environment's row of a field, or None if this model has no such thing."""
    array = getattr(data, name, None)
    if array is None:
      return None
    try:
      row = array[self.env_id]
    except Exception:
      return None
    return row if getattr(row, "numel", lambda: 0)() else None

  def _allocate(self, name: str, row: Any) -> Any:
    import torch

    buffer = torch.empty((self.capacity, *row.shape), dtype=row.dtype,
                         device=row.device)
    self._buffers[name] = buffer
    return buffer

  def _hand_over(self) -> None:
    """Move the finished window to the CPU and offer it to the player.

    The one synchronisation point in the whole recording path, and it happens
    once per window rather than once per step.
    """
    arrays = {name: buffer.detach().to("cpu").numpy()
              for name, buffer in self._buffers.items()}
    window = _Window(arrays, self.env_id, self._fill, time.time())
    with self._lock:
      if self._pending is not None:
        self._dropped += 1  # The player never came for the last one.
      self._pending = window
    self._windows += 1
    self._fill = 0
    self._recording = False

  # The player's thread.

  def take_window(self) -> Optional[_Window]:
    with self._lock:
      window, self._pending = self._pending, None
    return window

  def request_window(self) -> None:
    """Start recording again. Called as the player nears the end of a window."""
    self._recording = True

  def set_env(self, env_id: int) -> None:
    """Point the recording at another environment, from either end."""
    env_id = max(0, min(int(env_id), int(self.env.num_envs) - 1))
    if env_id == self.env_id:
      return
    self.env_id = env_id
    self._fill = 0
    self._recording = True
    with self._lock:
      self._pending = None
    if getattr(self.scene, "env_idx", env_id) != env_id:
      self.scene.env_idx = env_id

  def _play(self) -> None:
    try:
      self.viewer.run(catch_sigint=False)
    except Exception as exc:  # pragma: no cover - the thread must not vanish silently.
      self._error = f"{type(exc).__name__}: {exc}"

  # The rest of rlmcp.

  def describe(self) -> Dict[str, Any]:
    """What the buffer is doing, for the ``status`` payload."""
    window = self.viewer.window
    return {
        "buffer_frames": self.capacity,
        "buffer_seconds": self.seconds,
        "recording": self._recording,
        "recorded": self._fill,
        "windows_recorded": self._windows,
        "windows_dropped": self._dropped,
        "windows_restarted": self._gaps,
        "playing_frame": window.index if window else 0,
        "behind_seconds": (round(time.time() - window.recorded_at, 1)
                           if window else None),
        "speed": self.viewer.get_status().speed_label,
        "paused": self.viewer.get_status().paused,
        "starved": self.viewer.starved,
        "error": self._error or self.viewer.error,
    }

  def close(self) -> None:
    self.viewer.stop()
    self._thread.join(timeout=2.0)
    self._buffers.clear()
    self.scene = None
    self.server = None


def _no_policy(*_args: Any, **_kwargs: Any) -> Any:
  raise RuntimeError(
      "The replay viewer never runs a policy: it plays back what the training "
      "run already did."
  )


_PLAYER_CLASS: Any = None


def player_class() -> Any:
  """``ReplayPlayer`` combined with mjlab's ``BaseViewer``, built on first use.

  Composed rather than declared so that importing this module -- which the
  mjlab adapter does at import time -- does not require mjlab's viewer to be
  installed. The mixin below is the whole of what rlmcp writes; the base class
  it is mixed into is the whole of what it reuses, and the order puts the
  mixin's overrides in front.
  """
  global _PLAYER_CLASS
  if _PLAYER_CLASS is None:
    try:
      from mjlab.viewer.base import BaseViewer
    except ImportError as exc:  # pragma: no cover - depends on the install.
      raise NotSupported(f"mjlab's viewer is unavailable: {exc}") from exc
    _PLAYER_CLASS = type("ReplayViewer", (ReplayPlayer, BaseViewer), {})
  return _PLAYER_CLASS


class ReplayPlayer:
  """mjlab's viewer loop, playing a recorded window instead of a simulation.

  Everything inherited is the part worth having: the budget accumulator that
  makes "1x" mean one second of sim per second of wall clock whatever the
  machine is doing, the speed ladder from 1/32x to 8x, pause, single-step, and
  a thread-safe queue for the buttons that drive them. What is overridden is
  every place the base class would touch the environment -- because here the
  environment belongs to a training run, and the viewer is a spectator.
  """

  def __init__(self, env: Any, server: Any, scene: Any, source: Any,
               frame_rate: float = PLAYBACK_FPS):
    super().__init__(env=env, policy=_no_policy, frame_rate=frame_rate)
    self.server = server
    self.scene = scene
    self.source = source
    self.window: Optional[_Window] = None
    self.starved = 0
    self.error = ""
    self._frame: Optional[Dict[str, np.ndarray]] = None
    self._asked_for_next = False
    self._stop = threading.Event()
    self._status_handle: Any = None
    self._status_at = 0.0
    self._mj_data: Any = None

  # Lifecycle.

  def stop(self) -> None:
    self._stop.set()

  def is_running(self) -> bool:
    return not self._stop.is_set()

  def close(self) -> None:
    """Nothing to release: the server and the scene belong to the caller."""

  def setup(self) -> None:
    import mujoco

    self._mj_data = mujoco.MjData(self.scene.mj_model)
    self._build_gui()

  # The loop's environment-facing hooks, all pointed at the buffer instead.

  def sync_viewer_to_env(self) -> None:
    """Nothing goes back. A spectator does not steer the run."""

  def reset_environment(self) -> None:
    """Start a fresh window rather than restarting anybody's episodes.

    The base class would call ``env.reset()`` here, which on a training run
    would throw away the episodes it is learning from because somebody clicked
    a button in a browser. `rlmcp reset-envs` is how that is asked for, and it
    says who asked.
    """
    self.window = None
    self._frame = None
    self._asked_for_next = False
    self.source.request_window()

  def _execute_step(self) -> bool:
    """Advance the playback by one recorded step.

    Always reports success: an empty buffer is the run being slower than real
    time, not a failure, and the base class reads False as "the step raised"
    and pauses itself.
    """
    if self._pick_env_from_the_tab():
      return True
    window = self.window
    if window is None or window.done:
      window = self.source.take_window()
      if window is None:
        self.starved += 1
        self.source.request_window()
        return True
      self.window = window
      self._asked_for_next = False
      # Counted per window, not for the life of the view: "the run could not
      # keep up with the last three seconds" is a fact somebody can act on,
      # and a number that only ever grows is not.
      self.starved = 0
    self._frame = self.window.advance()
    # Ask for the next window before this one runs out, so the run has time to
    # record it while there is still something to watch.
    if not self._asked_for_next and self.window.remaining <= max(
        1, self.window.frames // 4):
      self._asked_for_next = True
      self.source.request_window()
    return True

  def _pick_env_from_the_tab(self) -> bool:
    """Follow the browser's environment slider. True if it just moved."""
    chosen = int(getattr(self.scene, "env_idx", self.source.env_id))
    if chosen == self.source.env_id:
      return False
    self.source.set_env(chosen)
    self.window = None
    self._frame = None
    return True

  def sync_env_to_viewer(self) -> None:
    """Draw the frame the playback is on. Reads the buffer, never the run."""
    self._refresh_status()
    frame = self._frame
    if frame is None:
      return
    try:
      self._draw(frame)
    except Exception as exc:
      self.error = f"{type(exc).__name__}: {exc}"

  def _draw(self, frame: Dict[str, np.ndarray]) -> None:
    import mujoco

    data = self._mj_data
    for name, row in frame.items():
      target = getattr(data, name, None)
      if target is None or target.size == 0:
        continue
      target[:] = row.reshape(target.shape)
    # Per-world visual variants (randomised sizes, meshes, materials) live on
    # the GPU model and have to be pulled into this thread's copy before the
    # kinematics run on it. update_from_mjdata does this too, but afterwards --
    # too late for the poses it is about to read.
    sync = getattr(self.scene, "_sync_model_fields", None)
    if sync is not None:
      sync(int(getattr(self.scene, "env_idx", 0)))
    mujoco.mj_forward(self.scene.mj_model, data)
    with self.server.atomic():
      self.scene.update_from_mjdata(data)

  # The tab.

  def _build_gui(self) -> None:
    import viser

    with self.server.gui.add_folder("Replay"):
      self._status_handle = self.server.gui.add_html("")
      pause = self.server.gui.add_button(
          "Pause", icon=viser.Icon.PLAYER_PAUSE)

      @pause.on_click
      def _(_: Any) -> None:
        self.request_toggle_pause()
        pause.label = "Play" if self._is_paused else "Pause"
        pause.icon = (viser.Icon.PLAYER_PLAY if self._is_paused
                      else viser.Icon.PLAYER_PAUSE)

      step = self.server.gui.add_button("Step", icon=viser.Icon.PLAYER_TRACK_NEXT)

      @step.on_click
      def _(_: Any) -> None:
        self.request_single_step()

      speed = self.server.gui.add_button_group(
          "Speed", options=["Slower", "1x", "Faster"])

      @speed.on_click
      def _(event: Any) -> None:
        if event.target.value == "Slower":
          self.request_speed_down()
        elif event.target.value == "1x":
          self.request_reset_speed()
        else:
          self.request_speed_up()

      fresh = self.server.gui.add_button(
          "Skip to a fresh window", icon=viser.Icon.REFRESH)

      @fresh.on_click
      def _(_: Any) -> None:
        self.request_reset()

    with self.server.gui.add_folder("Scene"):
      self.scene.create_scene_gui()

  def _refresh_status(self, every: float = 0.5) -> None:
    if self._status_handle is None:
      return
    now = time.perf_counter()
    if now - self._status_at < every:
      return
    self._status_at = now
    state = self.source.describe()
    window = self.window
    if window is not None and not window.done:
      where = (f"playing {window.index}/{window.frames} "
               f"({window.index * self.source.dt:.1f}s of "
               f"{self.source.seconds:g}s)")
    elif state["recording"]:
      where = f"recording {state['recorded']}/{state['buffer_frames']}"
    else:
      where = "waiting for the run"
    behind = state["behind_seconds"]
    self._status_handle.content = f"""
      <div style="font-size: 0.85em; line-height: 1.35; padding: 0 1em 0.5em 1em;">
        <strong>{where}</strong><br/>
        speed {self.get_status().speed_label}
        {'&middot; paused' if self._is_paused else ''}<br/>
        {'recorded %.0fs ago' % behind if behind is not None else '&nbsp;'}<br/>
        <span style="opacity:0.7;">env {self.source.env_id} &middot;
        {state['windows_recorded']} windows</span>
      </div>
      """


__all__ = [
    "DEFAULT_BUFFER_SECONDS",
    "MAX_BUFFER_SECONDS",
    "MjlabLiveScene",
    "MjlabReplayView",
    "ReplayPlayer",
    "build_scene",
    "player_class",
    "open_live_view",
]
