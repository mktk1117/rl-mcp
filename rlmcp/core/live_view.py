"""A browser view of the run, attached while it trains and never blocking it.

Progress clips answer "what did it look like at iteration 800". They cannot
answer "what is it doing *right now*", and that is the question somebody asks
when a curve turns over: they want to look at the robot, from wherever they
are, without stopping the run and without waiting four seconds for an encode.

The live view is that. It serves the training environment's own scene over
viser -- the same 3-D view ``rlmcp play --mode viser`` opens on a checkpoint,
except this one is bolted to the live run:

* **It attaches and detaches mid-run.** A run launched with no viewer gets one
  with ``rlmcp view --on``; ``rlmcp view --off`` gives the port back. Nothing
  restarts, no checkpoint is loaded, and the policy being watched is the one
  currently being trained, exploration noise and all.
* **It does not block.** The push is a state copy into a scene the browser is
  already holding, taken between environment steps at a bounded rate. There is
  no encoder, no render context, and no GL: a headless machine over ssh serves
  this fine, and the offscreen renderer -- the expensive thing -- is never
  built.
* **It costs nothing when nobody is looking.** With no browser connected the
  tick returns after a clock comparison. The GPU-to-CPU copy happens only for
  the frames somebody is actually watching, and what it costs is measured and
  reported in ``status`` rather than left for a reader to wonder about.
* **It can be paused, and a paused view costs nothing either.** The tab keeps
  the frame it last painted and the run goes straight back to the speed it
  trains at when nobody is watching -- :meth:`LiveView.tick` returns at its
  first gate, before it reads a clock. This is IsaacGym's sync toggle: the
  window stays where it is, the simulation stops paying for it. It is not a
  detach, so the port stays bound and resuming is a click, not a rebuild.

Because none of that costs anything unwatched, the view is on by default: a
run that nobody looks at has bound a port and no more, and the alternative --
remembering a flag at launch for a question that only occurs to you an hour
in -- is the thing this was built to remove.

What it is not: a recording. Nothing here is written to the run record, because
nothing here is evidence -- the clips and traces stay the run's memory. This is
the window.

Two modes, because "watch the run" means two different things:

* **live** (the default) -- every frame is the run's *current* state, pushed as
  it happens. Nothing is stored and nothing lags, but the robot moves at the
  run's pace: training steps as fast as the GPU allows, so a policy that walks
  at 1 m/s in simulation sprints across the tab.
* **realtime** (``--realtime``) -- the run records a few seconds of itself into
  a buffer, and the tab plays that window back at 1x with the player controls
  mjlab's own viewer has: pause, single-step, and speeds from 1/32x to 8x. You
  are then watching a stretch that finished a moment ago rather than the
  current step, which is the trade: gait you can actually read, a few seconds
  late.

The buffer belongs to the backend, not to this module -- only the backend knows
what one frame of its environment is. What is here is the mode, the rate limit,
and the accounting.
"""

from __future__ import annotations

import socket
import time
from typing import Any, Callable, Dict, Optional

DEFAULT_PORT = 8740
"""First port tried. Busy ones are skipped -- see :data:`PORT_SEARCH`."""

DEFAULT_HOST = "0.0.0.0"
"""Bound on every interface: the machine training is usually not the machine
looking. An ssh tunnel to localhost works against this binding too."""

DEFAULT_FPS = 20.0
"""Pushes per second while somebody is watching. Smooth enough to read a gait,
low enough that the copy it costs stays in the noise of a training step."""

MAX_FPS = 60.0
PORT_SEARCH = 20
"""How many consecutive ports to try before giving up -- enough for a box
running several runs at once, few enough to fail quickly if something is wrong."""

DEFAULT_BUFFER_SECONDS = 4.0
"""Sim time one realtime window holds. Long enough to read a gait, short enough
that what you are watching is still roughly what the policy is doing now."""

WATCH_EVERY_S = 0.5
"""How often the view re-asks whether anybody is connected.

Its own cadence because the two modes push at wildly different rates -- live at
``fps``, realtime at every environment step -- and neither should turn "is the
tab still open" into a per-step question.
"""

STATUS_EVERY_S = 1.0
"""How often the run's own numbers are refreshed in the tab.

Once a second, not once a frame: a robot moving at 20 Hz beside an iteration
counter that changes every few seconds does not need them on the same clock,
and the metrics read costs more than the pose copy does.
"""

MAX_CONSECUTIVE_FAILURES = 3
"""Failed pushes in a row before the view detaches itself.

A view that cannot push is broken for good reasons (the browser process died,
a scene handle went stale), and retrying it every step would turn one fault
into a per-step cost on the training loop. It stops, says why, and the port is
free again.
"""


def find_free_port(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                   tries: int = PORT_SEARCH) -> int:
  """The first bindable port at or after ``port``.

  Two training runs on one machine is the normal case, not the exception, so a
  fixed port would mean the second run's view silently fails to start -- or
  worse, appears to start and shows the first run. The port actually taken is
  reported in ``status``, which is where the URL comes from.
  """
  for candidate in range(int(port), int(port) + max(1, int(tries))):
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
      probe.bind((host, candidate))
      return candidate
    except OSError:
      continue
    finally:
      probe.close()
  raise RuntimeError(
      f"No free port in {port}..{port + tries - 1} to serve the live view on. "
      "Pass another with `rlmcp view --on --port N`."
  )


def _open_viser_server(host: str, port: int, label: str) -> Any:
  """Start a viser server, or say plainly that this install has no viser."""
  try:
    import viser
  except ImportError as exc:  # pragma: no cover - depends on the install.
    raise RuntimeError(
        f"The live view is served by viser, which this environment does not "
        f"have ({exc}). Install it (`uv pip install viser`), or use "
        "`rlmcp video` for clips, which needs nothing but a renderer."
    ) from exc
  return viser.ViserServer(host=host, port=port, label=label, verbose=False)


class LiveView:
  """The live browser view: its server, its scene, and what it is costing.

  Owns the viser server and the backend's scene object, and nothing else --
  the state pushed into it comes from the :class:`~rlmcp.adapters.base.SimAdapter`
  through :meth:`~rlmcp.adapters.base.SimAdapter.open_live_view`, so a backend
  that cannot mirror its environment simply says so and this stays off.

  :meth:`start` and :meth:`stop` are called from a service boundary (a command,
  or construction), never from mid-rollout. :meth:`tick` is the one called per
  step, and it is written to be nearly free: a comparison against a clock, and
  on most steps nothing else at all.
  """

  def __init__(
      self,
      sim: Any,
      enabled: bool = False,
      port: int = DEFAULT_PORT,
      host: str = DEFAULT_HOST,
      fps: float = DEFAULT_FPS,
      env_id: int = 0,
      realtime: bool = False,
      buffer_seconds: float = DEFAULT_BUFFER_SECONDS,
      label: str = "rlmcp",
      server_factory: Optional[Callable[[str, int, str], Any]] = None,
      clock: Callable[[], float] = time.monotonic,
      status_provider: Optional[Callable[[], str]] = None,
  ):
    self.sim = sim
    self.host = str(host or DEFAULT_HOST)
    self.requested_port = int(port)
    self.env_id = max(0, int(env_id))
    self.label = label
    self.fps = self._clamp_fps(fps)
    self.realtime = bool(realtime)
    self.buffer_seconds = float(buffer_seconds)
    self._server_factory = server_factory or _open_viser_server
    # What the run says about itself, as markdown, for the panel in the tab.
    # A callable rather than a value: a window on a live run should not be
    # shown numbers from whenever it happened to be opened. LiveView never
    # learns what a metric is -- whoever passes this formats it.
    self._status_provider = status_provider
    # The rate limiter's only source of time, injectable so a test can assert
    # on the schedule rather than on how fast the machine running it is.
    self._clock = clock

    self._server: Any = None
    self._scene: Any = None
    self.port = 0
    self.frames = 0
    self.failures = 0
    self.startup_ms = 0.0
    self.push_ms = 0.0
    self.last_error = ""
    self.stopped_because = ""
    self._last_push: Optional[float] = None  # None: no frame pushed yet.
    self._last_watch_check: Optional[float] = None
    self._last_status: Optional[float] = None
    self._status_handle: Any = None
    self._pause_button: Any = None
    self._watchers = 0
    self._paused = False
    # Whether the backend has a pause of its own, asked once at start rather
    # than per step: the answer cannot change while a scene is built.
    self._scene_can_pause = False

    if enabled:
      # A view asked for at launch starts here, before the first step, so the
      # tab is ready at iteration 0. A failure is recorded and the run goes on
      # -- nobody should lose a training run to a viewer.
      try:
        self.start()
      except Exception as exc:
        self.last_error = f"{type(exc).__name__}: {exc}"

  # State.

  @property
  def running(self) -> bool:
    return self._scene is not None

  @property
  def url(self) -> str:
    """Where to point a browser. Empty while the view is not running."""
    return f"http://localhost:{self.port}" if self.running else ""

  @property
  def host_url(self) -> str:
    """The same view named for the machine, for a run on the other end of ssh."""
    if not self.running:
      return ""
    try:
      return f"http://{socket.gethostname()}:{self.port}"
    except Exception:
      return self.url

  @property
  def paused(self) -> bool:
    """True while the view is attached but deliberately not being fed.

    Read on every step, so it is two attribute loads and no method call in the
    common case. A realtime view has a pause of its own -- the player's button
    in the tab -- and it means exactly this, which is why this reads the
    backend's answer as well as its own rather than letting a tab say it is
    paused while the run keeps recording for it.
    """
    if self._paused:
      return True
    return bool(self._scene_can_pause and getattr(self._scene, "paused", False))

  def set_paused(self, paused: bool) -> Dict[str, Any]:
    """Stop feeding the view, or start again. Returns :meth:`describe`.

    Deliberately not a detach: the port stays bound, the scene stays built in
    the browser, and the frame already painted stays on the screen. What stops
    is the work -- which is the whole request behind "let me watch it for a
    minute without paying for it for the next hour".
    """
    self._paused = bool(paused)
    # A backend with a player of its own has to hear about it: its thread is
    # what draws the tab, and left running against a buffer that nobody is
    # filling any more it would show motion the run has stopped producing.
    setter = getattr(self._scene, "set_paused", None)
    if setter is not None:
      try:
        setter(self._paused)
      except Exception:
        pass
    self._label_pause_button()
    return self.describe()

  @staticmethod
  def _clamp_fps(fps: Any) -> float:
    return float(min(max(float(fps), 1.0), MAX_FPS))

  # Lifecycle.

  def start(self) -> Dict[str, Any]:
    """Open the server and build the scene. Returns :meth:`describe`.

    Restarting an already-running view is not an error and not a rebuild: the
    scene the browser holds is expensive to construct, and "turn it on" from
    somebody who cannot see whether it is on should not cost a rebuild.
    """
    if self.running:
      return self.describe()

    started = time.perf_counter()
    self._refuse_a_backend_with_no_view()
    self.port = find_free_port(self.host, self.requested_port)
    server = self._server_factory(self.host, self.port, self.label)
    # Before the scene, so the run's own numbers read at the top of the panel
    # rather than below every camera control the backend adds.
    self._open_status_panel(server)
    # Beside the numbers it belongs with, and above everything the backend
    # adds: "stop paying for this" is about the view as a whole, not about
    # any one of the scene controls it would otherwise sit underneath.
    self._add_pause_button(server)
    try:
      # The backend decides what a live view of its environment is; this class
      # never touches a simulator. NotSupported from here is the honest answer
      # for a backend with no scene to mirror, and it reaches the caller as the
      # command's error.
      self._scene = self.sim.open_live_view(
          server, realtime=self.realtime, buffer_seconds=self.buffer_seconds)
    except Exception:
      self._close_server(server)
      self._status_handle = None
      self._pause_button = None
      self.port = 0
      raise
    self._server = server
    self._scene_can_pause = hasattr(self._scene, "paused")
    if self._paused:
      # Held across a rebuild (a mode switch is a stop and a start), and the
      # scene that came back has not been told yet.
      self.set_paused(True)
    self.startup_ms = round((time.perf_counter() - started) * 1000.0, 1)
    self.failures = 0
    self.frames = 0
    self.last_error = ""
    self.stopped_because = ""
    self._last_push = None
    self._last_status = None
    self._last_watch_check = None
    return self.describe()

  def _refuse_a_backend_with_no_view(self) -> None:
    """Answer "this backend has no live view" before spending anything on one.

    Order matters more than it looks. Opening the server first means an install
    without viser reports a missing package for a backend that could not have
    served a view even with it -- a true statement about the machine standing
    in for the true statement about the run. The base adapter's
    ``open_live_view`` *is* the "no live view here" answer, so an adapter that
    has not overridden it is one there is no point binding a port for, and
    calling that default is how its wording stays written in one place.
    """
    from rlmcp.adapters.base import SimAdapter

    declared = getattr(type(self.sim), "open_live_view", None)
    if declared is SimAdapter.open_live_view:
      declared(self.sim, server=None)  # Raises NotSupported, in the contract's words.

  def stop(self, reason: str = "") -> Dict[str, Any]:
    """Close the scene and give the port back. Safe to call when not running."""
    scene, server = self._scene, self._server
    self._scene, self._server, self._status_handle = None, None, None
    self._pause_button = None
    self._scene_can_pause = False
    if scene is not None:
      try:
        scene.close()
      except Exception:
        pass
    self._close_server(server)
    self.port = 0
    self._watchers = 0
    self.stopped_because = str(reason)
    return self.describe()

  @staticmethod
  def _close_server(server: Any) -> None:
    if server is None:
      return
    try:
      server.stop()
    except Exception:
      pass

  def configure(
      self,
      enabled: Optional[bool] = None,
      port: Optional[int] = None,
      host: Optional[str] = None,
      fps: Optional[float] = None,
      env_id: Optional[int] = None,
      realtime: Optional[bool] = None,
      buffer_seconds: Optional[float] = None,
      paused: Optional[bool] = None,
  ) -> Dict[str, Any]:
    """Read or change the view, starting or stopping it as asked.

    Changing the port, the host or the mode of a *running* view restarts it:
    the first two are the address somebody has open and the third decides what
    was built. Changing the rate or the environment does not, because the scene
    in the browser is still the right scene.
    """
    rebind = False
    if host is not None and str(host) != self.host:
      self.host, rebind = str(host), True
    if port is not None and int(port) != self.requested_port:
      self.requested_port, rebind = int(port), True
    # The mode decides what the backend built, so switching it is a rebuild
    # rather than a setting -- there is a player and a buffer on one side of it
    # and neither on the other.
    if realtime is not None and bool(realtime) != self.realtime:
      self.realtime, rebind = bool(realtime), True
    if buffer_seconds is not None and float(buffer_seconds) != self.buffer_seconds:
      self.buffer_seconds = float(buffer_seconds)
      rebind = rebind or self.realtime
    if fps is not None:
      self.fps = self._clamp_fps(fps)
    if env_id is not None:
      self.env_id = max(0, int(env_id))

    if enabled is False:
      # A detach ends the pause with the view: coming back with `--on` to a
      # window that is frozen for a reason nobody can see would be a puzzle.
      result = self.stop("asked to stop")
      self._paused = False
      return result
    if rebind and self.running:
      self.stop("rebinding")
      self.start()
    elif enabled:
      self.start()
    if paused is not None:
      return self.set_paused(paused)
    return self.describe()

  # The per-step hook.

  def tick(self) -> None:
    """Feed the view. Called once per environment step.

    Four gates, cheapest first: not running, paused, not yet due, nobody
    watching. Only past all four does anything touch the simulator, which is
    what keeps an attached-but-unwatched view off the training loop's bill --
    and what makes "paused" and "nobody is looking" the same speed as no view
    at all rather than merely a cheaper one.

    "Due" is where the two modes part. A live view pushes on a clock, at
    ``fps``, because a frame is only worth sending as often as somebody can see
    it. A realtime view is offered *every* step -- the backend is recording a
    contiguous stretch of the run to play back, and a stretch with every other
    step missing is not the run's motion, it is a different, jerkier one.
    """
    if self._scene is None or self.paused:
      return
    now = self._clock()
    if not self.realtime:
      if self._last_push is not None and now - self._last_push < 1.0 / self.fps:
        return
      self._last_push = now

    if (self._last_watch_check is None
        or now - self._last_watch_check >= WATCH_EVERY_S):
      self._last_watch_check = now
      self._watchers = self._count_watchers()
      self._tell_scene_watchers(self._watchers)
    if not self._watchers:
      return

    started = time.perf_counter()
    try:
      self._scene.update(self.env_id)
    except Exception as exc:
      self._push_failed(exc)
      return
    self.push_ms = round((time.perf_counter() - started) * 1000.0, 2)
    self.frames += 1
    self.failures = 0
    if self.realtime:
      self._last_push = now  # Only for the status cadence; realtime never waits.
    self._refresh_status(now)

  def _tell_scene_watchers(self, watchers: int) -> None:
    """Let a backend with a loop of its own know whether to bother.

    A live push is driven from this tick and simply does not happen with
    nobody connected. A realtime player is not driven from here -- it has its
    own thread -- and left uninformed it would spend an unwatched run asking
    for windows that the gate below is never going to let anybody record.
    """
    tell = getattr(self._scene, "set_watchers", None)
    if tell is None:
      return
    try:
      tell(int(watchers))
    except Exception:
      pass

  def _add_pause_button(self, server: Any) -> None:
    """Put a pause in the tab, for a mode with no player to put one in.

    A realtime view already has one: the player's Pause, which stops the same
    work this would. Two buttons that both stop the view would only raise the
    question of how they differ, so this is the live mode's copy of it and
    nothing is added beside the player's.
    """
    if self.realtime:
      return
    try:
      button = server.gui.add_button("Pause view")
    except Exception:
      # A tab without the button still has `rlmcp view --pause`, and a view
      # nobody can pause is better than no view. Nothing here names viser --
      # this class talks to a server through the same handful of calls a
      # second transport would have to answer, and a button is one of them.
      self._pause_button = None
      return

    @button.on_click
    def _(_: Any) -> None:
      self.set_paused(not self._paused)

    self._pause_button = button
    self._label_pause_button()

  def _label_pause_button(self) -> None:
    """Say which way the button goes now.

    Called from whoever changed the state rather than from a redraw, so the
    tab and `rlmcp view --pause` cannot end up disagreeing about what the
    button is currently offering to do.
    """
    button = self._pause_button
    if button is None:
      return
    try:
      button.label = "Resume view" if self._paused else "Pause view"
    except Exception:
      self._pause_button = None

  def _open_status_panel(self, server: Any) -> None:
    """Put the run's own numbers above the scene, so the tab says what it is.

    Added before the backend builds its scene, so this reads first in the
    panel. A viser without markdown is not a reason to have no view, so the
    whole thing is best-effort.
    """
    if self._status_provider is None:
      return
    try:
      self._status_handle = server.gui.add_markdown(self._status_text())
    except Exception:
      self._status_handle = None

  def _status_text(self) -> str:
    try:
      return str(self._status_provider())
    except Exception as exc:
      return f"*(the run could not be read: {type(exc).__name__})*"

  def _refresh_status(self, now: float) -> None:
    if self._status_handle is None:
      return
    if self._last_status is not None and now - self._last_status < STATUS_EVERY_S:
      return
    self._last_status = now
    try:
      self._status_handle.content = self._status_text()
    except Exception:
      # A panel that cannot be written is cosmetic; the view is not.
      self._status_handle = None

  def _count_watchers(self) -> int:
    """How many browsers are attached, asked at the transport.

    Not ``get_clients()``, which is the deliberate part: viser counts a client
    as connected only once its 3-D canvas has sent a camera message, so a tab
    that is open and holding the scene -- but has not moved its camera -- reads
    as nobody, and the view would sit frozen for somebody who is watching it.
    The websocket registry knows the moment the tab connects, which is the
    question actually being asked here. ``get_clients()`` is the fallback for a
    viser that keeps its connections somewhere else.
    """
    connections = getattr(
        getattr(self._server, "_websock_server", None), "_client_state_from_id", None)
    if connections is not None:
      try:
        return len(connections)
      except Exception:
        pass
    try:
      return len(self._server.get_clients())
    except Exception:
      # A server that cannot say who is connected is assumed to have somebody:
      # guessing "nobody" would silently freeze the view for whoever is there.
      return 1

  def _push_failed(self, exc: Exception) -> None:
    self.failures += 1
    self.last_error = f"{type(exc).__name__}: {exc}"
    if self.failures >= MAX_CONSECUTIVE_FAILURES:
      self.stop(f"{self.failures} pushes in a row failed: {self.last_error}")

  # Reporting.

  def describe(self) -> Dict[str, Any]:
    """The status-payload view: where it is, who is on it, what it costs."""
    payload = {
        "running": self.running,
        "backend": "viser",
        "mode": "realtime" if self.realtime else "live",
        "url": self.url,
        "host_url": self.host_url,
        "host": self.host,
        "port": self.port or None,
        "env_id": self.env_id,
        "fps": self.fps,
        "watchers": self._watchers,
        "paused": self.paused,
        "frames": self.frames,
        "push_ms": self.push_ms,
        "startup_ms": self.startup_ms,
        "last_error": self.last_error,
        "stopped_because": self.stopped_because,
        "at": time.time(),
    }
    if self.realtime:
      payload["buffer_seconds"] = self.buffer_seconds
    # Whatever the backend can say about its own playback -- how full the
    # buffer is, how far behind the tab is running. Optional, because a
    # backend that only pushes frames has nothing to add.
    described = self._safe_describe()
    if described:
      payload["playback"] = described
    return payload

  def _safe_describe(self) -> Dict[str, Any]:
    describe = getattr(self._scene, "describe", None)
    if describe is None:
      return {}
    try:
      return dict(describe() or {})
    except Exception:
      return {}

  def prose(self) -> str:
    """One line for a launch banner or a status header."""
    if not self.running:
      return "live view off"
    if self.paused:
      return (f"live view paused on {self.url} (attached, costing the run "
              "nothing; resume it in the tab or with `rlmcp view --resume`)")
    if self.realtime:
      return (f"live view on {self.url} (env {self.env_id}, {self.buffer_seconds:g}s "
              "windows played back at real speed)")
    return (f"live view on {self.url} (env {self.env_id}, up to {self.fps:g} "
            f"frames/s while a browser is open)")
