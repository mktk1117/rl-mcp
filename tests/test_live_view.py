"""The live browser view: attaching to a run in progress without disturbing it.

The promises under test are the ones that make it safe to leave attached: it
starts and stops mid-run, it pushes at a bounded rate, it costs nothing while
no browser is open, and a view that breaks detaches itself rather than charging
the training loop for a failure every step.

Nothing here imports viser. The server is faked at the one seam that talks to
it, which is also the seam a second transport would be written against.
"""

from __future__ import annotations

import socket
from typing import Any, Dict, List, Optional

import pytest
from conftest import FakeSimAdapter

from rlmcp.core import live_view as live_view_module
from rlmcp.core.controller import RlMcp
from rlmcp.core.live_view import LiveView, find_free_port
from rlmcp.session import Session


class FakeScene:
  """What a backend returns from ``open_live_view``: update(env_id), close()."""

  def __init__(self, fail_with: Optional[Exception] = None):
    self.updates: List[int] = []
    self.closed = False
    self.fail_with = fail_with

  def update(self, env_id: int = 0) -> None:
    if self.fail_with is not None:
      raise self.fail_with
    self.updates.append(int(env_id))

  def close(self) -> None:
    self.closed = True


class FakeMarkdown:
  def __init__(self, content: str):
    self.content = content


class FakeButton:
  """A viser button as far as LiveView is concerned: a label and a handler."""

  def __init__(self, label: str):
    self.label = label
    self.handler: Any = None

  def on_click(self, handler: Any) -> Any:
    self.handler = handler
    return handler

  def click(self) -> None:
    self.handler(self)


class FakeGui:
  def __init__(self):
    self.panels: List[FakeMarkdown] = []
    self.buttons: List[FakeButton] = []

  def add_markdown(self, content: str) -> FakeMarkdown:
    panel = FakeMarkdown(content)
    self.panels.append(panel)
    return panel

  def add_button(self, label: str) -> FakeButton:
    button = FakeButton(label)
    self.buttons.append(button)
    return button


class FakeServer:
  """A viser server as far as LiveView is concerned: clients, a panel, a stop."""

  def __init__(self, host: str, port: int, label: str, watchers: int = 1):
    self.host, self.port, self.label = host, port, label
    self.watchers = watchers
    self.stopped = False
    self.gui = FakeGui()

  def get_clients(self) -> Dict[int, Any]:
    return {i: object() for i in range(self.watchers)}

  def stop(self) -> None:
    self.stopped = True


class ViewableSim(FakeSimAdapter):
  """A backend that can mirror itself into a browser.

  The minimal implementation of the live-view half of the adapter contract:
  build something bound to the server, take state through ``update``, and let
  go of it in ``close``.
  """

  def __init__(self, scene: Optional[FakeScene] = None, **kwargs):
    super().__init__(**kwargs)
    self.scene = scene or FakeScene()
    self.opened: List[Any] = []
    self.modes: List[Dict[str, Any]] = []

  def open_live_view(self, server: Any, realtime: bool = False,
                     buffer_seconds: float = 4.0) -> FakeScene:
    self.opened.append(server)
    self.modes.append({"realtime": realtime, "buffer_seconds": buffer_seconds})
    return self.scene


@pytest.fixture
def servers(monkeypatch) -> List[FakeServer]:
  """Every server LiveView opens during a test, in order."""
  made: List[FakeServer] = []

  def factory(host: str, port: int, label: str) -> FakeServer:
    server = FakeServer(host, port, label)
    made.append(server)
    return server

  monkeypatch.setattr(live_view_module, "_open_viser_server", factory)
  return made


class Ticker:
  """A clock the test moves, so the rate limiter can be asserted on.

  ``step=0`` is time standing still -- every tick happens in the same instant,
  which is what a training loop looks like to a 20 Hz schedule.
  """

  def __init__(self, step: float = 1.0):
    self.now = 0.0
    self.step = step

  def __call__(self) -> float:
    self.now += self.step
    return self.now


def _view(sim: Any, **kwargs) -> LiveView:
  """A view whose clock advances a second per step: every tick is due."""
  kwargs.setdefault("clock", Ticker(step=1.0))
  return LiveView(sim=sim, **kwargs)


def _lab(tmp_path, sim, **kwargs) -> RlMcp:
  return RlMcp(sim_adapter=sim, session_dir=tmp_path / "session",
               video_every=0, **kwargs)


def _run(lab: RlMcp, cmd: str, **args):
  client = Session.open(lab.session.dir)
  request = client.submit(cmd, **args)
  lab.service(iteration=lab.iteration)
  return client.poll(request.req_id)


# Attaching and detaching.


def test_a_view_can_be_attached_to_a_run_that_is_already_going(tmp_path, servers):
  """The whole point: no restart, no checkpoint, no pause -- the run keeps
  going and gains a window."""
  sim = ViewableSim()
  lab = _lab(tmp_path, sim)
  assert lab.live_view.running is False

  reply = _run(lab, "live_view", enabled=True)

  assert reply.ok
  assert reply.result["running"] is True
  assert reply.result["url"].startswith("http://localhost:")
  assert lab.iteration == 0 and not lab.paused
  lab.close()


def test_detaching_gives_the_port_back(tmp_path, servers):
  sim = ViewableSim()
  lab = _lab(tmp_path, sim, viser=True)
  assert lab.live_view.running

  reply = _run(lab, "live_view", enabled=False)

  assert reply.result["running"] is False
  assert reply.result["url"] == ""
  assert servers[0].stopped and sim.scene.closed
  lab.close()


def test_the_run_ending_closes_the_view(tmp_path, servers):
  """A port left bound by a finished run is the next run's mysterious failure."""
  lab = _lab(tmp_path, ViewableSim(), viser=True)

  lab.close()

  assert servers[0].stopped
  assert lab.live_view.running is False


def test_turning_on_a_view_that_is_already_on_does_not_rebuild_it(tmp_path, servers):
  """The scene is the expensive half; somebody who cannot see whether it is on
  should not pay for a rebuild by asking."""
  sim = ViewableSim()
  lab = _lab(tmp_path, sim, viser=True)

  _run(lab, "live_view", enabled=True)

  assert len(servers) == 1 and len(sim.opened) == 1
  lab.close()


def test_moving_the_view_to_another_port_rebinds_it(tmp_path, servers):
  """The port is the address somebody has open, so changing it is a restart."""
  lab = _lab(tmp_path, ViewableSim(), viser=True)
  first = lab.live_view.port

  reply = _run(lab, "live_view", port=first + 7)

  assert reply.result["running"] is True
  assert reply.result["port"] != first
  assert servers[0].stopped
  lab.close()


def test_a_backend_that_cannot_mirror_itself_says_so_and_the_run_goes_on(
    tmp_path, servers):
  """The default from the adapter contract: an absent capability is refused by
  name, never faked, and never fatal.

  Nothing is spent finding that out, either -- no port is taken and no server
  is started for a backend that could not have filled one. Asked in the other
  order, a machine without viser would answer a question about the *run* with a
  fact about the *install*: "no viser" for a backend that has no scene to show
  even where viser is present."""
  lab = _lab(tmp_path, FakeSimAdapter())

  reply = _run(lab, "live_view", enabled=True)

  assert reply.ok is False
  assert "open_live_view" in reply.error
  assert servers == []
  assert lab.live_view.running is False
  lab.service(iteration=1)  # The run is unharmed.
  lab.close()


def test_a_view_that_cannot_start_at_launch_does_not_take_the_run_with_it(
    tmp_path, servers):
  """`--viser` on a backend without one is a missing window, not a lost run."""
  lab = _lab(tmp_path, FakeSimAdapter(), viser=True)

  assert lab.live_view.running is False
  assert "open_live_view" in lab.live_view.last_error
  lab.close()


# What it costs.


def test_nothing_is_pushed_while_no_browser_is_open(tmp_path, servers):
  """The claim that makes leaving it attached cheap: with nobody watching, a
  step costs a clock comparison and no copy off the GPU."""
  sim = ViewableSim()
  view = _view(sim, enabled=True)
  servers[0].watchers = 0

  for _ in range(50):
    view.tick()

  assert sim.scene.updates == []
  assert view.frames == 0

  servers[0].watchers = 1
  view.tick()

  assert view.frames == 1


def test_the_push_rate_is_bounded_however_fast_the_run_steps(tmp_path, servers):
  """A training loop calls on_step thousands of times a second; the view must
  not turn each one into a frame."""
  sim = ViewableSim()
  view = LiveView(sim=sim, enabled=True, fps=20.0, clock=Ticker(step=0.0))

  for _ in range(500):
    view.tick()

  assert view.frames == 1


def test_an_unattached_view_costs_nothing_per_step(tmp_path):
  """The common case is no view at all, and it must not reach the adapter."""
  sim = ViewableSim()
  view = LiveView(sim=sim, enabled=False, clock=Ticker(step=1.0))

  for _ in range(1000):
    view.tick()

  assert sim.opened == [] and sim.scene.updates == []


def test_status_reports_where_the_view_is_and_what_it_costs(tmp_path, servers):
  sim = ViewableSim()
  lab = _lab(tmp_path, sim, viser=True, viser_fps=1000.0)
  for _ in range(3):
    lab.on_step()

  reported = _run(lab, "status").result["live_view"]

  assert reported["running"] is True
  assert reported["backend"] == "viser"
  assert reported["watchers"] == 1
  assert reported["frames"] >= 1
  assert reported["url"].endswith(str(reported["port"]))
  lab.close()


# Realtime replay.


def test_a_realtime_view_records_every_step_rather_than_a_frame_now(tmp_path,
                                                                    servers):
  """A window with every other step missing is not the run's motion; it is a
  different, jerkier one. So the rate limit that governs the live view is
  exactly what a buffered one must not have."""
  sim = ViewableSim()
  view = LiveView(sim=sim, enabled=True, realtime=True, fps=20.0,
                  clock=Ticker(step=0.0))

  for _ in range(50):
    view.tick()

  assert sim.modes[0]["realtime"] is True
  assert len(sim.scene.updates) == 50


def test_a_realtime_view_still_costs_nothing_with_no_browser_open(tmp_path,
                                                                  servers):
  sim = ViewableSim()
  view = LiveView(sim=sim, enabled=True, realtime=True, clock=Ticker(step=1.0))
  servers[0].watchers = 0

  for _ in range(50):
    view.tick()

  assert sim.scene.updates == []


def test_the_mode_reaches_the_backend_with_the_window_it_asked_for(tmp_path,
                                                                  servers):
  lab = _lab(tmp_path, ViewableSim(), viser=True, viser_realtime=True,
             viser_buffer_seconds=8.0)

  reported = _run(lab, "status").result["live_view"]

  assert reported["mode"] == "realtime"
  assert reported["buffer_seconds"] == 8.0
  lab.close()


def test_switching_mode_rebuilds_the_view_because_the_two_are_not_one_thing(
    tmp_path, servers):
  """A player and a buffer exist on one side of the switch and not the other,
  so the backend has to be asked again rather than reconfigured."""
  sim = ViewableSim()
  lab = _lab(tmp_path, sim, viser=True)
  assert sim.modes == [{"realtime": False, "buffer_seconds": 4.0}]

  reply = _run(lab, "live_view", realtime=True)

  assert reply.result["mode"] == "realtime"
  assert sim.modes[-1]["realtime"] is True
  assert servers[0].stopped and len(servers) == 2
  lab.close()


def test_what_the_backend_says_about_its_buffer_reaches_status(tmp_path, servers):
  """`status` should answer "is it recording or playing" without anyone
  having to look at the tab."""
  class Playing(FakeScene):
    def describe(self):
      return {"recording": False, "playing_frame": 12, "behind_seconds": 3.1}

  lab = _lab(tmp_path, ViewableSim(scene=Playing()), viser=True,
             viser_realtime=True)

  playback = _run(lab, "status").result["live_view"]["playback"]

  assert playback["playing_frame"] == 12 and playback["behind_seconds"] == 3.1
  lab.close()


def test_a_backend_with_nothing_to_say_about_playback_says_nothing(tmp_path,
                                                                   servers):
  lab = _lab(tmp_path, ViewableSim(), viser=True)

  assert "playback" not in _run(lab, "status").result["live_view"]
  lab.close()


# The panel beside the robot.


def test_the_tab_says_where_the_run_has_got_to(tmp_path, servers):
  """A moving robot with no iteration beside it is a screensaver."""
  lab = _lab(tmp_path, ViewableSim(), viser=True, viser_fps=1000.0)

  panel = servers[0].gui.panels[0]
  assert "iteration 0" in panel.content

  lab.service(iteration=1234)
  lab.on_step()

  assert "iteration 1,234" in panel.content
  lab.close()


def test_a_status_line_that_cannot_be_read_does_not_cost_the_view(tmp_path, servers):
  """The panel is cosmetic; the window is not."""
  def explode() -> str:
    raise RuntimeError("no numbers today")

  view = _view(ViewableSim(), enabled=True, status_provider=explode)
  view.tick()

  assert view.running is True and view.frames == 1
  assert "RuntimeError" in servers[0].gui.panels[0].content


# Which environment.


def test_the_view_shows_the_environment_it_was_told_to(tmp_path, servers):
  sim = ViewableSim()
  lab = _lab(tmp_path, sim, viser=True, viser_fps=1000.0)

  _run(lab, "live_view", env_id=5)
  lab.on_step()

  assert sim.scene.updates[-1] == 5
  lab.close()


def test_an_environment_can_be_picked_by_description(tmp_path, servers, fake_terrain):
  """The same `--where` vocabulary a screenshot takes, so nobody has to know
  which index the stairs are on."""
  sim = ViewableSim()
  lab = _lab(tmp_path, sim, viser=True, viser_fps=1000.0, extensions=[fake_terrain])

  reply = _run(lab, "live_view", where={"terrain": "flat"})

  assert reply.ok
  assert isinstance(reply.result["env_id"], int)
  lab.close()


# Failure.


def test_a_view_that_keeps_failing_detaches_itself_and_says_why(tmp_path, servers):
  """One broken view must not become a per-step cost on the training loop."""
  sim = ViewableSim(scene=FakeScene(fail_with=RuntimeError("socket is gone")))
  view = _view(sim, enabled=True)

  for _ in range(live_view_module.MAX_CONSECUTIVE_FAILURES):
    view.tick()

  assert view.running is False
  assert "socket is gone" in view.stopped_because
  assert servers[0].stopped


def test_one_failed_push_is_not_a_detach(tmp_path, servers):
  """A single hiccup mid-frame should cost a frame, not the window."""
  scene = FakeScene(fail_with=RuntimeError("transient"))
  view = _view(ViewableSim(scene=scene), enabled=True)

  view.tick()
  assert view.running is True

  scene.fail_with = None
  view.tick()

  assert view.frames == 1 and view.failures == 0


# The port.


def test_a_busy_port_is_skipped_rather_than_failing_the_view(tmp_path):
  """Two runs on one machine is the normal case, not the exception."""
  taken = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  taken.bind(("127.0.0.1", 0))
  busy = taken.getsockname()[1]
  try:
    assert find_free_port("127.0.0.1", busy) == busy + 1
  finally:
    taken.close()


def test_a_wall_of_busy_ports_is_refused_with_the_range_it_tried(tmp_path,
                                                                monkeypatch):
  """A view that cannot bind must say so rather than appear to have started."""
  monkeypatch.setattr(live_view_module.socket, "socket", _always_busy)

  with pytest.raises(RuntimeError, match="No free port"):
    find_free_port("127.0.0.1", 9999, tries=3)


class _always_busy:  # A stand-in for socket.socket, not a class API.
  def __init__(self, *args, **kwargs):
    pass

  def bind(self, address):
    raise OSError("address in use")

  def close(self):
    pass


# The record.


def test_attaching_and_detaching_are_written_down(tmp_path, servers):
  """`what was done to this run` should include somebody watching it."""
  lab = _lab(tmp_path, ViewableSim())

  _run(lab, "live_view", enabled=True)
  _run(lab, "live_view", enabled=False)

  kinds = [e["kind"] for e in Session.open(lab.session.dir).events()]
  assert "live_view_started" in kinds and "live_view_stopped" in kinds
  lab.close()


# Pausing: attached, and costing the run nothing.
#
# The promise is stronger than "cheaper". A paused view is the same speed as no
# view at all -- the tick returns before it reads a clock, asks who is watching
# or touches the simulator -- and it is still attached, so resuming is a click
# rather than a rebuild. That is the whole difference between this and `--off`,
# and it is what these check.


class PausableScene(FakeScene):
  """A backend with a pause and a loop of its own, like the realtime one."""

  def __init__(self):
    super().__init__()
    self.paused = False
    self.watchers: List[int] = []

  def set_paused(self, paused: bool) -> None:
    self.paused = bool(paused)

  def set_watchers(self, watchers: int) -> None:
    self.watchers.append(int(watchers))


def test_a_paused_view_is_not_fed(servers):
  sim = ViewableSim()
  view = _view(sim, enabled=True)
  view.tick()
  assert sim.scene.updates

  view.set_paused(True)
  before = len(sim.scene.updates)
  for _ in range(50):
    view.tick()
  assert len(sim.scene.updates) == before, (
      "a paused view pushed frames; the point of pausing is that the training "
      "loop stops paying for it entirely")


def test_pausing_keeps_the_port_and_the_scene(servers):
  sim = ViewableSim()
  view = _view(sim, enabled=True)
  port, url = view.port, view.url

  view.set_paused(True)
  assert view.running, "pausing detached the view; that is what --off is for"
  assert (view.port, view.url) == (port, url)
  assert not sim.scene.closed
  assert not servers[0].stopped


def test_resuming_feeds_it_again(servers):
  sim = ViewableSim()
  view = _view(sim, enabled=True)
  view.set_paused(True)
  for _ in range(10):
    view.tick()
  view.set_paused(False)
  view.tick()
  assert sim.scene.updates


def test_the_button_in_the_tab_pauses_and_resumes(servers):
  sim = ViewableSim()
  view = _view(sim, enabled=True)
  button = view._pause_button
  assert button is not None, "the live mode's tab has no pause button"
  assert button.label == "Pause view"

  button.click()
  assert view.paused and button.label == "Resume view"
  before = len(sim.scene.updates)
  view.tick()
  assert len(sim.scene.updates) == before

  button.click()
  assert not view.paused and button.label == "Pause view"
  view.tick()
  assert len(sim.scene.updates) > before


def test_a_realtime_view_gets_no_second_pause_button(servers):
  # The player already has one, and it means the same thing.
  view = _view(ViewableSim(), enabled=True, realtime=True)
  assert view._pause_button is None


def test_a_backend_that_pauses_itself_stops_the_tick(servers):
  # The realtime player's Pause is clicked in the browser, so the backend
  # learns first; the owner has to read that rather than only its own flag.
  sim = ViewableSim(scene=PausableScene())
  view = _view(sim, enabled=True)
  sim.scene.set_paused(True)
  before = len(sim.scene.updates)
  for _ in range(20):
    view.tick()
  assert len(sim.scene.updates) == before
  assert view.paused
  assert view.describe()["paused"] is True


def test_pausing_reaches_a_backend_with_a_player(servers):
  sim = ViewableSim(scene=PausableScene())
  view = _view(sim, enabled=True)
  view.set_paused(True)
  assert sim.scene.paused, (
      "`rlmcp view --pause` left the player running; its thread draws the tab "
      "and would keep showing motion the run has stopped recording")
  view.set_paused(False)
  assert not sim.scene.paused


def test_a_detach_clears_the_pause(servers):
  sim = ViewableSim()
  view = _view(sim, enabled=True)
  view.set_paused(True)
  view.configure(enabled=False)
  view.configure(enabled=True)
  assert not view.paused, (
      "a view re-attached with --on came back frozen, for a reason nobody "
      "looking at it could see")
  view.tick()
  assert sim.scene.updates


def test_a_pause_survives_a_mode_switch(servers):
  # Switching modes is a stop and a start, and the scene that comes back has
  # not been told anything.
  sim = ViewableSim(scene=PausableScene())
  view = _view(sim, enabled=True)
  view.set_paused(True)
  view.configure(realtime=True)
  assert view.paused and sim.scene.paused


def test_the_backend_is_told_how_many_are_watching(servers):
  sim = ViewableSim(scene=PausableScene())
  view = _view(sim, enabled=True)
  servers[0].watchers = 0
  view.tick()
  assert sim.scene.watchers == [0], (
      "a backend with a loop of its own was not told nobody is there, so it "
      "spends an unwatched run asking for work")


def test_pause_is_reported_and_said_plainly(servers):
  view = _view(ViewableSim(), enabled=True)
  assert view.describe()["paused"] is False
  view.set_paused(True)
  assert view.describe()["paused"] is True
  assert "paused" in view.prose()
  assert view.url in view.prose()


# ── hosting somebody else's scene ─────────────────────────────────────────
def test_a_hosted_view_serves_without_a_scene_of_its_own(servers):
  """`host_for_viewer()` is for a play session, where mjlab's viewer owns the simulation
  loop and draws a far richer panel than this class should. LiveView keeps the
  port, the url and the status panel; the scene is the viewer's."""
  sim = ViewableSim()
  view = _view(sim)

  server = view.host_for_viewer()

  assert server is servers[0]
  assert view.running, "a hosted view is running: a browser can connect to it"
  assert view.url.endswith(str(view.port))
  assert not sim.opened, "the backend's own scene is exactly what is not built"
  assert view.describe()["url"] == view.url


def test_a_hosted_view_pushes_nothing(servers):
  """Whoever owns the scene owns the pushing. Ticking a hosted view must not
  touch the simulator -- in a play session another thread is stepping it."""
  scene = FakeScene()
  sim = ViewableSim(scene=scene)
  view = _view(sim)
  view.host_for_viewer()

  for _ in range(10):
    view.tick()

  assert scene.updates == [], "a hosted view has no scene to update"


def test_hosting_twice_returns_the_one_server(servers):
  """The viewer asks once, but a session that restarts a view must not leave a
  port bound behind it."""
  view = _view(ViewableSim())

  first = view.host_for_viewer()
  second = view.host_for_viewer()

  assert first is second
  assert len(servers) == 1


def test_a_hosted_view_gives_the_port_back(servers):
  """LiveView opened the server, so LiveView closes it -- mjlab's viewer is a
  guest on it and leaves an external server alone."""
  view = _view(ViewableSim())
  view.host_for_viewer()

  view.stop("the run ended")

  assert servers[0].stopped
  assert not view.running
