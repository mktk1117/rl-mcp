"""The realtime replay's recorder: what gets into a window, and what does not.

The player half is mjlab's own viewer loop and is not re-tested here. This is
the half rlmcp writes and the half that runs *on the training thread*, where
the promises are: a window is one contiguous stretch of the run, nothing is
recorded between windows, and no step costs a device synchronisation.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import pytest

torch = pytest.importorskip("torch")

from rlmcp.adapters.mjlab.state import live_view as mod  # noqa: E402


class FakePlayer:
  """Stands in for mjlab's viewer loop: it only has to be constructible."""

  def __init__(self, env, server, scene, source):
    self.source = source
    self.window = None
    self.starved = 0
    self.error = ""
    self.stopped = False

  def run(self, catch_sigint: bool = True) -> None:
    """The real one loops until stopped; nothing here needs it to."""

  def stop(self) -> None:
    self.stopped = True

  def get_status(self) -> Any:
    return SimpleNamespace(speed_label="1x", paused=self.paused)

  paused = False

  def request_paused(self, paused: bool) -> None:
    """The real one queues this and applies it a tick later, on its thread."""
    self.paused = bool(paused)


class FakeScene:
  env_idx = 0


class Clock:
  """A clock the test moves by hand, so a gap is a decision, not a delay."""

  def __init__(self):
    self.now = 0.0

  def __call__(self) -> float:
    return self.now


def _env(num_envs: int = 4, nq: int = 7, nu: int = 3) -> Any:
  """The smallest thing the recorder reads: step_dt, num_envs, sim.data."""
  data = SimpleNamespace(
      qpos=torch.zeros((num_envs, nq)),
      qvel=torch.zeros((num_envs, nq - 1)),
      ctrl=torch.zeros((num_envs, nu)),
      mocap_pos=torch.zeros((num_envs, 0, 3)),
      mocap_quat=torch.zeros((num_envs, 0, 4)),
  )
  return SimpleNamespace(num_envs=num_envs, step_dt=0.02,
                         sim=SimpleNamespace(data=data))


def _view(env: Any = None, seconds: float = 0.2, clock: Any = None) -> Any:
  view = mod.MjlabReplayView(
      env=env or _env(), server=object(), scene=FakeScene(), seconds=seconds,
      player_factory=FakePlayer, clock=clock or Clock(),
  )
  # The thread is the player's; with a fake player there is nothing to run.
  return view


def _record(view: Any, steps: int, env_id: int = 0, clock: Optional[Clock] = None,
            dt: float = 0.02) -> None:
  for _ in range(steps):
    if clock is not None:
      clock.now += dt
    view.update(env_id)


# What a window is.


def test_a_window_holds_the_seconds_of_sim_time_it_was_asked_for():
  view = _view(seconds=0.2)  # 10 steps at 50 Hz.

  assert view.capacity == 10
  assert view.seconds == 0.2


def test_a_full_window_is_handed_over_and_recording_stops_there():
  """Between windows the recorder costs nothing at all, which is what makes it
  affordable to leave a realtime view attached to a long run."""
  clock = Clock()
  view = _view(seconds=0.2, clock=clock)

  _record(view, 10, clock=clock)

  assert view.describe()["windows_recorded"] == 1
  assert view.describe()["recording"] is False

  _record(view, 50, clock=clock)  # The run goes on; nothing is recorded.

  assert view.describe()["recorded"] == 0
  assert view.describe()["windows_recorded"] == 1


def test_the_player_asking_starts_the_next_window():
  clock = Clock()
  view = _view(seconds=0.2, clock=clock)
  _record(view, 10, clock=clock)

  view.request_window()
  _record(view, 4, clock=clock)

  assert view.describe()["recording"] is True
  assert view.describe()["recorded"] == 4


def test_the_window_the_player_takes_is_the_run_it_recorded():
  clock = Clock()
  env = _env()
  view = _view(env=env, seconds=0.2, clock=clock)

  for step in range(10):
    clock.now += 0.02
    env.sim.data.qpos[0, 0] = float(step)
    view.update(0)

  window = view.take_window()

  assert window is not None and window.frames == 10
  assert [window.advance()["qpos"][0] for _ in range(10)] == list(range(10))


def test_a_taken_window_is_not_handed_out_twice():
  clock = Clock()
  view = _view(seconds=0.2, clock=clock)
  _record(view, 10, clock=clock)

  assert view.take_window() is not None
  assert view.take_window() is None


# Contiguity: the property that makes 1x mean anything.


def test_a_gap_in_the_recording_starts_the_window_over():
  """A browser closed for a minute, or a paused run, leaves frames either side
  of the gap a minute apart. Played in sequence they would show a robot that
  teleports -- a lie about the motion, told at exactly the right speed."""
  clock = Clock()
  view = _view(seconds=0.2, clock=clock)

  _record(view, 5, clock=clock)
  clock.now += mod.GAP_S * 2  # Nobody was watching for a while.
  _record(view, 3, clock=clock)

  assert view.describe()["recorded"] == 3
  assert view.describe()["windows_restarted"] == 1


def test_a_normal_step_gap_is_not_a_restart():
  clock = Clock()
  view = _view(seconds=0.4, clock=clock)

  _record(view, 8, clock=clock, dt=mod.GAP_S / 2)

  assert view.describe()["recorded"] == 8
  assert view.describe()["windows_restarted"] == 0


def test_watching_another_environment_throws_the_half_window_away():
  """Half a window of one robot and half of another is neither."""
  clock = Clock()
  view = _view(seconds=0.2, clock=clock)
  _record(view, 5, clock=clock)

  _record(view, 2, env_id=3, clock=clock)

  assert view.env_id == 3
  assert view.describe()["recorded"] == 2


def test_an_environment_that_does_not_exist_is_clamped_not_crashed():
  view = _view(_env(num_envs=4))

  view.update(99)

  assert view.env_id == 3


# What it reports.


def test_describe_says_whether_it_is_recording_or_waiting():
  clock = Clock()
  view = _view(seconds=0.2, clock=clock)

  fresh = view.describe()
  assert fresh["recording"] is True and fresh["buffer_frames"] == 10

  _record(view, 10, clock=clock)

  assert view.describe()["recording"] is False


def test_closing_stops_the_player():
  view = _view()

  view.close()

  assert view.viewer.stopped is True


def test_the_players_thread_reports_rather_than_dying_quietly():
  """A player thread that raised and vanished would leave a tab frozen with
  nothing anywhere saying why."""
  class Exploding(FakePlayer):
    def run(self, catch_sigint: bool = True) -> None:
      raise RuntimeError("the player fell over")

  view = mod.MjlabReplayView(
      env=_env(), server=object(), scene=FakeScene(), seconds=0.2,
      player_factory=Exploding, clock=Clock())
  view._thread.join(timeout=2.0)

  assert "the player fell over" in view.describe()["error"]


# Pausing the replay: the recording stops with the playback.
#
# A window recorded during a pause is a stretch of the run nobody watched,
# handed to a player that is not asking for one. So "pause" here has to mean
# the training thread stops recording too -- otherwise pausing a view to look
# at a pose would leave the run doing all the same work with nothing to show
# for it.


def test_pausing_stops_the_recording():
  view = _view()
  view.update(0)
  assert view.describe()["recorded"] == 1

  view.set_paused(True)
  for _ in range(20):
    view.update(0)
  state = view.describe()
  assert state["recording"] is False
  assert state["recorded"] == 0, (
      "a paused replay kept filling a window; the run is supposed to go back "
      "to the speed it trains at unwatched")


def test_a_paused_replay_is_not_talked_back_into_recording():
  # Single-stepping to the end of the held window still asks for the next one.
  view = _view()
  view.set_paused(True)
  view.request_window()
  assert view.describe()["recording"] is False
  view.update(0)
  assert view.describe()["recorded"] == 0


def test_re_pointing_a_paused_replay_leaves_it_paused():
  view = _view()
  view.set_paused(True)
  view.set_env(2)
  assert view.paused and view.describe()["recording"] is False


def test_resuming_records_a_fresh_window():
  view = _view()
  view.update(0)
  view.set_paused(True)
  view.set_paused(False)
  assert view.describe()["recording"] is True
  assert view.describe()["recorded"] == 0, (
      "the window resumed mid-fill, so it splices the frames either side of "
      "the pause together and calls the result motion")
  view.update(0)
  assert view.describe()["recorded"] == 1


def test_the_pause_flag_flips_when_asked_not_a_tick_later():
  # The base class applies a queued toggle on its own thread, so reading it
  # back is answering the previous question -- which is what the button label
  # and the owner's per-step gate would both get wrong.
  view = _view()
  view.set_paused(True)
  assert view.paused is True
  assert view.describe()["paused"] is True
  view.set_paused(True)  # Asking twice must not toggle it back.
  assert view.paused is True
  view.set_paused(False)
  assert view.paused is False


def test_the_player_holds_still_when_nobody_is_watching():
  # The player has its own thread, so unlike a live push it does not stop just
  # because the training loop stopped feeding it.
  view = _view()
  view.set_watchers(0)
  player = SimpleNamespace(source=view, starved=0, window=None)
  assert mod.ReplayPlayer._execute_step(player) is True
  assert player.starved == 0, (
      "the player starved through windows nobody could have recorded; that "
      "counter is supposed to mean the run could not keep up")


def test_a_view_assumes_somebody_until_told_otherwise():
  # A backend that froze because nobody told it who was connected would be a
  # worse failure than one that draws for an empty room.
  assert _view().watchers == 1


def test_the_player_is_asked_for_a_state_not_a_toggle():
  # A toggle is only correct while nothing else can flip the player's own
  # flag. The moment a second control can, a toggle drifts and the view sits
  # frozen with the tab insisting it is running.
  view = _view()
  view.set_paused(True)
  view.viewer.paused = False  # Something else moved it.
  view._paused = False
  view.set_paused(True)
  assert view.viewer.paused is True, (
      "asking for paused twice unpaused it; the request has to name the state")
