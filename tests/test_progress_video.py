"""Clips the run takes of itself, on a schedule nobody has to remember."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import FakeSimAdapter

from rlmcp.adapters.base import NotSupported
from rlmcp.core.controller import RlMcp
from rlmcp.core.progress_video import Cadence, ProgressVideoSchedule
from rlmcp.records.filestore import FileStore
from rlmcp.records.link import RecordLink
from rlmcp.session import Session


def _lab(tmp_path, fake_sim, records=None, **kwargs) -> RlMcp:
  return RlMcp(
      sim_adapter=fake_sim,
      session_dir=tmp_path / "session",
      records=records,
      **kwargs,
  )


def _collect(lab: RlMcp, iteration: int, steps: int = 400) -> None:
  """Service once, then feed the clip its frames and let it encode."""
  lab.service(iteration=iteration)
  for _ in range(steps):
    lab.on_step()
  lab.service(iteration=iteration + 1)


def _events(lab: RlMcp, kind: str):
  return [e for e in Session.open(lab.session.dir).events() if e["kind"] == kind]


# The cadence.


def _sequence(cadence, stop_after: int = 12):
  """The iterations this cadence would take clips at, starting from zero."""
  out, iteration = [0], 0
  for _ in range(stop_after):
    iteration = cadence.next_after(iteration)
    out.append(iteration)
  return out


def test_the_default_cadence_doubles_from_fifty():
  """0, 50, 100, 200, 400 ... dense while the run is changing fastest, and
  thinning out on its own so a long run cannot fill a disk with clips."""
  assert _sequence(Cadence.parse(None), 7) == [0, 50, 100, 200, 400, 800, 1600, 3200]


def test_doubling_stops_growing_so_a_long_run_still_shows_what_it_is_doing():
  """Without a cap the clip after 51 200 would be at 102 400 -- days later."""
  sequence = _sequence(Cadence.parse("double"), 11)

  assert sequence[-4:] == [5200, 7200, 9200, 11200]   # even, 2000 apart
  assert Cadence.parse("double").every_at(9999) == 2000


def test_a_plain_number_is_a_flat_cadence():
  for spec in (200, "200"):
    assert _sequence(Cadence.parse(spec), 3) == [0, 200, 400, 600]
  assert Cadence.parse(200).prose() == "every 200 iterations"


def test_the_first_gap_and_the_cap_are_both_movable():
  assert _sequence(Cadence.parse("double:100:5000"), 8) == [
      0, 100, 200, 400, 800, 1600, 3200, 6400, 11400]


def test_zero_and_its_synonyms_mean_no_clips():
  for spec in (0, "0", "off", "none", ""):
    assert Cadence.parse(spec) is None


@pytest.mark.parametrize("spec", ["every 200", "banana", "-5"])
def test_a_cadence_that_does_not_parse_says_so_rather_than_guessing(spec):
  """A mistyped schedule must never quietly become a different one."""
  with pytest.raises(ValueError, match="cadence"):
    Cadence.parse(spec)


def test_the_schedule_takes_the_cadence_at_construction(tmp_path, fake_sim):
  lab = _lab(tmp_path, fake_sim, video_every="double:100:5000")
  assert lab.progress_video.describe()["cadence"] == "double:100:5000"


# Taking the clips.


def test_a_run_films_itself_whether_or_not_a_runner_ever_attaches(tmp_path, fake_sim,
                                                                  fake_runner):
  """Both wiring orders must work. `RlMcp(runner_adapter=...)` is a documented
  way to build one, and a cadence that only existed after `attach_runner` left
  that whole path silently filming nothing."""
  lab = RlMcp(sim_adapter=fake_sim, runner_adapter=fake_runner,
              session_dir=tmp_path / "constructed", video_seconds=0.2)
  try:
    _collect(lab, 0)
    assert len(_events(lab, "progress_clip")) == 1
  finally:
    lab.close()


def test_clips_repeat_on_the_cadence_and_not_between_it(tmp_path, fake_sim):
  lab = _lab(tmp_path, fake_sim, video_every=200, video_seconds=0.2)

  _collect(lab, 0)
  for iteration in (50, 100, 199):
    _collect(lab, iteration)
  assert len(_events(lab, "progress_clip")) == 1

  _collect(lab, 200)
  taken = [e["iteration"] for e in _events(lab, "progress_clip")]
  assert taken == [0, 200]


def test_a_run_on_the_default_cadence_films_itself_at_doubling_gaps(tmp_path, fake_sim):
  """The whole point, end to end: nobody asked for any of these. Iteration 0 is
  the untrained baseline every later clip is read against."""
  lab = _lab(tmp_path, fake_sim, video_seconds=0.2)

  for iteration in range(0, 401, 25):
    _collect(lab, iteration)

  clips = _events(lab, "progress_clip")
  assert [e["iteration"] for e in clips] == [0, 50, 100, 200, 400]
  first = Path(clips[0]["video_path"])
  assert first.is_file() and first.name == "progress_env0_it000000.mp4"
  assert lab.progress_video.describe()["clips"] == 5


def test_the_clip_is_named_for_the_iteration_it_is_of(tmp_path, fake_sim):
  """Collection spans iterations; the file belongs to the one it started at."""
  lab = _lab(tmp_path, fake_sim, video_every=200, video_seconds=0.2)
  lab.service(iteration=200)
  for _ in range(400):
    lab.on_step()
  lab.service(iteration=207)  # The run moved on while the clip collected.

  clip = _events(lab, "progress_clip")[-1]
  assert clip["iteration"] == 200
  assert Path(clip["video_path"]).name == "progress_env0_it000200.mp4"


def test_every_zero_means_no_automatic_clips(tmp_path, fake_sim):
  lab = _lab(tmp_path, fake_sim, video_every=0)
  _collect(lab, 0)
  assert _events(lab, "progress_clip") == []
  assert lab.progress_video.describe()["enabled"] is False


# Filing them in the record.


def test_a_clip_is_attached_to_the_run_record(tmp_path, fake_sim):
  store = FileStore(tmp_path / "records", slots=1)
  record = store.new_record("walk", hypothesis="it walks")
  link = RecordLink(store, record_id=record.id)
  link.start(str(tmp_path / "session"))

  lab = _lab(tmp_path, fake_sim, records=link, video_every=200, video_seconds=0.2)
  _collect(lab, 0)

  filed = store.get_record(record.id).assets["videos"]
  assert len(filed) == 1
  key, caption = filed[0]
  assert caption == "iteration 0"
  assert store.media.exists(key)
  # And the record's copy is its own file, so cleaning the logs cannot empty it.
  assert Path(store.media.get(key)).read_bytes()


def test_attaching_the_same_clip_twice_lists_it_once(tmp_path):
  store = FileStore(tmp_path / "records", slots=1)
  record = store.new_record("walk")
  link = RecordLink(store, record_id=record.id)
  clip = tmp_path / "progress_env0_it000000.mp4"
  clip.write_bytes(b"not really an mp4")

  first = link.attach_asset(str(clip), caption="iteration 0")
  second = link.attach_asset(str(clip), caption="iteration 0")

  assert first == second
  assert len(store.get_record(record.id).assets["videos"]) == 1


def test_a_records_failure_does_not_fail_the_clip(tmp_path, fake_sim, capsys):
  store = FileStore(tmp_path / "records", slots=1)
  record = store.new_record("walk")
  link = RecordLink(store, record_id=record.id)
  link.start(str(tmp_path / "session"))

  def explode(*args, **kwargs):
    raise OSError("media directory is read-only")

  store.media.put = explode
  lab = _lab(tmp_path, fake_sim, records=link, video_every=200, video_seconds=0.2)
  _collect(lab, 0)

  clip = _events(lab, "progress_clip")[-1]
  assert clip["asset_key"] == ""
  assert Path(clip["video_path"]).is_file()  # The clip itself is fine.
  assert "could not attach" in capsys.readouterr().out


# When a clip cannot be taken.


class _Blind(FakeSimAdapter):
  """A backend with no renderer -- the failure the schedule has to survive."""

  def render(self, env_id: int = 0):
    raise NotSupported("this environment cannot render")


def test_a_run_that_cannot_render_keeps_training_and_says_why(tmp_path):
  """The clips are a nicety; a backend that cannot render must cost events,
  never the run. Doubling gaps are what keep those events to a handful."""
  lab = _lab(tmp_path, _Blind(), video_every=50, video_seconds=0.2)

  for iteration in (0, 50, 100):
    _collect(lab, iteration, steps=20)

  skipped = _events(lab, "progress_clip_skipped")
  assert len(skipped) == 3
  assert "cannot render" in skipped[-1]["reason"]
  assert lab.progress_video.describe()["skipped"] == 3


def test_a_clip_that_cannot_start_is_a_note_not_a_crash(tmp_path, fake_sim):
  """The job cap is a legitimate refusal; training must not notice it."""
  from rlmcp.core.controller import MAX_CONCURRENT_JOBS

  lab = _lab(tmp_path, fake_sim, video_every=200, video_seconds=0.2)
  client = Session.open(lab.session.dir)
  for _ in range(MAX_CONCURRENT_JOBS):
    client.submit("record_video", seconds=0.2)

  lab.service(iteration=0)  # Fires the scheduled clip into a full queue.

  skipped = _events(lab, "progress_clip_skipped")
  assert len(skipped) == 1
  assert "Deferred-job limit reached" in skipped[0]["reason"]
  # The schedule moved on rather than retrying at every iteration.
  assert lab.progress_video.describe()["next_iteration"] == 200


# Changing the schedule while the run is going.


def test_a_tighter_interval_takes_effect_immediately(tmp_path, fake_sim):
  lab = _lab(tmp_path, fake_sim, video_every=1000, video_seconds=0.2)
  _collect(lab, 0)
  assert lab.progress_video.describe()["next_iteration"] == 1000

  reply = lab.run_command("progress_video", every=50)

  assert reply["every"] == 50 and reply["enabled"] is True
  _collect(lab, 300)
  assert [e["iteration"] for e in _events(lab, "progress_clip")] == [0, 300]


def test_the_schedule_can_be_turned_off_and_back_on(tmp_path, fake_sim):
  lab = _lab(tmp_path, fake_sim, video_every=100, video_seconds=0.2)

  assert lab.run_command("progress_video", every=0)["enabled"] is False
  _collect(lab, 0)
  assert _events(lab, "progress_clip") == []

  assert lab.run_command("progress_video", every="double")["enabled"] is True
  _collect(lab, 500)
  assert len(_events(lab, "progress_clip")) == 1


def test_the_status_payload_says_what_is_scheduled(tmp_path, fake_sim):
  lab = _lab(tmp_path, fake_sim, video_every=200, video_seconds=0.2)
  _collect(lab, 0)

  schedule = lab.run_command("status")["progress_video"]
  assert schedule["every"] == 200
  assert schedule["clips"] == 1
  assert schedule["last_iteration"] == 0
  assert schedule["next_iteration"] == 200
  assert schedule["last_path"].endswith("progress_env0_it000000.mp4")


def test_a_schedule_describes_itself_before_a_run_has_started():
  schedule = ProgressVideoSchedule()

  described = schedule.describe()
  assert described["enabled"] is True
  assert described["cadence"] == "double:50:2000"
  assert described["next_iteration"] == 0        # the untrained baseline
  assert "twice the last" in described["cadence_prose"]
  assert schedule.due(0) is True


# What the clips cost.


def test_clips_stop_when_they_have_spent_their_disk_budget():
  """A run that quietly filled its log directory overnight is a worse outcome
  than a trajectory that stops at iteration 200 and says so."""
  schedule = ProgressVideoSchedule(every=200, budget_mb=1.0)

  assert schedule.completed(0, "a.mp4", size_bytes=400_000) == ""
  note = schedule.completed(200, "b.mp4", size_bytes=700_000)

  assert "1 MB budget" in note and "iteration 200" in note
  assert schedule.active is False and schedule.due(400) is False
  described = schedule.describe()
  assert described["megabytes"] == 1.1 and described["enabled"] is False
  assert "budget" in described["stopped_because"]

  # Zero means no limit at all -- the same clip against no budget is fine.
  unlimited = ProgressVideoSchedule(every=200, budget_mb=0)
  assert unlimited.completed(0, "a.mp4", size_bytes=50_000_000) == ""
  assert unlimited.active is True
  assert unlimited.describe()["budget_mb"] is None


def test_raising_the_budget_starts_the_clips_again(tmp_path, fake_sim):
  lab = _lab(tmp_path, fake_sim, video_every=200, video_seconds=0.2,
             video_budget_mb=0.000_001)
  _collect(lab, 0)
  assert lab.progress_video.active is False
  assert _events(lab, "progress_clips_stopped")

  lab.run_command("progress_video", budget_mb=50)

  _collect(lab, 400)
  assert [e["iteration"] for e in _events(lab, "progress_clip")] == [0, 400]
