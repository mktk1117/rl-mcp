"""Reading a run's clips back as a series rather than as attachments."""

from __future__ import annotations

from rlmcp.records.clips import (
    caption_for,
    from_record,
    is_progress_clip,
    iteration_of,
    order,
)
from rlmcp.records.record import RunRecord


def _record(videos) -> RunRecord:
  record = RunRecord(id="015", slug="walk")
  record.assets = {"videos": videos}
  return record


def test_the_caption_is_written_and_read_in_one_place():
  assert iteration_of(caption_for(1200)) == 1200


def test_the_filename_answers_when_the_caption_does_not():
  """A clip attached by hand carries whatever caption somebody typed."""
  assert iteration_of("the good one", "015/videos/progress_env0_it000200.mp4") == 200
  assert iteration_of("", "progress_env0_it000000.mp4") == 0


def test_a_clip_nobody_can_place_is_listed_last_not_dropped():
  clips = from_record(_record([
      ["015/videos/tour.mp4", "the final policy"],
      ["015/videos/progress_env0_it000200.mp4", caption_for(200)],
      ["015/videos/progress_env0_it000000.mp4", caption_for(0)],
  ]))

  assert [c["iteration"] for c in clips] == [0, 200, None]
  assert clips[-1]["caption"] == "the final policy"


def test_a_scheduled_clip_is_distinguishable_from_an_attached_one():
  clips = from_record(_record([
      ["015/videos/progress_env0_it000000.mp4", caption_for(0)],
      ["015/videos/highlight.mp4", "the moment it worked"],
  ]))

  assert [c["scheduled"] for c in clips] == [True, False]
  assert is_progress_clip("015/videos/progress_env0_it000400.mp4")
  assert not is_progress_clip("015/videos/clip_env0_it000400.mp4")


def test_posters_are_carried_through_when_they_are_offered():
  key = "015/videos/progress_env0_it000000.mp4"
  clips = from_record(_record([[key, caption_for(0)]]),
                      posters={key: "015/posters/abc-progress.png"})

  assert clips[0]["poster"] == "015/posters/abc-progress.png"


def test_a_record_with_no_videos_has_no_clips():
  assert from_record(RunRecord(id="016", slug="x")) == []


def test_ordering_is_stable_for_things_with_no_iteration():
  clips = order([{"key": "b"}, {"key": "a"}, {"key": "c", "iteration": 3}])
  assert [c["key"] for c in clips] == ["c", "a", "b"]
