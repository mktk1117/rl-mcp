"""A still frame to stand in for a clip.

The poster is a nicety, and every one of these is really the same assertion in
a different disguise: nothing about deriving one may be able to fail a render.
"""

from __future__ import annotations

from pathlib import Path

from rlmcp.records.poster import ensure_posters, poster_key, record_posters
from rlmcp.records.record import RunRecord


def _r(rid: str, videos=None) -> RunRecord:
  record = RunRecord(id=rid, slug=f"run_{rid}", seq=int(rid[-1]))
  if videos:
    record.assets = {"videos": videos}
  return record


def test_the_key_is_a_pure_function_of_the_record_and_the_clip():
  """That is what makes the cache work with nothing written down: the renderer
  asks for the key it would have used, and a file already there is a poster
  already made."""
  key = poster_key("001", "archive/clips/tour.mp4")

  assert key == poster_key("001", "archive/clips/tour.mp4")
  assert key.startswith("001/posters/") and key.endswith("-tour.png")


def test_two_clips_sharing_a_basename_do_not_share_a_still():
  """A thumbnail silently standing for the wrong run is worse than none."""
  assert (poster_key("001", "a/tour.mp4") != poster_key("001", "b/tour.mp4"))


def test_a_hostile_record_id_cannot_climb_out_of_the_media_root():
  key = poster_key("../../etc", "../../../secrets.mp4")

  assert ".." not in key
  assert key.startswith("etc/posters/")


def test_an_existing_still_is_reused_without_decoding_anything(tmp_path):
  """The second render of a records store must be a stat, not a decode."""
  record = _r("001", [["001/videos/tour.mp4", "tour"]])
  cached = tmp_path / poster_key("001", "001/videos/tour.mp4")
  cached.parent.mkdir(parents=True)
  cached.write_bytes(b"\x89PNG\r\n")

  # derive=False proves nothing was decoded: there is no video on disk at all.
  assert ensure_posters([record], tmp_path, derive=False) == {
      "001/videos/tour.mp4": poster_key("001", "001/videos/tour.mp4")}


def test_a_still_written_into_the_record_is_believed_rather_than_re_derived(tmp_path):
  record = _r("001", [["001/videos/tour.mp4", "tour", "001/posters/written.png"]])

  assert ensure_posters([record], tmp_path) == {
      "001/videos/tour.mp4": "001/posters/written.png"}


def test_a_records_store_with_no_video_asks_for_nothing(tmp_path):
  """The common case: most runs carry no clip, and the whole viewer has a
  no-poster path because of it."""
  plain = RunRecord(id="001", slug="run_001", seq=1)
  plain.assets = {"plots": [["001/plots/curves.png", "curves"]]}

  assert ensure_posters([plain, _r("002")], tmp_path) == {}


def test_a_clip_that_is_not_there_yields_no_still(tmp_path):
  """A media key pointing at a file that has been cleaned up is not an error;
  it is an ancestry older than its artifacts."""
  record = _r("001", [["001/videos/gone.mp4", "gone"]])

  assert ensure_posters([record], tmp_path) == {}
  assert not list(tmp_path.rglob("*.png"))


def test_an_unreadable_clip_costs_the_thumbnail_and_nothing_else(tmp_path):
  """A codec this machine cannot open must not be able to fail a render."""
  video = tmp_path / "001" / "videos" / "clip.mp4"
  video.parent.mkdir(parents=True)
  video.write_bytes(b"not actually an mp4")
  record = _r("001", [["001/videos/clip.mp4", "clip"]])

  assert ensure_posters([record], tmp_path) == {}


def test_a_key_that_climbs_out_of_the_media_root_is_refused(tmp_path):
  """Asset keys are attacker-reachable through a hand-edited meta.json, and
  this is the one place a record-supplied string is turned into a path that is
  then read from and written beside."""
  outside = tmp_path.parent / "outside.mp4"
  outside.write_bytes(b"\x00")
  record = _r("001", [["../outside.mp4", "escape"]])

  assert ensure_posters([record], tmp_path) == {}


def test_a_symlinked_media_directory_is_a_normal_way_to_run_this(tmp_path):
  """A media root whose subdirectories point at wherever the artifacts actually
  live must not be mistaken for an escape attempt -- which is what resolving
  the path before checking it would do."""
  elsewhere = tmp_path / "elsewhere"
  (elsewhere / "clips").mkdir(parents=True)
  (elsewhere / "clips" / "tour.mp4").write_bytes(b"not actually an mp4")
  media = tmp_path / "media"
  media.mkdir()
  (media / "archive").symlink_to(elsewhere)
  record = _r("001", [["archive/clips/tour.mp4", "tour"]])
  cached = media / poster_key("001", "archive/clips/tour.mp4")
  cached.parent.mkdir(parents=True)
  cached.write_bytes(b"\x89PNG\r\n")

  assert ensure_posters([record], media, derive=False) == {
      "archive/clips/tour.mp4": poster_key("001", "archive/clips/tour.mp4")}


def test_only_the_headline_clip_is_decoded(tmp_path):
  """One still per record rather than one per video: filling a cache nothing
  reads would make rendering a large records store slow for no visible gain."""
  record = _r("001", [["001/videos/early.mp4", "early"],
                      ["001/videos/late.mp4", "late"]])
  for name in ("early", "late"):
    p = tmp_path / poster_key("001", f"001/videos/{name}.mp4")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x89PNG\r\n")

  assert list(ensure_posters([record], tmp_path, derive=False)) == \
      ["001/videos/late.mp4"]


def test_a_real_clip_becomes_a_real_still(tmp_path):
  """The one end-to-end pin, on a video this test writes itself so it needs no
  fixture: a frame comes out, and the second call does not make it again."""
  import imageio.v2 as imageio
  import numpy as np
  import pytest

  video = tmp_path / "001" / "videos" / "clip.mp4"
  video.parent.mkdir(parents=True)
  frames = [np.full((64, 96, 3), i * 20, dtype=np.uint8) for i in range(1, 6)]
  try:
    imageio.mimwrite(str(video), frames, fps=5)
  except Exception:  # pragma: no cover -- no encoder on this machine.
    pytest.skip("no encoder available to write a test clip")
  record = _r("001", [["001/videos/clip.mp4", "clip"]])

  first = ensure_posters([record], tmp_path)
  key = poster_key("001", "001/videos/clip.mp4")

  assert first == {"001/videos/clip.mp4": key}
  still = Path(tmp_path / key)
  assert still.read_bytes()[:4] == b"\x89PNG"
  stamp = still.stat().st_mtime_ns
  assert ensure_posters([record], tmp_path) == first
  assert still.stat().st_mtime_ns == stamp  # cached, not re-derived


def test_a_run_that_filmed_itself_gets_a_still_per_clip(tmp_path):
  """`ensure_posters` asks what each run looks like; this asks what one run
  looked like over time, which is what a strip of progress clips is for."""
  record = _r("001", [["001/videos/progress_env0_it000000.mp4", "iteration 0"],
                      ["001/videos/progress_env0_it000200.mp4", "iteration 200"]])
  for key in (entry[0] for entry in record.assets["videos"]):
    cached = tmp_path / poster_key("001", key)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(b"\x89PNG\r\n")

  stills = record_posters(record, tmp_path, derive=False)

  assert sorted(stills) == [entry[0] for entry in record.assets["videos"]]
  # And the headline-only view still returns exactly one, unchanged.
  assert len(ensure_posters([record], tmp_path, derive=False)) == 1


def test_a_run_with_no_clips_has_no_stills(tmp_path):
  assert record_posters(_r("002"), tmp_path) == {}
