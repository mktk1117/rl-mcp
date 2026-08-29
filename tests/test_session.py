"""The trainer <-> agent file protocol."""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path

import pytest

from rlmcp.session import (
    RESERVED_METRIC_KEYS,
    WIRE_SURFACE,
    Session,
    SessionClient,
    iter_sessions,
)


def test_request_response_roundtrip(tmp_path):
  session = Session(tmp_path / "sess").create({"task": "demo"})

  request = session.submit("status", verbose=True)
  pending = session.pop_requests()

  assert [r.cmd for r in pending] == ["status"]
  assert pending[0].args == {"verbose": True}

  from rlmcp.session import Response

  session.respond(Response(req_id=request.req_id, ok=True, result={"iteration": 7}))
  answer = session.wait(request.req_id, timeout=2.0)
  assert answer.ok and answer.result == {"iteration": 7}


def test_requests_are_claimed_once(tmp_path):
  session = Session(tmp_path / "sess").create({})
  session.submit("a")
  session.submit("b")

  first = session.pop_requests()
  second = session.pop_requests()

  assert [r.cmd for r in first] == ["a", "b"]  # FIFO
  assert second == []


def test_wait_times_out_when_nobody_answers(tmp_path):
  session = Session(tmp_path / "sess").create({})
  request = session.submit("status")

  answer = session.wait(request.req_id, timeout=0.3, interval=0.05)

  assert not answer.ok
  assert "Timed out" in (answer.error or "")


def test_status_is_never_read_half_written(tmp_path):
  session = Session(tmp_path / "sess").create({})
  for i in range(50):
    session.publish_status({"iteration": i})
    assert json.loads(session.status_file.read_text())["iteration"] == i


def test_metrics_and_events_append(tmp_path):
  session = Session(tmp_path / "sess").create({})
  session.append_metrics(1, {"reward": 0.5})
  session.append_metrics(2, {"reward": 0.7})
  session.append_event("note", {"text": "hello"})

  rows = session.metrics()
  assert [r["iteration"] for r in rows] == [1, 2]
  assert rows[-1]["reward"] == 0.7
  assert session.events()[0]["kind"] == "note"


def test_torn_jsonl_line_is_skipped(tmp_path):
  session = Session(tmp_path / "sess").create({})
  session.append_metrics(1, {"reward": 0.5})
  with session.metrics_file.open("a") as f:
    f.write('{"iteration": 2, "rew')  # Simulate a kill mid-write.

  rows = session.metrics()

  assert len(rows) == 1 and rows[0]["iteration"] == 1


def test_open_rejects_a_directory_without_a_session(tmp_path):
  with pytest.raises(FileNotFoundError):
    Session.open(tmp_path)


def test_find_latest_picks_the_newest(tmp_path):
  Session(tmp_path / "old").create({"started_at": 1.0})
  Session(tmp_path / "new").create({})
  # create() stamps started_at itself; force the ordering deterministically.
  old = json.loads((tmp_path / "old" / "session.json").read_text())
  old["started_at"] = 1.0
  (tmp_path / "old" / "session.json").write_text(json.dumps(old))

  latest = Session.find_latest(tmp_path)

  assert latest is not None and latest.dir.name == "new"
  assert {s.dir.name for s in iter_sessions(tmp_path)} == {"old", "new"}


def test_is_alive_tracks_the_creating_process(tmp_path):
  session = Session(tmp_path / "sess").create({})
  assert session.is_alive()  # This process created it.

  info = json.loads(session.session_file.read_text())
  info["pid"] = 999_999_999  # A pid that cannot exist.
  session.session_file.write_text(json.dumps(info))

  assert not Session(session.dir).is_alive()


def test_prune_outbox_removes_old_responses(tmp_path):
  from rlmcp.session import Response

  session = Session(tmp_path / "sess").create({})
  session.respond(Response(req_id="abc", ok=True, result=1))
  path = session.outbox / "abc.json"
  os.utime(path, (0, 0))

  assert session.prune_outbox(keep_seconds=60) == 1
  assert not path.exists()


# Request TTL: a command nobody is waiting for anymore must never execute.


def _age_request(session, req_id, seconds):
  """Rewrite a queued request's timestamp as if it were submitted earlier."""
  path = next(p for p in session.inbox.glob("*.json") if req_id in p.name)
  payload = json.loads(path.read_text())
  payload["created_at"] = time.time() - seconds
  path.write_text(json.dumps(payload))


def test_stale_request_gets_error_response_instead_of_executing(tmp_path):
  session = Session(tmp_path / "sess").create({})
  request = session.submit("stop_training")
  _age_request(session, request.req_id, seconds=300.0)

  popped = session.pop_requests()

  assert popped == []  # Never handed to the trainer loop.
  answer = session.poll(request.req_id)
  assert answer is not None and not answer.ok
  assert "expired before execution" in (answer.error or "")
  assert "limit 120s" in (answer.error or "")  # Session.REQUEST_MAX_AGE_S default.
  expired = [e for e in session.events() if e["kind"] == "request_expired"]
  assert [e["req_id"] for e in expired] == [request.req_id]
  assert expired[0]["cmd"] == "stop_training"
  assert expired[0]["age_s"] == pytest.approx(300.0, abs=5.0)
  assert expired[0]["max_age_s"] == Session.REQUEST_MAX_AGE_S


def test_expired_request_is_skipped_in_place_and_fifo_is_kept(tmp_path):
  session = Session(tmp_path / "sess").create({})
  session.submit("first")
  stale = session.submit("second")
  session.submit("third")
  _age_request(session, stale.req_id, seconds=999.0)

  popped = session.pop_requests()

  assert [r.cmd for r in popped] == ["first", "third"]
  answer = session.poll(stale.req_id)
  assert answer is not None and not answer.ok


def test_request_without_timestamp_is_treated_as_fresh(tmp_path):
  session = Session(tmp_path / "sess").create({})
  # An agent-side writer that predates the created_at field.
  (session.inbox / "000-legacy.json").write_text(
      json.dumps({"req_id": "legacy01", "cmd": "status", "args": {}})
  )

  popped = session.pop_requests()

  assert [r.cmd for r in popped] == ["status"]
  assert session.poll("legacy01") is None  # No refusal was written.


def test_pop_requests_honours_max_age_parameter(tmp_path):
  session = Session(tmp_path / "sess").create({})
  request = session.submit("status")
  _age_request(session, request.req_id, seconds=10.0)

  assert session.pop_requests(max_age_s=5.0) == []
  answer = session.poll(request.req_id)
  assert answer is not None and "limit 5s" in (answer.error or "")

  ok_request = session.submit("status")
  assert [r.req_id for r in session.pop_requests(max_age_s=5.0)] == [ok_request.req_id]


def test_create_sweeps_backlog_and_claimed_orphans(tmp_path):
  session = Session(tmp_path / "sess").create({})
  queued = session.submit("stop_training")
  orphan = session.submit("load_checkpoint", path="x")
  orphan_file = next(p for p in session.inbox.glob("*.json") if orphan.req_id in p.name)
  os.replace(orphan_file, orphan_file.with_suffix(".claimed"))  # Died mid-claim.
  (session.inbox / "000-torn.json").write_text('{"req_id": "to')  # Killed mid-write.

  Session(session.dir).create({})  # The trainer restarts on the same directory.

  assert list(session.inbox.iterdir()) == []
  assert session.pop_requests() == []  # Nothing survives to execute.
  for req in (queued, orphan):
    answer = session.poll(req.req_id)
    assert answer is not None and not answer.ok
    assert "expired before execution" in (answer.error or "")
    assert "previous training process" in (answer.error or "")
  expired = {e["req_id"] for e in session.events() if e["kind"] == "request_expired"}
  assert expired == {queued.req_id, orphan.req_id}


# Tail reads: last_n must cost the tail of the file, not the whole file, and
# must mean exactly what a whole-file read-and-slice always meant.


def test_tail_read_matches_whole_file_slicing_across_blocks(tmp_path, monkeypatch):
  from rlmcp import session as session_mod

  monkeypatch.setattr(session_mod, "_TAIL_BLOCK_BYTES", 64)  # Force multi-block reads.
  path = tmp_path / "metrics.jsonl"
  rows = [{"iteration": i, "pad": "x" * (i % 37)} for i in range(60)]
  path.write_text("".join(json.dumps(r) + "\n" for r in rows))
  assert path.stat().st_size > 3 * 64  # The file genuinely spans several blocks.

  assert session_mod.read_jsonl(path) == rows
  for last_n in (1, 2, 3, 7, 59, 60, 61, 1000):
    assert session_mod.read_jsonl(path, last_n=last_n) == rows[-last_n:], last_n
  assert session_mod.read_jsonl(path, last_n=0) == []
  assert session_mod.read_jsonl(tmp_path / "absent.jsonl", last_n=5) == []
  assert session_mod.read_jsonl(tmp_path / "absent.jsonl") == []


def test_tail_read_tolerates_a_torn_last_line(tmp_path, monkeypatch):
  from rlmcp import session as session_mod

  monkeypatch.setattr(session_mod, "_TAIL_BLOCK_BYTES", 32)
  path = tmp_path / "metrics.jsonl"
  rows = [{"iteration": i} for i in range(20)]
  path.write_text("".join(json.dumps(r) + "\n" for r in rows))
  with path.open("a") as f:
    f.write('{"iteration": 20, "torn": "' + "y" * 100)  # Longer than one block.

  # last_n counts physical lines; the torn one occupies a slot and is skipped.
  assert session_mod.read_jsonl(path, last_n=3) == rows[-2:]
  assert session_mod.read_jsonl(path, last_n=1) == []


# Outbox lifecycle: responses are consumed by their one reader and swept by
# the trainer's heartbeat, so the directory stops growing one file per command.


def test_wait_consumes_the_response_file(tmp_path):
  from rlmcp.session import Response

  session = Session(tmp_path / "sess").create({})
  request = session.submit("status")
  session.respond(Response(req_id=request.req_id, ok=True, result=1))

  answer = session.wait(request.req_id, timeout=2.0)

  assert answer.ok and answer.result == 1
  assert list(session.outbox.iterdir()) == []  # The requester consumed it.
  assert session.poll(request.req_id) is None


def test_poll_leaves_the_response_unless_asked_to_consume(tmp_path):
  from rlmcp.session import Response

  session = Session(tmp_path / "sess").create({})
  session.respond(Response(req_id="r1", ok=True))

  assert session.poll("r1") is not None
  assert session.poll("r1") is not None  # Default polling is a pure read.
  assert session.poll("r1", consume=True) is not None
  assert session.poll("r1") is None


def test_a_malformed_response_is_left_on_disk_not_consumed(tmp_path):
  """Parse before unlink: valid JSON that is not a Response must return None
  and stay put as evidence, never be consumed and then crash the reader."""
  session = Session(tmp_path / "sess").create({})
  path = session.outbox / "r1.json"
  path.write_text(json.dumps({"result": 1, "finished_at": 0.0}))  # No req_id/ok.

  assert session.poll("r1", consume=True) is None

  assert path.exists()  # The evidence stays for the heartbeat prune to age out.


def test_publish_status_prunes_old_responses_on_a_throttle(tmp_path):
  from rlmcp.session import Response

  session = Session(tmp_path / "sess").create({})
  session.respond(Response(req_id="old", ok=True))
  session.respond(Response(req_id="fresh", ok=True))
  os.utime(session.outbox / "old.json", (0, 0))

  session.publish_status({"iteration": 1})

  assert not (session.outbox / "old.json").exists()
  assert (session.outbox / "fresh.json").exists()

  # Within the throttle interval nothing is pruned, however old it is: the
  # pause loop publishes several times a second and must not rescan each time.
  session.respond(Response(req_id="old2", ok=True))
  os.utime(session.outbox / "old2.json", (0, 0))
  session.publish_status({"iteration": 2})
  assert (session.outbox / "old2.json").exists()


# Liveness: dead is noticed immediately, stalled is a state of its own, and a
# recycled pid does not keep a corpse looking alive for days.


def _rewrite_pid(session, pid):
  info = json.loads(session.session_file.read_text())
  info["pid"] = pid
  session.session_file.write_text(json.dumps(info))


def _rewrite_heartbeat_age(session, seconds):
  status = json.loads(session.status_file.read_text())
  status["updated_at"] = time.time() - seconds
  session.status_file.write_text(json.dumps(status))


def test_wait_reports_a_dead_trainer_on_the_first_poll(tmp_path):
  session = Session(tmp_path / "sess").create({})
  request = session.submit("status")
  _rewrite_pid(session, 999_999_999)
  dead = Session(session.dir)  # Fresh handle: no cached pid.

  t0 = time.monotonic()
  answer = dead.wait(request.req_id, timeout=30.0, interval=0.05)
  elapsed = time.monotonic() - t0

  assert not answer.ok
  assert "not running" in (answer.error or "")
  assert elapsed < 5.0  # The old `deadline - timeout/2` guard would sit ~15s.


def test_wait_returns_a_response_the_trainer_wrote_before_dying(tmp_path):
  from rlmcp.session import Response

  session = Session(tmp_path / "sess").create({})
  request = session.submit("status")
  session.respond(Response(req_id=request.req_id, ok=True, result=5))
  _rewrite_pid(session, 999_999_999)

  answer = Session(session.dir).wait(request.req_id, timeout=2.0)

  assert answer.ok and answer.result == 5


def test_is_alive_caches_the_pid_and_forgets_it_on_death(tmp_path):
  session = Session(tmp_path / "sess").create({})
  assert session.is_alive()

  # The cached pid (this process) answers without re-reading session.json...
  _rewrite_pid(session, 999_999_999)
  assert session.is_alive()

  # ...but a dead answer clears the cache, so a trainer restarted onto the
  # same directory (new pid in session.json) is noticed on the next call.
  fresh = Session(session.dir)
  assert not fresh.is_alive()
  _rewrite_pid(session, os.getpid())
  assert fresh.is_alive()


def test_liveness_derives_running_stalled_and_dead(tmp_path):
  session = Session(tmp_path / "sess").create({})
  session.publish_status({"iteration": 3})
  assert session.liveness() == "running"

  _rewrite_heartbeat_age(session, Session.STALL_AFTER_S + 60)
  info = session.liveness_info()
  assert info["state"] == "stalled"
  assert info["pid_alive"] is True
  assert "long iteration" in info["note"]  # Truthful, not fatal.

  # A recycled pid can pass the existence check for days; a heartbeat a day
  # stale outranks it and the run is treated as dead, with a note saying why.
  _rewrite_heartbeat_age(session, Session.PRESUMED_DEAD_AFTER_S + 60)
  info = session.liveness_info()
  assert info["state"] == "dead"
  assert info["pid_alive"] is True
  assert "recycled" in info["note"]

  _rewrite_pid(session, 999_999_999)
  assert Session(session.dir).liveness() == "dead"

  # No status.json yet: a run still starting up is running, not stalled.
  assert Session(tmp_path / "young").create({}).liveness() == "running"


# Discovery: no crawling of artifact trees, and "newest" means started_at.


def test_discovery_skips_junk_dirs_but_still_finds_nested_sessions(tmp_path):
  real = Session(tmp_path / "logs" / "run1" / "rlmcp").create({"task": "walk"})
  # Decoys planted where no session belongs: inside the session's own
  # artifacts/ and inside a virtualenv. A blind rglob would crawl (and find)
  # both; the pruned walk must still see the real session above them.
  decoy = real.dir / "artifacts" / "session.json"
  decoy.write_text(json.dumps({"schema_version": 1, "started_at": 9e9}))
  venv_decoy = tmp_path / ".venv" / "session.json"
  venv_decoy.parent.mkdir()
  venv_decoy.write_text(json.dumps({"schema_version": 1, "started_at": 9e9}))

  assert [s.dir for s in iter_sessions(tmp_path)] == [real.dir]
  assert Session.find_latest(tmp_path).dir == real.dir


def test_sessions_are_ordered_by_started_at_not_path(tmp_path):
  # Path order (a, b, c) deliberately contradicts start order (c, a, b).
  for name, started in (("a", 200.0), ("b", 100.0), ("c", 300.0)):
    session = Session(tmp_path / name).create({})
    info = json.loads(session.session_file.read_text())
    info["started_at"] = started
    session.session_file.write_text(json.dumps(info))

  assert [s.dir.name for s in iter_sessions(tmp_path)] == ["c", "a", "b"]
  assert Session.find_latest(tmp_path).dir.name == "c"


# Strict JSON: a diverged run's NaN metrics must not poison the files.


def test_non_finite_values_are_written_as_null(tmp_path):
  session = Session(tmp_path / "sess").create({})
  session.publish_status({"loss": float("nan"), "nested": {"v": float("inf")}})
  session.append_metrics(1, {"reward": float("nan"), "fine": 1.5})
  session.append_event("note", {"values": [float("-inf"), 2.0]})

  status_text = session.status_file.read_text()
  metrics_text = session.metrics_file.read_text()
  events_text = session.events_file.read_text()
  for text in (status_text, metrics_text, events_text):
    assert "NaN" not in text and "Infinity" not in text
  json.loads(status_text)  # Strict-parseable, the way jq or a JS client reads it.
  for line in metrics_text.splitlines() + events_text.splitlines():
    json.loads(line)

  assert session.status()["loss"] is None
  assert session.status()["nested"]["v"] is None
  row = session.metrics()[-1]
  assert row["reward"] is None and row["fine"] == 1.5
  assert session.events()[-1]["values"] == [None, 2.0]


def test_numpy_nan_scalars_cannot_sneak_past_the_sanitizer(tmp_path):
  np = pytest.importorskip("numpy")

  session = Session(tmp_path / "sess").create({})
  session.append_metrics(
      2, {"f32": np.float32("nan"), "f64": np.float64("nan"), "fine": np.float32(0.5)}
  )

  text = session.metrics_file.read_text()
  assert "NaN" not in text
  json.loads(text.splitlines()[-1])
  row = session.metrics()[-1]
  assert row["f32"] is None and row["f64"] is None
  assert row["fine"] == pytest.approx(0.5)


# MCP server image helpers. The server module imports its SDK at module scope,
# so these only run when one is installed (e.g. `uv run --with mcp`); the rest
# of the suite must stay green without it.


def _server_module():
  pytest.importorskip("mcp")
  from rlmcp.server import mcp_server

  return mcp_server


def _write_png(path, size=(32, 24), color=(10, 200, 30)):
  from PIL import Image as PILImage

  PILImage.new("RGB", size, color).save(path)
  return path


def test_image_format_normalises_to_standard_mime_names():
  srv = _server_module()
  assert srv._image_format(".jpg") == "jpeg"
  assert srv._image_format(".JPG") == "jpeg"
  assert srv._image_format(".jpeg") == "jpeg"
  assert srv._image_format(".png") == "png"
  assert srv._image_format("") == "png"


def test_image_result_keeps_payload_and_attaches_image(tmp_path):
  srv = _server_module()
  png = _write_png(tmp_path / "frame.png")
  payload = {"ok": True, "jerk_hz": 3.2, "image_path": str(png)}

  out = srv._image_result(payload)

  assert isinstance(out, list) and len(out) == 2
  assert out[0] is payload  # The numeric report survives untouched.
  image = out[1]
  assert type(image).__name__ == "Image"
  assert image._mime_type == "image/png"
  assert image.data == png.read_bytes()


def test_image_result_uses_image_jpeg_for_jpg_files(tmp_path):
  srv = _server_module()
  from PIL import Image as PILImage

  jpg = tmp_path / "frame.jpg"
  PILImage.new("RGB", (32, 24), (5, 5, 5)).save(jpg)

  out = srv._image_result({"ok": True, "image_path": str(jpg)})

  assert out[1]._mime_type == "image/jpeg"


def test_image_result_without_readable_image_returns_payload(tmp_path):
  srv = _server_module()
  missing = {"ok": True, "image_path": str(tmp_path / "gone.png")}
  failed = {"ok": False, "error": "render failed", "image_path": "ignored"}
  plain = {"ok": True, "result": 4}

  assert srv._image_result(missing) is missing
  assert srv._image_result(failed) is failed
  assert srv._image_result(plain) is plain


def test_oversized_image_is_downscaled_under_the_limit(tmp_path):
  srv = _server_module()
  import numpy as np
  from PIL import Image as PILImage

  rng = np.random.default_rng(0)  # Noise compresses badly: a reliably big PNG.
  big = tmp_path / "big.png"
  PILImage.fromarray(rng.integers(0, 255, size=(1200, 1600, 3), dtype="uint8")).save(big)
  assert big.stat().st_size > srv.IMAGE_BYTE_LIMIT

  data, fmt, note = srv._prepare_image(big, byte_limit=srv.IMAGE_BYTE_LIMIT, max_dim=512)

  assert note is None and data is not None
  assert len(data) <= srv.IMAGE_BYTE_LIMIT
  assert fmt in ("png", "jpeg")
  with PILImage.open(io.BytesIO(data)) as reopened:
    assert max(reopened.size) <= 512


def test_unshrinkable_image_falls_back_to_path_with_note(tmp_path, monkeypatch):
  srv = _server_module()
  png = _write_png(tmp_path / "frame.png", size=(64, 64))
  monkeypatch.setattr(srv, "IMAGE_BYTE_LIMIT", 10)  # Nothing real fits in 10 bytes.
  payload = {"ok": True, "score": 1.0, "image_path": str(png)}

  out = srv._image_result(payload)

  assert isinstance(out, dict)
  assert out["score"] == 1.0 and out["image_path"] == str(png)
  assert "reply budget" in out["image_note"]


# MCP server session handling: one pinned run per server, truthful liveness,
# and a post-mortem that answers from disk instead of shrugging.


def _make_run(tmp_path, name, started_at=None, alive=True, iterations=5):
  """A session directory shaped like a real run: logs/<name>/rlmcp."""
  session = Session(tmp_path / name / "rlmcp").create({"task": name})
  for i in range(iterations):
    session.append_metrics(i, {"Train/mean_reward": float(i), "rlmcp/level": i * 0.1})
  session.append_event("note", {"text": f"hello from {name}"})
  session.publish_params({"reward.x.weight": {"category": "reward", "value": 1.0}})
  session.publish_status({"iteration": iterations - 1})
  info = json.loads(session.session_file.read_text())
  if started_at is not None:
    info["started_at"] = started_at
  if not alive:
    info["pid"] = 999_999_999
  session.session_file.write_text(json.dumps(info))
  return Session(session.dir)  # Fresh handle: no cached pid.


def test_server_stays_pinned_when_its_run_dies_and_a_newer_one_appears(tmp_path):
  srv = _server_module()
  old = _make_run(tmp_path, "old", started_at=100.0, alive=False)
  handle = srv._SessionHandle(None, str(tmp_path))
  assert handle.get().dir == old.dir  # Pinned at first use.

  _make_run(tmp_path, "newer", started_at=200.0)  # Newer and alive.
  status = srv._status_payload(handle)

  # The old handle would silently re-resolve to "newer" here; the pin holds
  # and the death is reported instead.
  assert status["session_dir"] == str(old.dir)
  assert status["state"] == "dead"
  assert status["session"] == "old/rlmcp"


def test_switch_session_is_the_only_way_to_move(tmp_path):
  srv = _server_module()
  a = _make_run(tmp_path, "a", started_at=100.0)
  b = _make_run(tmp_path, "b", started_at=200.0)
  handle = srv._SessionHandle(str(a.dir), str(tmp_path))
  assert handle.get().dir == a.dir

  out = srv._switch_payload(handle, str(b.dir))
  assert out["ok"] and out["session_dir"] == str(b.dir)
  assert out["task"] == "b" and out["state"] == "running"
  assert handle.get().dir == b.dir

  out = srv._switch_payload(handle, "newest")
  assert out["ok"] and out["session_dir"] == str(b.dir)  # b started later.

  out = srv._switch_payload(handle, str(tmp_path / "missing"))
  assert not out["ok"] and "No rlmcp session" in out["error"]
  assert handle.get().dir == b.dir  # A failed switch moves nothing.


def test_live_commands_refuse_with_pointers_when_the_trainer_is_dead(tmp_path):
  srv = _server_module()
  dead = _make_run(tmp_path, "gone", alive=False)
  handle = srv._SessionHandle(str(dead.dir), None)

  out = srv._call(handle, "set_parameter", key="x", value=1)

  assert not out["ok"]
  assert "dead" in out["error"] and "set_parameter" in out["error"]
  assert "switch_session" in out["hint"]
  assert out["session"] == "gone/rlmcp"
  assert out["last_status"]["iteration"] == 4
  assert list(dead.inbox.iterdir()) == []  # Nothing was queued at a corpse.


def test_get_metrics_answers_from_disk_for_a_dead_run(tmp_path):
  srv = _server_module()
  dead = _make_run(tmp_path, "done", alive=False)
  handle = srv._SessionHandle(str(dead.dir), None)

  out = srv._get_metrics_impl(handle, ["Train/mean_reward"], 3)

  assert out["ok"] and out["live"] is False and out["source"] == "metrics.jsonl"
  assert out["metrics"]["Train/mean_reward"] == [[2, 2.0], [3, 3.0], [4, 4.0]]
  assert out["session"] == "done/rlmcp"
  assert out["summary"]["Train/mean_reward"]["latest"] == 4.0

  names = srv._list_metrics_impl(handle, "reward")
  assert names["ok"] and names["metrics"] == ["Train/mean_reward"]

  params = srv._list_parameters_impl(handle, None, "weight")
  assert params["ok"] and params["live"] is False and params["count"] == 1

  events = srv._events_payload(handle.get(), 10)
  assert events["ok"] and events["session"] == "done/rlmcp"
  assert events["events"][-1]["kind"] == "note"


def test_plot_metrics_renders_from_disk_for_a_dead_run(tmp_path):
  srv = _server_module()
  pytest.importorskip("matplotlib")
  dead = _make_run(tmp_path, "plotme", alive=False, iterations=30)
  handle = srv._SessionHandle(str(dead.dir), None)

  out = srv._plot_metrics_impl(handle, ["Train/mean_reward"], 20, 3, None)

  assert isinstance(out, list) and len(out) == 2  # Payload plus attached image.
  payload = out[0]
  assert payload["ok"] and payload["live"] is False
  assert Path(payload["image_path"]).exists()
  assert payload["session"] == "plotme/rlmcp"


def test_list_sessions_reports_start_state_and_pin_newest_first(tmp_path):
  srv = _server_module()
  older = _make_run(tmp_path, "older", started_at=100.0, alive=False)
  newer = _make_run(tmp_path, "newer", started_at=200.0)
  handle = srv._SessionHandle(None, str(tmp_path))
  handle.get()  # Pins "newer", the newest at first use.

  rows = srv._sessions_payload(handle)

  assert [r["session_dir"] for r in rows] == [str(newer.dir), str(older.dir)]
  assert rows[0]["started_at"] == 200.0 and rows[0]["state"] == "running"
  assert rows[0]["pinned"] is True
  assert rows[1]["state"] == "dead" and rows[1]["pinned"] is False


def test_artifacts_are_listable_without_a_live_trainer(tmp_path):
  srv = _server_module()
  dead = _make_run(tmp_path, "arty", alive=False)
  dead.artifact_path("metrics.png").write_bytes(b"png")
  dead.artifact_path("clip.mp4").write_bytes(b"mp4mp4")

  out = srv._artifacts_payload(dead)

  assert out["ok"] and out["count"] == 2
  assert {r["name"] for r in out["artifacts"]} == {"metrics.png", "clip.mp4"}
  assert all(r["bytes"] > 0 for r in out["artifacts"])


# The client surface: what a reader of a run may use, and nothing else.


def test_the_local_session_satisfies_the_client_protocol(tmp_path):
  session = Session(tmp_path / "run" / "rlmcp").create({})

  assert isinstance(session, SessionClient)


def test_every_name_on_the_wire_surface_exists(tmp_path):
  """`WIRE_SURFACE` is the promise; this is the check that it is not fiction."""
  session = Session(tmp_path / "run" / "rlmcp").create({})

  missing = [name for name in WIRE_SURFACE if not hasattr(session, name)]

  assert missing == []


def test_the_wire_surface_does_not_grow_by_accident():
  """Adding a name here is a decision about the wire, so it is written twice.

  A second implementation lives behind a connection, and every name added to
  this list is one it has to answer. That should cost a moment's thought and a
  failing test, not a passing import.
  """
  assert set(WIRE_SURFACE) == {
      "address", "key", "name",
      "info", "status", "params",
      "metrics", "metrics_count", "events",
      "list_artifacts", "read_artifact",
      "submit", "poll", "wait", "call",
      "liveness", "liveness_info",
  }


def test_a_session_names_itself_three_ways(tmp_path):
  session = Session(tmp_path / "2026-01-01_g1" / "rlmcp").create({})

  assert session.address == str(tmp_path / "2026-01-01_g1" / "rlmcp")
  assert session.key == "2026-01-01_g1/rlmcp"   # tells two runs apart
  assert session.name == "2026-01-01_g1"        # what a plot title wants


def test_key_falls_back_to_the_leaf_at_the_filesystem_root():
  """A session directly under `/` has no parent name to disambiguate with."""
  session = Session("/rlmcp")

  assert session.key == "rlmcp"
  assert session.name == "rlmcp"


def test_metrics_count_is_the_total_not_the_window(tmp_path):
  session = Session(tmp_path / "sess").create({})
  for i in range(5):
    session.append_metrics(i, {"reward": float(i)})

  assert session.metrics_count() == 5
  assert len(session.metrics(last_n=2)) == 2


def test_metrics_count_of_a_run_that_logged_nothing_is_zero(tmp_path):
  assert Session(tmp_path / "sess").create({}).metrics_count() == 0


def test_list_artifacts_reports_names_sizes_and_newest_first(tmp_path):
  session = Session(tmp_path / "sess").create({})
  older = session.artifact_path("first.png")
  older.write_bytes(b"one")
  newer = session.artifact_path("second.mp4")
  newer.write_bytes(b"two!")
  os.utime(older, (1, 1))

  rows = session.list_artifacts()

  assert [r["name"] for r in rows] == ["second.mp4", "first.png"]
  assert [r["bytes"] for r in rows] == [4, 3]


def test_list_artifacts_skips_directories(tmp_path):
  session = Session(tmp_path / "sess").create({})
  session.artifact_path("keep.png").write_bytes(b"x")
  (session.artifacts / "traces").mkdir()

  assert [r["name"] for r in session.list_artifacts()] == ["keep.png"]


def test_read_artifact_returns_the_bytes(tmp_path):
  session = Session(tmp_path / "sess").create({})
  session.artifact_path("shot.png").write_bytes(b"\x89PNG")

  assert session.read_artifact("shot.png") == b"\x89PNG"


@pytest.mark.parametrize("name", ["../session.json", "/etc/passwd", "sub/shot.png", "", "..", "."])
def test_read_artifact_refuses_anything_that_is_not_a_bare_name(tmp_path, name):
  """The argument crosses a network in the remote implementation of this."""
  session = Session(tmp_path / "sess").create({})

  with pytest.raises(ValueError):
    session.read_artifact(name)


# Sequence numbers: the cursor a reader that is not on this machine needs.


def test_status_carries_a_sequence_that_grows(tmp_path):
  session = Session(tmp_path / "sess").create({})

  seqs = []
  for i in range(3):
    session.publish_status({"iteration": i})
    seqs.append(session.status()["seq"])

  assert seqs == [1, 2, 3]


def test_events_and_metrics_are_numbered_from_one(tmp_path):
  session = Session(tmp_path / "sess").create({})
  session.append_metrics(0, {"reward": 1.0})
  session.append_metrics(1, {"reward": 2.0})
  session.append_event("note", {"text": "hello"})

  assert [r["seq"] for r in session.metrics()] == [1, 2]
  assert [e["seq"] for e in session.events()] == [1]


def test_a_second_process_on_the_same_directory_continues_the_count(tmp_path):
  """A trainer restarted onto a directory must not replay sequence numbers.

  A reader holding `seq=7` would otherwise be handed a different row 8, and
  would never know it had missed the first one.
  """
  first = Session(tmp_path / "sess").create({})
  first.publish_status({"iteration": 1})
  first.append_event("note", {"text": "before"})
  first.append_metrics(0, {"reward": 1.0})

  second = Session(tmp_path / "sess")
  second.publish_status({"iteration": 2})
  second.append_event("note", {"text": "after"})
  second.append_metrics(1, {"reward": 2.0})

  assert second.status()["seq"] == 2
  assert [e["seq"] for e in second.events()] == [1, 2]
  assert [r["seq"] for r in second.metrics()] == [1, 2]


def test_since_seq_returns_only_what_is_new(tmp_path):
  session = Session(tmp_path / "sess").create({})
  for i in range(5):
    session.append_event("note", {"text": str(i)})
    session.append_metrics(i, {"reward": float(i)})

  fresh = session.events(since_seq=3)

  assert [e["text"] for e in fresh] == ["3", "4"]
  assert [r["iteration"] for r in session.metrics(since_seq=3)] == [3, 4]
  assert session.events(since_seq=5) == []
  assert len(session.events(since_seq=0)) == 5


def test_a_log_written_before_sequences_existed_is_read_once(tmp_path):
  """Old runs have no cursor: a reader starting from scratch gets their
  history, and one holding a cursor is not handed it again every poll."""
  session = Session(tmp_path / "sess").create({})
  session.events_file.write_text(
      '{"t": 1.0, "kind": "note", "text": "old"}\n'
  )

  assert [e["text"] for e in session.events(since_seq=0)] == ["old"]
  assert session.events(since_seq=1) == []


def test_a_sequence_never_shows_up_as_a_metric(tmp_path):
  """`seq` is bookkeeping in a row of measurements, which is a trap.

  Every reader that asks what a run logged walks the row's keys, so a new
  field lands on the CLI's metric list and the studio's headline unless it is
  named as reserved.
  """
  from rlmcp.cli import _default_metric_names

  # A task whose metrics the CLI has no preferred name for, so the default
  # selection falls back to "whatever this run logged" -- which is where a
  # bookkeeping field becomes something a user is offered to plot.
  session = Session(tmp_path / "sess").create({})
  session.append_metrics(0, {"cartpole/angle": 1.0})

  names = {k for row in session.metrics() for k in row if k not in RESERVED_METRIC_KEYS}

  assert names == {"cartpole/angle"}
  assert _default_metric_names(session) == ["cartpole/angle"]
