"""The store contract.

Every test here is written against the `RecordStore` protocol, never against a
filesystem. When the fleet backend arrives, it joins the `store` fixture's params
and this file runs against it unchanged — which is the only real proof that the
seam between "one machine" and "a hundred" is where it claims to be.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import time
from pathlib import Path

import pytest

from rlmcp.records.filestore import FileStore
from rlmcp.records.record import Falsifier, RunRecord, Weights
from rlmcp.records.store import (
    ConflictError,
    SlotUnavailable,
    StoreError,
    next_display_id,
)
from rlmcp.records.validate import check_verdict_change


@pytest.fixture(params=["file"])
def store(request, tmp_path):
  """A RecordStore. Add "service" here in the fleet slice; change nothing else."""
  if request.param == "file":
    return FileStore(tmp_path / "records", slots=1)
  raise AssertionError(f"unknown store {request.param}")


# Ids and creation.


def test_next_display_id_counts_past_what_exists():
  assert next_display_id([]) == "001"
  assert next_display_id(["001", "002"]) == "003"
  assert next_display_id(["001", "R5", "legacy"]) == "002"  # non-numeric ignored


def test_the_store_assigns_ids_and_sequence(store):
  """Callers never pick ids — two machines would both choose the same one."""
  first = store.new_record("baseline")
  second = store.new_record("follow_up")

  assert (first.id, second.id) == ("001", "002")
  assert second.seq > first.seq


def test_a_created_record_is_readable_back(store):
  created = store.new_record(
      "flat_first",
      stage="locomotion",
      hypothesis="flat before rough",
      falsifier=Falsifier(prose="never leaves the spawn"),
      change=["flat terrain only"],
  )

  loaded = store.get_record(created.id)

  assert loaded is not None
  assert loaded.slug == "flat_first"
  assert loaded.hypothesis == "flat before rough"
  assert loaded.falsifier.prose == "never leaves the spawn"


def test_a_slug_is_normalised_on_the_way_in(store):
  assert store.new_record("Flat, then Rough!").slug == "flat_then_rough"


def test_getting_an_unknown_record_returns_none(store):
  assert store.get_record("999") is None


def test_records_list_in_sequence_order(store):
  ids = [store.new_record(f"run_{i}").id for i in range(4)]
  assert [r.id for r in store.list_records()] == ids


def test_updating_a_record_overwrites_it(store):
  record = store.new_record("run")
  record.verdict = "falsified"
  record.outcome = "the hypothesis died"
  record.metrics = [["falsifier fired", "yes"]]

  store.put_record(record)

  assert store.get_record(record.id).verdict == "falsified"


def test_deleting_removes_it_from_listings(store):
  record = store.new_record("doomed")
  assert store.delete_record(record.id) is True
  assert store.get_record(record.id) is None
  assert store.delete_record(record.id) is False


# Query.


def test_query_filters_on_the_lineage_and_the_text(store):
  root = store.new_record("baseline", stage="locomotion",
                          hypothesis="walking works on flat")
  child = store.new_record("stairs", stage="locomotion", parent=root.id,
                           hypothesis="stairs need clearance")
  store.new_record("gripper", stage="manipulation", proposed_by="orchestrator")

  assert [r.id for r in store.query(parent=root.id)] == [child.id]
  assert {r.id for r in store.query(stage="locomotion")} == {root.id, child.id}
  assert [r.id for r in store.query(proposed_by="orchestrator")][0] not in (root.id, child.id)
  assert [r.id for r in store.query(text="clearance")] == [child.id]


def test_query_respects_a_limit(store):
  for i in range(5):
    store.new_record(f"run_{i}", stage="s")
  assert len(store.query(stage="s", limit=2)) == 2


# Documents.


def test_plan_and_report_live_beside_the_record(store):
  record = store.new_record("documented")

  store.write_document(record.id, "PLAN.md", "# Hypothesis\nit works")
  store.write_document(record.id, "REPORT.md", "# Outcome\nit did not")

  assert "it works" in store.read_document(record.id, "PLAN.md")
  assert "it did not" in store.read_document(record.id, "REPORT.md")
  assert store.read_document(record.id, "MISSING.md") is None


# Media.


def test_an_asset_is_copied_out_of_its_session(tmp_path, store):
  """A record has to survive its log directory being cleaned."""
  source = tmp_path / "session" / "artifacts"
  source.mkdir(parents=True)
  frame = source / "shot.png"
  frame.write_bytes(b"\x89PNG fake")

  record = store.new_record("with_media")
  key = store.media.put(record.id, str(frame), caption="on stairs", kind="plots")

  assert store.media.exists(key)
  frame.unlink()  # the session goes away
  assert store.media.exists(key)  # the record does not
  assert open(store.media.get(key), "rb").read() == b"\x89PNG fake"


def test_recording_a_missing_asset_is_an_error(store):
  record = store.new_record("run")
  with pytest.raises(StoreError, match="No such file"):
    store.media.put(record.id, "/nope/does_not_exist.png")


# Leases — the generalised mutex.


def test_claiming_a_slot_marks_the_record(store):
  record = store.new_record("training")

  claimed = store.claim(record.id, slot="gpu0", holder="pid-42")

  assert claimed.lease.slot == "gpu0"
  assert claimed.lease.holder == "pid-42"
  assert not claimed.lease.expired()


def test_a_second_claim_on_the_same_slot_is_refused(store):
  first = store.new_record("first")
  second = store.new_record("second")
  store.claim(first.id, slot="gpu0")

  with pytest.raises(SlotUnavailable, match="held by run"):
    store.claim(second.id, slot="gpu0")


def test_claiming_beyond_the_slot_count_is_refused(store):
  """One machine, one trainer — the original mutex, expressed as capacity."""
  first = store.new_record("first")
  second = store.new_record("second")
  store.claim(first.id, slot="gpu0")

  with pytest.raises(SlotUnavailable, match="slot"):
    store.claim(second.id, slot="gpu1")


def test_releasing_frees_the_slot(store):
  first = store.new_record("first")
  second = store.new_record("second")
  store.claim(first.id, slot="gpu0")

  store.release(first.id)

  assert store.claim(second.id, slot="gpu0").lease.slot == "gpu0"


def test_a_heartbeat_renews_the_lease(store):
  record = store.claim(store.new_record("run").id, slot="gpu0", ttl_seconds=100.0)
  original = record.lease.renewed_at
  record.lease.renewed_at -= 50
  store.put_record(record)

  renewed = store.heartbeat(record.id)

  assert renewed.lease.renewed_at >= original


def test_heartbeating_without_a_lease_returns_none(store):
  assert store.heartbeat(store.new_record("run").id) is None


def test_a_dead_job_releases_its_slot_and_stops_being_running(store):
  """A stale 'running' must not block the next launch forever."""
  record = store.new_record("crashed", verdict="running")
  store.claim(record.id, slot="gpu0", ttl_seconds=60.0)
  stale = store.get_record(record.id)
  stale.lease.renewed_at -= 10_000
  store.put_record(stale)

  reaped = store.reap_expired()

  assert [r.id for r in reaped] == [record.id]
  assert store.get_record(record.id).verdict == "interrupted"
  assert store.get_record(record.id).lease is None
  assert store.get_record(record.id).links["reaped"] == "lease-expired"
  # And the slot is usable again.
  other = store.new_record("next")
  assert store.claim(other.id, slot="gpu0").lease.slot == "gpu0"


def test_a_lease_whose_process_is_gone_is_reaped_without_waiting(store):
  """A TTL alone means a crashed trainer keeps its slot for the whole timeout.

  Naming a holder makes this a runner lease -- the kind whose pid means
  something. A bare claim is a manual reservation and is never pid-reaped.
  """
  record = store.new_record("crashed", verdict="running")
  store.claim(record.id, slot="gpu0", holder="pid-trainer", ttl_seconds=100_000.0)
  stale = store.get_record(record.id)
  stale.lease.pid = 999_999_999  # a pid that cannot exist, on this host
  store.put_record(stale)

  reaped = store.reap_expired()

  assert [r.id for r in reaped] == [record.id]
  assert store.get_record(record.id).verdict == "interrupted"
  assert store.get_record(record.id).links["reaped"] == "holder-gone"


def test_a_lease_held_on_another_host_is_left_to_its_ttl(store):
  """Only the TTL can judge a process this machine cannot see."""
  record = store.new_record("remote", verdict="running")
  store.claim(record.id, slot="gpu0", holder="pid-trainer", ttl_seconds=100_000.0)
  stale = store.get_record(record.id)
  stale.lease.host = "some-other-node"
  stale.lease.pid = 999_999_999
  store.put_record(stale)

  assert store.reap_expired() == []
  assert store.get_record(record.id).verdict == "running"


def test_reaping_leaves_a_closed_record_alone(store):
  record = store.new_record("finished", verdict="provisional")
  store.claim(record.id, slot="gpu0")
  stale = store.get_record(record.id)
  stale.lease.renewed_at -= 10_000
  store.put_record(stale)

  store.reap_expired()

  assert store.get_record(record.id).verdict == "provisional"
  # Freeing the dead lease is housekeeping, not a death worth annotating.
  assert "reaped" not in store.get_record(record.id).links


# FileStore specifics: the index is derived, the files are the truth.


def test_the_index_can_be_destroyed_without_losing_anything(tmp_path):
  store = FileStore(tmp_path / "records")
  kept = store.new_record("keeper", stage="locomotion", hypothesis="findable")

  store.index_path.unlink()
  rebuilt = FileStore(tmp_path / "records")

  assert rebuilt.reindex() == 1
  assert [r.id for r in rebuilt.query(stage="locomotion")] == [kept.id]
  assert rebuilt.get_record(kept.id).hypothesis == "findable"


def test_a_hand_edited_record_wins_over_the_index(tmp_path):
  """Files are the truth: editing meta.json in a PR has to take effect."""
  store = FileStore(tmp_path / "records")
  record = store.new_record("edited", hypothesis="before")

  meta = store.record_dir(record) / "meta.json"
  payload = json.loads(meta.read_text())
  payload["hypothesis"] = "after"
  meta.write_text(json.dumps(payload))

  assert store.get_record(record.id).hypothesis == "after"


def test_a_corrupt_record_is_skipped_not_fatal(tmp_path):
  store = FileStore(tmp_path / "records")
  good = store.new_record("good")
  broken = store.runs_dir / "099-broken"
  broken.mkdir()
  (broken / "meta.json").write_text("{not json")

  assert [r.id for r in store.list_records()] == [good.id]
  assert store.reindex() == 1


def test_the_store_reopens_where_it_left_off(tmp_path):
  first = FileStore(tmp_path / "records").new_record("first")
  second = FileStore(tmp_path / "records").new_record("second")

  assert (first.id, second.id) == ("001", "002")


def test_warm_start_lineage_survives_a_round_trip(store):
  root = store.new_record("scratch")
  warm = store.new_record(
      "finetune", parent=root.id, weights=Weights(root.id, "model_100.pt")
  )

  loaded = store.get_record(warm.id)

  assert loaded.parent == root.id
  assert loaded.weights.run == root.id
  assert loaded.weights.checkpoint == "model_100.pt"
  assert not loaded.from_scratch


def test_a_slug_is_capped_for_directory_names(store):
  assert len(store.new_record("word " * 60).slug) <= 80


# Concurrency — real processes, because this is a file-and-sqlite store.


def _mp():
  methods = multiprocessing.get_all_start_methods()
  return multiprocessing.get_context("fork" if "fork" in methods else "spawn")


def _create_racer(root, barrier, queue, i):
  store = FileStore(root)
  barrier.wait()
  queue.put((i, store.new_record(f"racer_{i}").id))


def _claim_racer(root, barrier, queue, i, record_id):
  store = FileStore(root)
  barrier.wait()
  try:
    store.claim(record_id, slot="gpu0")
    queue.put((i, "won"))
  except SlotUnavailable:
    queue.put((i, "lost"))


def _claim_and_exit(root, record_id, holder):
  FileStore(root).claim(record_id, slot="gpu0", holder=holder)


def _run_all(processes):
  for p in processes:
    p.start()
  for p in processes:
    p.join(30)
  assert all(p.exitcode == 0 for p in processes)


def test_racing_creators_each_get_their_own_id(tmp_path):
  """Eight simultaneous new_record calls: eight ids, eight dirs, no shadowing."""
  root = str(FileStore(tmp_path / "records").root)
  ctx = _mp()
  barrier, queue = ctx.Barrier(8), ctx.Queue()
  _run_all([
      ctx.Process(target=_create_racer, args=(root, barrier, queue, i))
      for i in range(8)
  ])

  results = dict(queue.get(timeout=10) for _ in range(8))
  store = FileStore(root)

  assert sorted(results.values()) == [str(n).zfill(3) for n in range(1, 9)]
  assert len(list(store.runs_dir.iterdir())) == 8
  # Every id resolves to the record its creator wrote — nobody was shadowed.
  for i, record_id in results.items():
    assert store.get_record(record_id).slug == f"racer_{i}"


def test_racing_claimants_for_one_slot_produce_exactly_one_lease(tmp_path):
  store = FileStore(tmp_path / "records", slots=1)
  records = [store.new_record(f"contender_{i}") for i in range(8)]
  ctx = _mp()
  barrier, queue = ctx.Barrier(8), ctx.Queue()
  _run_all([
      ctx.Process(target=_claim_racer,
                  args=(str(store.root), barrier, queue, i, records[i].id))
      for i in range(8)
  ])

  outcomes = [queue.get(timeout=10)[1] for _ in range(8)]

  assert outcomes.count("won") == 1
  assert outcomes.count("lost") == 7
  assert sum(1 for r in store.list_records() if r.lease is not None) == 1


# Versioned writes — a lost update is a conflict, not a silent overwrite.


def test_a_stale_write_raises_conflict_with_both_versions(store):
  created = store.new_record("contended")
  ours = store.get_record(created.id)
  theirs = store.get_record(created.id)

  theirs.outcome = "their result"
  store.put_record(theirs)
  ours.outcome = "our result"
  with pytest.raises(ConflictError) as caught:
    store.put_record(ours)

  assert caught.value.expected == 1
  assert caught.value.found == 2
  assert store.get_record(created.id).outcome == "their result"  # nothing lost


def test_update_record_retries_through_a_conflict(store):
  created = store.new_record("contended")
  attempts = []

  def mutate(record):
    if not attempts:  # A rival write lands between our read and our write.
      rival = store.get_record(created.id)
      rival.tags = ["rival"]
      store.put_record(rival)
    attempts.append(record.version)
    record.outcome = "ours, eventually"

  updated = store.update_record(created.id, mutate)

  assert len(attempts) == 2  # first try conflicted, the re-read won
  assert updated.outcome == "ours, eventually"
  loaded = store.get_record(created.id)
  assert loaded.outcome == "ours, eventually"
  assert loaded.tags == ["rival"]  # both writes survived


# Tombstones — an id is a citation, and a citation never changes referent.


def test_a_deleted_id_is_never_reissued_and_its_media_goes(tmp_path, store):
  store.new_record("keeper")
  doomed = store.new_record("doomed")
  asset = tmp_path / "shot.png"
  asset.write_bytes(b"\x89PNG fake")
  key = store.media.put(doomed.id, str(asset))

  assert store.delete_record(doomed.id) is True

  assert not store.media.exists(key)  # no inheritance by a later record
  assert store.new_record("successor").id == "003"  # 002 stays a tombstone
  store.reindex()
  assert store.new_record("after_reindex").id == "004"  # reindex keeps it


def test_a_tombstone_survives_losing_the_index(tmp_path):
  """Deleting the index is the documented recovery flow; a routine recovery
  must not rebind a citation to a different run."""
  store = FileStore(tmp_path / "records")
  store.new_record("keeper")
  doomed = store.new_record("doomed")
  assert store.delete_record(doomed.id) is True

  store.index_path.unlink()
  for sidecar in store.index_path.parent.glob(f"{store.index_path.name}-*"):
    sidecar.unlink()
  reopened = FileStore(tmp_path / "records")

  assert reopened.reindex() == 1
  assert reopened.new_record("successor").id == "003"  # 002 is still spoken for


def test_a_corrupt_field_warns_skips_and_reserves_its_id(tmp_path, capsys):
  """One meta.json with "seq": "seven" must not brick the records."""
  store = FileStore(tmp_path / "records")
  good = store.new_record("good")
  broken = store.runs_dir / "007-broken"
  broken.mkdir()
  (broken / "meta.json").write_text(
      json.dumps({"id": "007", "slug": "broken", "seq": "seven"})
  )

  assert [r.id for r in store.list_records()] == [good.id]
  assert "007-broken" in capsys.readouterr().err
  assert store.get_record("007") is None
  assert store.reindex() == 1
  fresh = store.new_record("fresh")
  assert fresh.id == "008"  # the broken directory keeps its id
  store.claim(fresh.id, slot="gpu0")
  assert store.reap_expired() == []


# Lease ownership — a reservation is not a process.


def test_a_manual_reservation_survives_its_claimer_exiting(store):
  """`rlmcp record claim` runs and exits; the GPU must stay reserved."""
  record = store.new_record("reserved")
  ctx = _mp()

  _run_all([ctx.Process(target=_claim_and_exit,
                        args=(str(store.root), record.id, ""))])
  store.reap_expired()
  lease = store.get_record(record.id).lease
  assert lease is not None and lease.owner == "manual"

  # The same claim made by the trainer process itself dies with it.
  store.release(record.id)
  _run_all([ctx.Process(target=_claim_and_exit,
                        args=(str(store.root), record.id, "pid-trainer"))])
  assert store.get_record(record.id).lease.owner == "runner"
  store.reap_expired()
  assert store.get_record(record.id).lease is None

  # A manual reservation still answers to its TTL.
  store.claim(record.id, slot="gpu0", ttl_seconds=60.0)
  held = store.get_record(record.id)
  held.lease.renewed_at -= 10_000
  store.put_record(held)
  store.reap_expired()
  assert store.get_record(record.id).lease is None


def test_a_reaped_run_is_interrupted_with_cause_and_still_closable(store):
  """The warn-path shape: started under a claim warning, then died silently.

  Death is not a result: the reaper parks the run at ``interrupted`` -- the
  one terminal verdict that stays re-closable -- with the cause in links, and
  the real close-out (with evidence) still lands afterwards.
  """
  silent = store.new_record("warn_path", verdict="running")
  fresh = store.new_record("fresh_start", verdict="running")

  meta = store.record_dir(silent) / "meta.json"
  payload = json.loads(meta.read_text())
  payload["updated_at"] = time.time() - 10_000
  meta.write_text(json.dumps(payload))

  reaped = store.reap_expired()

  assert [r.id for r in reaped] == [silent.id]
  loaded = store.get_record(silent.id)
  assert loaded.verdict == "interrupted"
  assert loaded.links["reaped"] == "stale-leaseless"
  assert store.get_record(fresh.id).verdict == "running"  # recency protects it

  # The run that died still owes its real close-out -- and can receive it.
  loaded.outcome = "died mid-run; the hypothesis was already dead at iter 300"
  loaded.metrics.append(["terrain_level_mean", "0.0 at 300"])
  assert check_verdict_change(loaded, "falsified", store.list_records()) is None
  loaded.verdict = "falsified"
  store.put_record(loaded)
  assert store.get_record(silent.id).verdict == "falsified"


# Sanitisation — nothing a record says can name a path outside the lab.


def test_path_traversal_is_refused_at_every_join(tmp_path, store):
  good = store.new_record("good")

  with pytest.raises(StoreError, match="Unsafe"):
    store.put_record(RunRecord(id="../../escaped", slug="evil"))
  with pytest.raises(StoreError, match="Unsafe"):
    store.put_record(RunRecord(id="099", slug="a/b"))
  with pytest.raises(StoreError, match="Unsafe"):
    store.get_record("../secrets")
  with pytest.raises(StoreError, match="Unsafe"):
    store.write_document(good.id, "../evil.md", "text")

  asset = tmp_path / "a.png"
  asset.write_bytes(b"x")
  with pytest.raises(StoreError, match="Unsafe"):
    store.media.put("..", str(asset))
  assert store.media.get("../index.sqlite") is None  # exists, but outside
  assert not store.media.exists("../runs")
  assert not (tmp_path / "escaped").exists()


# Durability.


def test_a_crashed_writers_temp_file_is_swept_on_open(tmp_path, capsys):
  store = FileStore(tmp_path / "records")
  record = store.new_record("survivor")
  dead = store.record_dir(record) / ".meta.json.999999999.abcdef.tmp"
  dead.write_text("half a write")
  live = store.record_dir(record) / f".meta.json.{os.getpid()}.abcdef.tmp"
  live.write_text("a write in flight")

  FileStore(tmp_path / "records")  # reopening sweeps

  assert not dead.exists()
  assert live.exists()  # its writer (this process) is still alive
  assert "swept" in capsys.readouterr().err
  live.unlink()


# Backward compatibility — the production lab must load with no migration.


_SMP_LAB = Path(__file__).resolve().parent.parent / "logs" / "smp_lab"


@pytest.mark.skipif(not _SMP_LAB.exists(), reason="production lab not present")
def test_the_production_lab_loads_unchanged(tmp_path):
  """Reading an old lab is pure: same records, version 0, nothing rewritten."""
  copy_root = tmp_path / "smp_lab"
  copy_root.mkdir()
  shutil.copytree(_SMP_LAB / "runs", copy_root / "runs")
  raw = {
      payload["id"]: payload
      for payload in (json.loads(p.read_text())
                      for p in (copy_root / "runs").glob("*/meta.json"))
  }

  store = FileStore(copy_root)
  loaded = store.list_records()

  assert [r.id for r in loaded] == sorted(raw)
  for record in loaded:
    assert record.version == 0
    assert record.to_dict() == {**raw[record.id], "version": 0}
  assert store.reindex() == len(raw)
  assert store.new_record("continuation").id == str(len(raw) + 1).zfill(3)
