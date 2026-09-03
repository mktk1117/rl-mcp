"""Turning a finished run into something that runs again.

A record says what was tried and what happened. What it does not say is *how to
do it again*, and the answer is spread across four places: the package the run
used (its code snapshot), the config it started from (folded down the parent
chain), the ladder it climbed, and the ad-hoc edits somebody made while it
trained. A recipe is those four, assembled into a directory.

The output is not a document about the run. It is a directory you can launch:
``curriculum.json`` loads into :class:`~rlmcp.core.curriculum.StageSchedule`
unchanged, ``package/`` is the code at the tree that ran, and ``launch.sh`` is
the command. Reproducing is running it.

**Distillation, not transcription.** An edit made at iteration 900 because the
entropy had collapsed does not belong in the replay as "wait 900 iterations,
then panic". It belongs as the value that rung starts with, carrying the reason
it was needed. Refused edits are left out of the ladder entirely -- they never
applied -- and said in the notes instead.

**And it cannot promise a policy.** RL is not bit-reproducible: GPU
nondeterminism, a different env count, a different seed. A recipe reproduces the
*procedure*, which is why ``expect.json`` names the numbers to check rather than
a hash to match. Claim "statistically equivalent", never "identical".
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

from rlmcp.core.curriculum import CurriculumStage, StageSchedule
from rlmcp.records.record import RunRecord, fold_recipe

DEFAULT_MIN_ITERATIONS = 100


def distil(interventions: list[dict[str, Any]],
           ladder: dict[str, dict[str, Any]] | None = None,
           entered: list[str] | None = None,
           entries: dict[str, int] | None = None,
           unplaced: list[dict[str, Any]] | None = None) -> StageSchedule:
  """The ladder that would have produced this run's history.

  Two shapes, because runs come in two kinds. A run that *had* a curriculum
  keeps its rungs and its promotion conditions -- those are real, they were
  written before the run -- in the order it actually climbed them, and the
  edits made while a rung was active fold into that rung's entry parameters,
  last value winning. A run with no curriculum gets one rung per group of
  edits, held for as long as the original ran before the next change.

  ``ladder``, ``entered`` and ``entries`` are what :mod:`rlmcp.core.replay`
  reads off a session: the stages as planned, the names in the order they were
  entered, and the iteration each was entered at. Only the rungs the run
  actually reached are in the recipe -- a rung it never climbed is a plan, not
  a result.

  An edit is folded into a rung only when that rung's window is known: its
  entry iteration and the next rung's. A ladder whose entry iterations were
  not logged gets its edits in ``unplaced`` (and in the notes) rather than
  smeared across every rung, because a five-rung ladder that starts every rung
  at the values of an edit made on rung three looks exactly like a recipe and
  is a different experiment.
  """
  edits = [i for i in interventions
           if i["kind"] == "set_parameter" and "(refused)" not in i["what"]]
  stage_changes = [i for i in interventions if i["kind"] == "curriculum_stage"]
  if unplaced is None:
    unplaced = []

  if ladder:
    names = [n for n in (entered or list(entries or {}) or list(ladder))
             if n in ladder]
    stages = [CurriculumStage.from_dict(ladder[name]) for name in names]
    if stages:
      schedule = StageSchedule(stages)
      entry = dict(entries) if entries else _entry_iterations(schedule, stage_changes)
      entry.setdefault(schedule.stages[0].name, 0)
      windows = _windows(schedule, entry)
      for edit in edits:
        key, value = _parsed(edit)
        if key is None:
          continue
        home = next((s for s, (start, end) in windows.items()
                     if start is not None and end is not None
                     and start <= edit["iteration"] < end), None)
        if home is None:
          unplaced.append(edit)
          continue
        stage = next(s for s in schedule.stages if s.name == home)
        stage.parameters[key] = value
        stage.notes = _append_note(stage.notes, edit)
      if unplaced:
        last = schedule.stages[-1]
        last.notes = _append_note(
            last.notes,
            {"iteration": unplaced[0]["iteration"],
             "what": (f"{len(unplaced)} edit(s) could not be placed in a rung: the "
                      "event log does not say when the rungs were entered. They "
                      "are listed in the recipe's README and NOT folded in."),
             "why": ""})
      return schedule

  stages: list[CurriculumStage] = []
  for index, edit in enumerate(edits):
    key, value = _parsed(edit)
    if key is None:
      continue
    following = edits[index + 1]["iteration"] if index + 1 < len(edits) else None
    held = (following - edit["iteration"]) if following else DEFAULT_MIN_ITERATIONS
    stages.append(CurriculumStage(
        name=f"{len(stages)}_{key.split('.')[-2] if '.' in key else key}",
        parameters={key: value},
        min_iterations=max(1, held),
        notes=_append_note("", edit),
    ))
  if not stages:
    stages = [CurriculumStage(name="0_as_launched",
                              notes="Nothing was changed mid-run.")]
  return StageSchedule(stages)


def _parsed(edit: dict[str, Any]) -> tuple:
  """``key`` and the value it was set to, read out of the phrased line."""
  what = edit["what"]
  if ": " not in what or " → " not in what:
    return None, None
  key, _, rest = what.partition(": ")
  _, _, new = rest.partition(" → ")
  try:
    return key, json.loads(new)
  except (TypeError, ValueError):
    return key, new


def _append_note(notes: str, edit: dict[str, Any]) -> str:
  line = f"it {edit['iteration']}: {edit['what']}"
  if edit.get("why"):
    line += f" — {edit['why']}"
  return f"{notes}\n{line}".strip()


def _entry_iterations(schedule: StageSchedule,
                      stage_changes: list[dict[str, Any]]) -> dict[str, int]:
  """Entry iterations read back off phrased stage lines, for callers with no
  event log. ``replay.stage_entries`` is the source when there is one."""
  entered = {schedule.stages[0].name: 0}
  for change in stage_changes:
    name = change["what"].split("→")[-1].strip()
    if name in {s.name for s in schedule.stages}:
      entered.setdefault(name, change["iteration"])
  return entered


def _windows(schedule: StageSchedule,
             entered: dict[str, int]) -> dict[str, tuple[int | None, int | None]]:
  """Each rung's ``[start, end)`` in iterations; None where the log is silent.

  A rung's window ends where the next rung starts. The last rung in the recipe
  is open-ended; a rung that was never entered (it is in the ladder but not in
  the log) has no window at all, so nothing folds into it -- and neither does
  anything fold into the rung before it, whose end is then unknown.
  """
  names = [s.name for s in schedule.stages]
  windows: dict[str, tuple[int | None, int | None]] = {}
  for index, name in enumerate(names):
    if name not in entered:
      windows[name] = (None, None)
      continue
    if index + 1 == len(names):
      windows[name] = (entered[name], 1 << 30)
      continue
    # The window closes where the next rung opens. If the log never says the
    # next rung was entered, this rung's end is unknown -- not infinite.
    following = names[index + 1]
    windows[name] = (entered[name], entered.get(following))
  return windows


def build(store: Any, record_id: str, out: Path | str,
          session_dir: str | None = None,
          policy: bool = True) -> dict[str, Any]:
  """Write a runnable recipe directory for ``record_id``.

  Every part is best-effort and says so: a run with no code snapshot still gets
  its config, its ladder and its warm-start chain, and the README names what is
  missing. A recipe that refused to exist because one input was absent would be
  a worse answer than a recipe that tells you what it could not find.

  ``policy=False`` leaves the trained weights out. They are the largest thing
  here by far, and a recipe meant only to be *read* does not need them.
  """
  record = store.get_record(record_id)
  if record is None:
    raise ValueError(f"No record '{record_id}'.")
  out = Path(out).expanduser().resolve()
  out.mkdir(parents=True, exist_ok=True)

  records = {r.id: r for r in store.list_records()}
  chain = phase_chain(record, records)
  session = Path(session_dir or record.session or "")
  interventions, ladder, entered, entries = _history(session)
  unplaced: list[dict[str, Any]] = []
  schedule = distil(interventions, ladder, entered, entries, unplaced)
  missing: list[str] = []
  if unplaced:
    missing.append(
        f"the rung for {len(unplaced)} mid-run edit(s) -- the event log does not "
        "say when the rungs were entered, so they are listed here and not "
        "folded into the ladder: "
        + "; ".join(f"it {e['iteration']}: {e['what']}" for e in unplaced))
  if not ladder and entered:
    # The rungs are named in the log but the ladder itself was never written
    # down, so the promotion conditions are gone. Say so: a recipe that quietly
    # replaced a five-rung curriculum with a list of ad-hoc edits would look
    # exactly like a correct one.
    missing.append(
        f"the promotion conditions for {len(entered)} curriculum rung(s) "
        f"({', '.join(entered)}) -- this run did not save its ladder")

  _write_json(out / "curriculum.json", schedule.to_dict())
  _write_json(out / "config.json", record.config or {})
  if not record.config:
    missing.append("the launch config (this run recorded none)")

  package = _restore_package(record, out / "package")
  if package is None:
    missing.append("the task package (this run recorded no code snapshot)")

  # The environment, materialised. `package/` is the repository at the tree the
  # run launched with, which is the right answer when you have that repository
  # and want to develop in it. `env/` is the complementary one: the terms as
  # they were actually running, weights included, inlined so it needs nothing
  # installed. A run that added a reward term mid-run has it only here.
  env = _write_env(session, out / "env")
  if not env.get("ok"):
    missing.append(f"the materialised environment ({env.get('error', 'not captured')})")

  weights = _copy_policy(session, out / "policy") if policy else None
  if policy and weights is None:
    missing.append("the trained policy (no checkpoint found for this run)")

  expectations = _expectations(record, session, weights)
  _write_json(out / "expect.json", expectations)
  (out / "phases.md").write_text(_phases(record, records))
  launch, launch_missing = _launch(record, session, package is not None,
                                   iterations=expectations.get("final_iteration"))
  missing += launch_missing
  (out / "launch.sh").write_text(launch)
  (out / "launch.sh").chmod(0o755)
  (out / "README.md").write_text(
      _readme(record, schedule, interventions, missing, env, weights))
  return {
      "record": record.id,
      "path": str(out),
      # The segments this recipe covers: a reader can check at a glance that a
      # branch point and a from-scratch restart were resolved the way they
      # expect, without opening phases.md.
      "phases": [node.id for node in chain],
      "stages": [s.name for s in schedule.stages],
      "interventions": len(interventions),
      "package": str(package) if package else "",
      "env": {k: env.get(k) for k in ("ok", "counts", "missing_source")},
      "policy": str(weights["path"]) if weights else "",
      "missing": missing,
  }


def _write_env(session: Path, destination: Path) -> dict[str, Any]:
  """The env config and its inlined implementations, from the run's capture."""
  if not session or not (session / "session.json").exists():
    return {"ok": False, "error": "this run has no session directory to read"}
  from rlmcp import env_export
  from rlmcp.session import Session

  try:
    return env_export.export_env(Session(session), destination)
  except Exception as exc:
    return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _copy_policy(session: Path, destination: Path) -> dict[str, Any] | None:
  """Copy the run's final checkpoint into the recipe.

  ``play.find_checkpoint`` resolves it, so a recipe and ``rlmcp play`` agree on
  which checkpoint a run "ended on" -- it picks by iteration rather than mtime,
  which matters for a run that had an older checkpoint copied back in for
  comparison.
  """
  import shutil

  from rlmcp.play import PlayError, checkpoint_iteration, find_checkpoint

  if not session:
    return None
  # The trainer's checkpoints sit in the run directory, one above the session;
  # `<session>/checkpoints/` holds the ones an agent saved by name mid-run.
  # Resolving from the session would stop at those -- `pre-smoothness-fix.pt`
  # is not what a run ended on -- so the run directory is asked first and the
  # session only when the run directory has nothing.
  checkpoint = None
  for candidate in (session.parent, session):
    try:
      found = find_checkpoint(candidate)
    except (PlayError, OSError):
      continue
    if checkpoint_iteration(found) >= 0 or checkpoint is None:
      checkpoint = found
    if checkpoint_iteration(checkpoint) >= 0:
      break
  if checkpoint is None:
    return None
  destination.mkdir(parents=True, exist_ok=True)
  target = destination / checkpoint.name
  try:
    shutil.copy2(checkpoint, target)
  except OSError:
    return None
  return {
      "path": target,
      "name": checkpoint.name,
      "source": str(checkpoint),
      "iteration": checkpoint_iteration(checkpoint),
      "size_mb": round(target.stat().st_size / 1e6, 2),
  }


DEFAULT_TOLERANCE = 0.2
"""How far a retrain may sit from the original and still count as reproduced.

20% of the original value. Loose on purpose: this is the band inside which two
RL runs of the same recipe differ from seed and GPU nondeterminism alone, and a
tighter default would fail honest reproductions and teach everyone to ignore
it. Tighten it per check when the metric deserves it."""


def verify(recipe_dir: Path | str, session_dir: Path | str,
           tolerance: float = DEFAULT_TOLERANCE) -> dict[str, Any]:
  """Did running this recipe get back to where the original run got?

  Compares the metrics the original run *claimed* -- ``expect.json``'s
  ``metrics``, the ones a human wrote into the record -- against the same
  metrics in a candidate run's telemetry.

  This answers "is it statistically equivalent", and cannot answer more than
  that. A pass is evidence the recipe reproduces the procedure; it is not proof
  the weights match, and nothing here compares weights.

  A metric the candidate never published is ``missing``, not a failure: it is
  usually a run that has not got far enough yet, and calling that a regression
  would be wrong.
  """
  recipe_dir = Path(recipe_dir).expanduser()
  try:
    expectations = json.loads((recipe_dir / "expect.json").read_text())
  except (OSError, ValueError) as exc:
    raise ValueError(
        f"No readable expect.json in {recipe_dir} ({exc}). Point this at a "
        "directory written by `rlmcp recipe build`.") from exc

  candidate, iteration = _final_metrics(Path(session_dir).expanduser())
  if not candidate:
    raise ValueError(
        f"No metrics in {session_dir}. Point this at the session of the run "
        "you launched from this recipe.")

  expected = dict(expectations.get("metrics") or {})
  if not expected:
    # Nothing was claimed, so nothing can be checked. The full telemetry is
    # still worth diffing by eye, and saying so beats inventing a verdict.
    expected = {}

  checks: list[dict[str, Any]] = []
  for name, raw in expected.items():
    # A record's metrics are typed by whoever wrote them, and `record close`
    # takes them as text -- "16.8" is the normal shape, not the exception.
    # Skipping non-floats without parsing would compare nothing and report a
    # confident "no claimed metrics to check".
    want = _as_number(raw)
    if want is None:
      checks.append({"metric": name, "expected": raw, "got": None,
                     "status": "not a number"})
      continue
    key = _telemetry_key(name, candidate)
    got = candidate.get(key) if key else None
    if got is None:
      checks.append({"metric": name, "expected": want, "got": None,
                     "status": "missing"})
      continue
    band = abs(want) * tolerance
    within = abs(got - want) <= band if band else got == want
    checks.append({
        "metric": name,
        "key": key,
        "expected": want,
        "got": got,
        "delta": got - want,
        "relative": (got - want) / want if want else None,
        "status": "within" if within else "outside",
    })

  compared = [c for c in checks
              if c["status"] in ("within", "outside")]
  outside = [c for c in compared if c["status"] == "outside"]
  return {
      "recipe": str(recipe_dir),
      "session": str(session_dir),
      "from_run": expectations.get("from_run"),
      "tolerance": tolerance,
      "iteration": iteration,
      "checks": checks,
      "compared": len(compared),
      "outside": len(outside),
      "missing": len([c for c in checks if c["status"] == "missing"]),
      # No claimed metrics means no verdict, which is not the same as a pass.
      "reproduced": bool(compared) and not outside,
      "verdict": (
          "no claimed metrics to check" if not compared
          else "statistically equivalent within the band" if not outside
          else f"{len(outside)} metric(s) outside the band"),
  }


def _history(session: Path) -> tuple:
  """The run's interventions, the ladder it planned, and the rungs it climbed.

  The ladder is read through :mod:`rlmcp.core.replay`, which already knows the
  two places a run writes it and that only the event log knows which rungs were
  actually entered. Re-deriving that here is how a recipe ends up describing a
  different run than ``rlmcp play`` does.
  """
  from rlmcp.core import replay
  from rlmcp.records.interventions import from_session

  if not session or not session.exists():
    return [], {}, [], {}
  try:
    interventions = from_session(session)
  except Exception:
    interventions = []
  try:
    entries = replay.stage_entries(session)
    return interventions, replay.read_ladder(session), list(entries), entries
  except Exception:
    return interventions, {}, [], {}


def _restore_package(record: RunRecord, destination: Path) -> Path | None:
  code = record.code or {}
  if code.get("kind") != "git" or not code.get("tree"):
    return None
  from rlmcp.records import snapshot

  try:
    return snapshot.restore(code["repo"], code["tree"], destination)
  except Exception:
    return None


def _expectations(record: RunRecord, session: Path | None = None,
                  weights: dict[str, Any] | None = None) -> dict[str, Any]:
  """What a replay is checked against. Numbers, never a hash.

  Two sets, and the difference matters. ``metrics`` are the ones a human wrote
  into the record: the claim the run was making, few and chosen. ``final`` are
  the telemetry this run actually ended on, every scalar it published. A
  retrain is judged against the first and can be *inspected* against the
  second, which is what makes "similar performance" a check rather than an
  impression.
  """
  expectations: dict[str, Any] = {
      "from_run": record.id,
      "verdict": record.verdict,
      "metrics": dict(record.metrics or []),
      "falsifier": record.falsifier.prose if record.falsifier else "",
      "note": ("RL is not bit-reproducible -- GPU nondeterminism, a different "
               "env count or seed all move the weights. Check these numbers as "
               "a band; claim 'statistically equivalent', never 'identical'."),
  }
  final, iteration = _final_metrics(session)
  if final:
    expectations["final"] = final
    expectations["final_iteration"] = iteration
  if weights:
    expectations["policy"] = {
        "file": weights["name"],
        "iteration": weights["iteration"],
        "size_mb": weights["size_mb"],
    }
  return expectations


def _size(size_mb: float) -> str:
  """A file size a reader can act on. A policy under 50 KB rounds to 0.0 MB."""
  if size_mb >= 0.1:
    return f"{size_mb} MB"
  return f"{round(size_mb * 1000):d} KB"


def _telemetry_key(name: str, candidate: dict[str, Any]) -> str | None:
  """The telemetry key a claimed metric name refers to.

  A record's metrics are named by a person -- ``joint_vel_rms`` -- while the
  telemetry publishes ``rlmcp/joint_vel_rms``, and a run made under another
  harness published ``mcplab/joint_vel_rms``. The exact key wins; otherwise
  the one key whose last path segment is the name. Two candidates is an
  ambiguity, not a match.
  """
  if name in candidate:
    return name
  tail = name.rsplit("/", 1)[-1]
  matches = [k for k in candidate if k.rsplit("/", 1)[-1] == tail]
  return matches[0] if len(matches) == 1 else None


def _as_number(value: Any) -> float | None:
  """A metric's value as a float, whether it was written as one or as text."""
  if isinstance(value, bool):
    return None
  if isinstance(value, (int, float)):
    return float(value)
  if isinstance(value, str):
    try:
      return float(value.strip())
    except ValueError:
      return None
  return None


def _final_metrics(session: Path | None) -> tuple:
  """The last telemetry row this run published, and the iteration it was at."""
  if not session or not (session / "session.json").exists():
    return {}, None
  from rlmcp.session import Session

  try:
    rows = Session(session).metrics(last_n=1)
  except Exception:
    return {}, None
  if not rows:
    return {}, None
  row = dict(rows[-1])
  iteration = row.pop("iteration", None)
  row.pop("t", None)
  numeric = {k: v for k, v in row.items()
             if isinstance(v, (int, float)) and not isinstance(v, bool)}
  return numeric, iteration


def phase_chain(record: RunRecord,
                records: dict[str, RunRecord]) -> list[RunRecord]:
  """The training segments this policy actually came through, oldest first.

  Walk the warm-start edges back and **stop at the last from-scratch run**,
  because that is where this policy began. Given

      1 -> 2 -> 3 (from scratch) -> 4 -> 5 -> 7

  the chain for 7 is ``3, 4, 5, 7``. Runs 1 and 2 are ancestors of 3's
  *config*, and their weights were thrown away when 3 restarted -- replaying
  them would be replaying history that this policy does not contain. Sibling
  branches (a 6 that also came off 5) are not on the path to 7 and are not
  here either.

  A warm start pointing at a record this store does not have ends the walk: an
  honest short chain beats a chain with a hole in it. Cycles end it too.
  """
  chain: list[RunRecord] = []
  current: RunRecord | None = record
  seen: set = set()
  while current is not None and current.id not in seen:
    seen.add(current.id)
    chain.append(current)
    weights = current.weights
    current = records.get(weights.run) if weights else None
  chain.reverse()
  return chain


def config_history(chain: list[RunRecord],
                   records: dict[str, RunRecord]) -> list[Any]:
  """The config changes worth reading, folded down to the target.

  The full fold walks the *parent* chain to the config root, which for a policy
  that restarted from scratch reaches back past the restart -- and the changes
  before it are already baked into the snapshot in ``config.json``. So the fold
  is cut at the run the training actually started from. What is left is the
  history that produced this policy, and nothing that did not.
  """
  if not chain:
    return []
  full = fold_recipe(chain[-1].id, records)
  ids = [rid for rid, _ in full]
  root = chain[0].id
  return full[ids.index(root):] if root in ids else full


def _phases(record: RunRecord, records: dict[str, RunRecord]) -> str:
  """The warm-start chain, flattened, and how to run each segment."""
  chain = phase_chain(record, records)

  lines = [f"# Phases for {record.display()}", ""]
  if len(chain) == 1:
    lines += ["This policy was trained from scratch: one phase, no warm start.",
              ""]
  else:
    lines += [
        (f"This policy came through {len(chain)} training segments. Each one "
        "starts from the weights of the one before it, so reproducing it means "
        "running them in order -- one launch per phase, each warm-started from "
        "the checkpoint the previous phase left."), "",
        ("Anything earlier than phase 1 is deliberately absent: the policy "
        "restarted from scratch there, so nothing before it survives in these "
        "weights. Its settings are already folded into `config.json`."), "",
    ]
  for index, node in enumerate(chain, start=1):
    start = node.weights.describe() if node.weights else "random init"
    lines += [f"## Phase {index}: {node.display()}", "",
              f"- starts from: {start}",
              f"- verdict: {node.verdict}",
              f"- what changed: {'; '.join(node.change) or 'not recorded'}"]
    if index == 1:
      lines += ["- launch: `./launch.sh <new-record-id>`", ""]
    else:
      previous = chain[index - 2]
      lines += [(f"- launch: `./launch.sh <new-record-id> --resume "
                f"<checkpoint from phase {index - 1} ({previous.id})>`"), ""]

  history = config_history(chain, records)
  lines += ["## Config recipe", "",
            (f"Folded from {chain[0].display() if chain else 'this run'} down to "
            f"{record.display()} -- the runs this policy came through."), ""]
  lines += [f"- **{rid}**: {'; '.join(changes) or 'no change recorded'}"
            for rid, changes in history]
  return "\n".join(lines) + "\n"


def _launch(record: RunRecord, session: Path, has_package: bool,
            iterations: int | None = None) -> tuple[str, list[str]]:
  """The command that runs this recipe, and what it could not fill in.

  Everything the recipe carries is on the command line: the task packages
  whose import registers the task, the launch config, the ladder, the seed,
  the env count and how long the original trained. A recipe whose ladder is
  "in curriculum.json, load it yourself" is a document; this is a launch.
  """
  info: dict[str, Any] = {}
  with contextlib.suppress(OSError, ValueError):
    info = json.loads((session / "session.json").read_text())
  missing: list[str] = []
  task = info.get("task") or record.task or "<task-id>"
  envs = info.get("num_envs")
  device = info.get("device") or "cuda:0"
  packages = [str(p) for p in (info.get("task_packages") or []) if p]
  seed = info.get("seed")
  if seed is None:
    seed = _seed_from_run_dir(session)

  lines = [
      "#!/usr/bin/env bash",
      "# Generated by `rlmcp recipe build`. Everything the recipe carries is on",
      "# this command line; check it before you run it, because the record knows",
      "# the task, the packages, the seed and the ladder, but not the flags",
      "# somebody typed around them. Run it from a checkout where the task",
      "# packages import (the recipe's package/ is on PYTHONPATH below).",
      "set -euo pipefail",
      "",
      'RECORD="${1:?pass the new record id: ./launch.sh 021}"',
      "shift",
      'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
  ]
  if has_package:
    lines.append('export PYTHONPATH="$HERE/package${PYTHONPATH:+:$PYTHONPATH}"')
  if packages:
    lines.append("TASK_PACKAGES=(" + " ".join(f'"{p}"' for p in packages) + ")")
  else:
    missing.append(
        "the task packages (this run did not record which modules register "
        "its task, so launch.sh takes them from $TASK_PACKAGES, "
        "space-separated, or fails to resolve the task)")
    lines += [
        "# This run did not record which modules register its task. Name them:",
        '#   TASK_PACKAGES="shand.tasks shand.rlmcp_ext" ./launch.sh 021',
        'read -r -a TASK_PACKAGES <<< "${TASK_PACKAGES:-}"',
    ]
  lines += [
      "",
      "rlmcp-train " + task + " \\",
      '    "${TASK_PACKAGES[@]/#/--task-package=}" \\',
      '    --config-json "$HERE/config.json" \\',
      '    --curriculum-json "$HERE/curriculum.json" \\',
  ]
  if envs:
    lines.append(f"    --num-envs {int(envs)} \\")
  if seed is not None:
    lines.append(f"    --seed {int(seed)} \\")
  else:
    missing.append("the seed (neither the session nor params/env.yaml records one)")
  if iterations:
    lines.append(f"    --max-iterations {int(iterations)} \\")
  lines += [
      f"    --device {device} \\",
      '    --record-run "$RECORD" \\',
      '    --code-root "' + ("$HERE/package" if has_package else ".") + '" \\',
      '    "$@"',
      "",
      "# Extra flags after the record id go straight to rlmcp-train: a",
      "# --records-root, a --run-name, --no-viser.",
  ]
  return "\n".join(lines) + "\n", missing


def _seed_from_run_dir(session: Path) -> int | None:
  """The seed from the run directory's saved env config, if it is there."""
  for name in ("agent.yaml", "env.yaml"):
    path = session.parent / "params" / name
    if not path.exists():
      continue
    with contextlib.suppress(OSError, ValueError):
      seed = json.loads(path.read_text()).get("seed")
      if isinstance(seed, int) and not isinstance(seed, bool):
        return seed
  return None


def _readme(record: RunRecord, schedule: StageSchedule,
            interventions: list[dict[str, Any]], missing: list[str],
            env: dict[str, Any] | None = None,
            weights: dict[str, Any] | None = None) -> str:
  env = env or {}
  lines = [
      f"# Recipe for {record.display()}", "",
      record.headline or record.one_line(), "",
      "## What is here", "",
      "| file | what it is |",
      "| --- | --- |",
      "| `package/` | the task package at the tree this run launched with |",
      ("| `env/` | the environment as it actually ran: config plus every term's "
      "implementation, inlined |"),
      "| `policy/` | the weights this run ended on |",
      "| `config.json` | the resolved parameters it started from |",
      "| `curriculum.json` | the ladder, loadable by `StageSchedule.from_dict` |",
      "| `launch.sh` | the command, as close as the record can say |",
      "| `phases.md` | the warm-start chain, flattened |",
      "| `expect.json` | the numbers a replay is checked against |",
      "",
      ("`package/` and `env/` are not duplicates. The package is the repository "
      "at the commit this run launched with — the thing to develop in. `env/` "
      "is the environment as it was *running*: final weights, and every term "
      "inlined as source, so it needs nothing installed. A reward term added "
      "by an agent mid-run exists only there."),
      "",
  ]
  if weights:
    lines += [
        f"The policy is `policy/{weights['name']}` "
        f"({_size(weights['size_mb'])}"
        + (f", iteration {weights['iteration']}"
           if weights["iteration"] >= 0 else "") + ").",
        "",
    ]
  counts = env.get("counts") or {}
  if counts:
    lines += [
        (f"The environment is {counts.get('rewards', 0)} reward, "
        f"{counts.get('observations', 0)} observation and "
        f"{counts.get('actions', 0)} action term(s) — see `env/README.md`."),
        "",
    ]
  lines += [
      "## Checking it worked", "",
      "```bash",
      "./launch.sh <new-record-id>          # train it again",
      "rlmcp recipe verify . --session <the new run's session>",
      "```",
      "",
      ("`verify` compares the new run against the metrics this one claimed, "
      "inside a band, because two RL runs of the same recipe differ by seed "
      "and GPU nondeterminism alone. It answers *statistically equivalent*, "
      "and nothing stronger — it does not compare weights."),
      "", "## The ladder", "",
  ]
  for stage in schedule.stages:
    conditions = "; ".join(
        f"{c.metric} {c.op} {c.value}" for c in stage.promote_when) or \
        f"after {stage.min_iterations} iterations"
    lines.append(f"- **{stage.name}** — promotes when {conditions}")
    if stage.parameters:
      lines.append(f"  - starts with: {stage.parameters}")
    if stage.notes:
      lines += [f"  - {line}" for line in stage.notes.splitlines()]
  lines += ["", f"Distilled from {len(interventions)} recorded intervention(s).",
            ""]
  if missing:
    lines += ["## What this recipe does not have", ""]
    lines += [f"- {item}" for item in missing]
    lines += [""]
  lines += [
      "## What it can and cannot promise", "",
      "It reproduces the **procedure**, not the policy. RL is not",
      "bit-reproducible, so check `expect.json` as a band of numbers rather",
      "than comparing weights, and claim 'statistically equivalent'.", "",
  ]
  return "\n".join(lines)


def _write_json(path: Path, payload: Any) -> None:
  path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
