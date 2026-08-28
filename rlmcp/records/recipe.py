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

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from rlmcp.core.curriculum import CurriculumStage, StageSchedule
from rlmcp.records.record import RunRecord, fold_recipe

DEFAULT_MIN_ITERATIONS = 100


def distil(interventions: List[Dict[str, Any]],
           ladder: Optional[Dict[str, Dict[str, Any]]] = None,
           entered: Optional[List[str]] = None) -> StageSchedule:
  """The ladder that would have produced this run's history.

  Two shapes, because runs come in two kinds. A run that *had* a curriculum
  keeps its rungs and its promotion conditions -- those are real, they were
  written before the run -- in the order it actually climbed them, and the
  edits made while a rung was active fold into that rung's entry parameters,
  last value winning. A run with no curriculum gets one rung per group of
  edits, held for as long as the original ran before the next change.

  ``ladder`` and ``entered`` are what :mod:`rlmcp.core.replay` reads off a
  session: the stages as planned, and the names in the order they were entered.
  Only the rungs the run actually reached are in the recipe -- a rung it never
  climbed is a plan, not a result.
  """
  edits = [i for i in interventions
           if i["kind"] == "set_parameter" and "(refused)" not in i["what"]]
  stage_changes = [i for i in interventions if i["kind"] == "curriculum_stage"]

  if ladder:
    names = [n for n in (entered or list(ladder)) if n in ladder]
    stages = [CurriculumStage.from_dict(ladder[name]) for name in names]
    if stages:
      schedule = StageSchedule(stages)
      entry = _entry_iterations(schedule, stage_changes)
      for stage in schedule.stages:
        start = entry.get(stage.name, 0)
        end = _next_entry(entry, schedule, stage.name)
        for edit in edits:
          if start <= edit["iteration"] < end:
            key, value = _parsed(edit)
            if key is not None:
              stage.parameters[key] = value
              stage.notes = _append_note(stage.notes, edit)
      return schedule

  stages: List[CurriculumStage] = []
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


def _parsed(edit: Dict[str, Any]) -> tuple:
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


def _append_note(notes: str, edit: Dict[str, Any]) -> str:
  line = f"it {edit['iteration']}: {edit['what']}"
  if edit.get("why"):
    line += f" — {edit['why']}"
  return f"{notes}\n{line}".strip()


def _entry_iterations(schedule: StageSchedule,
                      stage_changes: List[Dict[str, Any]]) -> Dict[str, int]:
  entered = {schedule.stages[0].name: 0}
  for change in stage_changes:
    name = change["what"].split("→")[-1].strip()
    entered.setdefault(name, change["iteration"])
  return entered


def _next_entry(entered: Dict[str, int], schedule: StageSchedule,
                name: str) -> int:
  names = [s.name for s in schedule.stages]
  index = names.index(name)
  for later in names[index + 1:]:
    if later in entered:
      return entered[later]
  return 1 << 30


def build(store: Any, record_id: str, out: Path | str,
          session_dir: Optional[str] = None,
          policy: bool = True) -> Dict[str, Any]:
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
  interventions, ladder, entered = _history(session)
  schedule = distil(interventions, ladder, entered)
  missing: List[str] = []
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

  _write_json(out / "expect.json", _expectations(record, session, weights))
  (out / "phases.md").write_text(_phases(record, records))
  (out / "launch.sh").write_text(_launch(record, session, package is not None))
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


def _write_env(session: Path, destination: Path) -> Dict[str, Any]:
  """The env config and its inlined implementations, from the run's capture."""
  if not session or not (session / "session.json").exists():
    return {"ok": False, "error": "this run has no session directory to read"}
  from rlmcp import env_export
  from rlmcp.session import Session

  try:
    return env_export.export_env(Session(session), destination)
  except Exception as exc:  # noqa: BLE001 -- never fail the whole recipe.
    return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _copy_policy(session: Path, destination: Path) -> Optional[Dict[str, Any]]:
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
  try:
    checkpoint = find_checkpoint(session)
  except (PlayError, OSError):
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
           tolerance: float = DEFAULT_TOLERANCE) -> Dict[str, Any]:
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

  checks: List[Dict[str, Any]] = []
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
    got = candidate.get(name)
    if got is None:
      checks.append({"metric": name, "expected": want, "got": None,
                     "status": "missing"})
      continue
    band = abs(want) * tolerance
    within = abs(got - want) <= band if band else got == want
    checks.append({
        "metric": name,
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
    return [], {}, []
  try:
    interventions = from_session(session)
  except Exception:  # noqa: BLE001 -- an unreadable log costs the notes, not the recipe.
    interventions = []
  try:
    return interventions, replay.read_ladder(session), replay.stage_names(session)
  except Exception:  # noqa: BLE001 -- same rule: the recipe still gets written.
    return interventions, {}, []


def _restore_package(record: RunRecord, destination: Path) -> Optional[Path]:
  code = record.code or {}
  if code.get("kind") != "git" or not code.get("tree"):
    return None
  from rlmcp.records import snapshot

  try:
    return snapshot.restore(code["repo"], code["tree"], destination)
  except Exception:  # noqa: BLE001 -- a pruned tree must not fail the recipe.
    return None


def _expectations(record: RunRecord, session: Optional[Path] = None,
                  weights: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
  """What a replay is checked against. Numbers, never a hash.

  Two sets, and the difference matters. ``metrics`` are the ones a human wrote
  into the record: the claim the run was making, few and chosen. ``final`` are
  the telemetry this run actually ended on, every scalar it published. A
  retrain is judged against the first and can be *inspected* against the
  second, which is what makes "similar performance" a check rather than an
  impression.
  """
  expectations: Dict[str, Any] = {
      "from_run": record.id,
      "verdict": record.verdict,
      "metrics": {name: value for name, value in (record.metrics or [])},
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


def _as_number(value: Any) -> Optional[float]:
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


def _final_metrics(session: Optional[Path]) -> tuple:
  """The last telemetry row this run published, and the iteration it was at."""
  if not session or not (session / "session.json").exists():
    return {}, None
  from rlmcp.session import Session

  try:
    rows = Session(session).metrics(last_n=1)
  except Exception:  # noqa: BLE001 -- an unreadable log costs the band, not the recipe.
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
                records: Dict[str, RunRecord]) -> List[RunRecord]:
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
  chain: List[RunRecord] = []
  current: Optional[RunRecord] = record
  seen: set = set()
  while current is not None and current.id not in seen:
    seen.add(current.id)
    chain.append(current)
    weights = current.weights
    current = records.get(weights.run) if weights else None
  chain.reverse()
  return chain


def config_history(chain: List[RunRecord],
                   records: Dict[str, RunRecord]) -> List[Any]:
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


def _phases(record: RunRecord, records: Dict[str, RunRecord]) -> str:
  """The warm-start chain, flattened, and how to run each segment."""
  chain = phase_chain(record, records)

  lines = [f"# Phases for {record.display()}", ""]
  if len(chain) == 1:
    lines += ["This policy was trained from scratch: one phase, no warm start.",
              ""]
  else:
    lines += [
        f"This policy came through {len(chain)} training segments. Each one "
        "starts from the weights of the one before it, so reproducing it means "
        "running them in order -- one launch per phase, each warm-started from "
        "the checkpoint the previous phase left.", "",
        "Anything earlier than phase 1 is deliberately absent: the policy "
        "restarted from scratch there, so nothing before it survives in these "
        "weights. Its settings are already folded into `config.json`.", "",
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
      lines += [f"- launch: `./launch.sh <new-record-id> --resume "
                f"<checkpoint from phase {index - 1} ({previous.id})>`", ""]

  history = config_history(chain, records)
  lines += ["## Config recipe", "",
            f"Folded from {chain[0].display() if chain else 'this run'} down to "
            f"{record.display()} -- the runs this policy came through.", ""]
  lines += [f"- **{rid}**: {'; '.join(changes) or 'no change recorded'}"
            for rid, changes in history]
  return "\n".join(lines) + "\n"


def _launch(record: RunRecord, session: Path, has_package: bool) -> str:
  """The command, as close to the original as the record can say."""
  info: Dict[str, Any] = {}
  try:
    info = json.loads((session / "session.json").read_text())
  except (OSError, ValueError):
    pass
  task = info.get("task") or record.task or "<task-id>"
  envs = info.get("num_envs")
  device = info.get("device") or "cuda:0"
  lines = [
      "#!/usr/bin/env bash",
      "# Generated by `rlmcp recipe build`. Check it before you run it: the",
      "# record knows the task, the env count and the device, but not the",
      "# flags somebody typed around them.",
      "set -euo pipefail",
      "",
      'RECORD="${1:?pass the new record id: ./launch.sh 021}"',
      "",
      "rlmcp-train " + task + " \\",
  ]
  if envs:
    lines.append(f"    --num-envs {int(envs)} \\")
  lines += [
      f"    --device {device} \\",
      '    --record-run "$RECORD" \\',
      "    --code-root " + ("./package" if has_package else "."),
      "",
      "# The ladder is curriculum.json; load it with StageSchedule.from_dict",
      "# and pass it to rlmcp.wrap(curriculum=...) from your own launcher.",
  ]
  return "\n".join(lines) + "\n"


def _readme(record: RunRecord, schedule: StageSchedule,
            interventions: List[Dict[str, Any]], missing: List[str],
            env: Optional[Dict[str, Any]] = None,
            weights: Optional[Dict[str, Any]] = None) -> str:
  env = env or {}
  lines = [
      f"# Recipe for {record.display()}", "",
      record.headline or record.one_line(), "",
      "## What is here", "",
      "| file | what it is |",
      "| --- | --- |",
      "| `package/` | the task package at the tree this run launched with |",
      "| `env/` | the environment as it actually ran: config plus every term's "
      "implementation, inlined |",
      "| `policy/` | the weights this run ended on |",
      "| `config.json` | the resolved parameters it started from |",
      "| `curriculum.json` | the ladder, loadable by `StageSchedule.from_dict` |",
      "| `launch.sh` | the command, as close as the record can say |",
      "| `phases.md` | the warm-start chain, flattened |",
      "| `expect.json` | the numbers a replay is checked against |",
      "",
      "`package/` and `env/` are not duplicates. The package is the repository "
      "at the commit this run launched with — the thing to develop in. `env/` "
      "is the environment as it was *running*: final weights, and every term "
      "inlined as source, so it needs nothing installed. A reward term added "
      "by an agent mid-run exists only there.",
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
        f"The environment is {counts.get('rewards', 0)} reward, "
        f"{counts.get('observations', 0)} observation and "
        f"{counts.get('actions', 0)} action term(s) — see `env/README.md`.",
        "",
    ]
  lines += [
      "## Checking it worked", "",
      "```bash",
      "./launch.sh <new-record-id>          # train it again",
      "rlmcp recipe verify . --session <the new run's session>",
      "```",
      "",
      "`verify` compares the new run against the metrics this one claimed, "
      "inside a band, because two RL runs of the same recipe differ by seed "
      "and GPU nondeterminism alone. It answers *statistically equivalent*, "
      "and nothing stronger — it does not compare weights.",
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
