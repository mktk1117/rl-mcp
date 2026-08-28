"""``rlmcp`` command line: talk to a live training session from a shell.

Everything the MCP server exposes is reachable here too, which makes the whole
system debuggable without an LLM in the loop::

    rlmcp sessions
    rlmcp status
    rlmcp params --contains foot
    rlmcp set reward.foot_slip.weight -0.3 --why "feet skating on slopes"
    rlmcp diagnose --seconds 4 --where terrain=pyramid_stairs
    rlmcp reset-envs --where terrain=pyramid_stairs   # fresh episodes, there only
    rlmcp commands                       # what this run accepts, extensions included
    rlmcp run set_terrain terrains='["flat","random_rough"]' max_level=4
    rlmcp curriculum advance

Output follows the reader. A pipe gets the JSON it has always got, because an
agent's parser is a contract; a terminal gets the same payload formatted, and
any picture the command produced is opened. ``--json`` / ``--text`` and
``RLMCP_OUTPUT`` override the guess, ``--no-open`` / ``RLMCP_OPEN`` the
showing. ``RLMCP_SESSION`` pins a session and ``RLMCP_ROOT`` points the
search, the same two variables the MCP server reads.

Depends only on the standard library, so it runs in any interpreter -- the
simulator lives in the training process, not here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rlmcp import cli_output
from rlmcp.records.record import FEEDBACK_KINDS
from rlmcp.session import Session, iter_sessions

DEFAULT_ROOTS = ("./logs", "./rlmcp_session", ".")


def _search_roots(args: argparse.Namespace) -> Tuple[List[str], str]:
  """The roots a bare command searches, and where that list came from.

  Precedence matches the MCP server's: an explicit ``--root`` wins, then
  ``$RLMCP_ROOT``, then the cwd-relative defaults. The second element names
  the source, so an error can say "looked in X (from RLMCP_ROOT)" instead of
  leaving the reader to guess why X was searched.
  """
  if getattr(args, "root", None):
    return [args.root], "--root"
  env_root = os.environ.get("RLMCP_ROOT")
  if env_root:
    return [env_root], "RLMCP_ROOT"
  return list(DEFAULT_ROOTS), "default"


def _resolve_session(args: argparse.Namespace) -> Session:
  if getattr(args, "session", None):
    return _remember(Session.open(args.session))
  env_dir = os.environ.get("RLMCP_SESSION")
  if env_dir:
    return _remember(Session.open(env_dir))
  roots, origin = _search_roots(args)
  for root in roots:
    found = Session.find_latest(root)
    if found is not None:
      return _remember(found)
  if origin == "default":
    # Nothing under the cwd. Before refusing, ask the registry: trainers,
    # servers and past resolutions announce themselves there precisely so a
    # bare command works from any directory. An explicit --root or
    # RLMCP_ROOT is never widened this way -- a scoped question answered
    # from outside its scope is how a reader ends up on the wrong run.
    found, how = _registry_fallback()
    if found is not None:
      print(
          cli_output.note(
              f"[rlmcp] no session under {Path.cwd()}; "
              f"using {found.dir} ({how})"
          ),
          file=sys.stderr,
      )
      return _remember(found)
  _refuse_no_session(roots, origin)
  raise AssertionError("unreachable")  # _refuse_no_session always exits.


def _remember(session: Session) -> Session:
  """File the resolved session in the registry; recent runs stay findable."""
  from rlmcp import registry

  registry.register(registry.KIND_SEEN, session_dir=session.dir,
                    session_kind=session.info().get("kind"))
  return session


def _registry_fallback() -> Tuple[Optional[Session], str]:
  """The newest session the registry can still vouch for, with provenance.

  Live registrants outrank dead ones -- "what is running?" is the question a
  bare command asks -- and within each class newest registration wins. Play
  sessions are skipped, the same way :meth:`Session.find_latest` skips them.
  """
  from rlmcp import registry
  from rlmcp.session import PLAY_SESSION_KIND

  rows = registry.entries()
  for live in (True, False):
    for row in rows:
      if row["pid_alive"] is not live:
        continue
      if row.get("session_kind") == PLAY_SESSION_KIND:
        continue
      state = "running" if live else "now gone"
      who = {
          registry.KIND_TRAINER:
              f"registered by its trainer, pid {row.get('pid')}, {state}",
          registry.KIND_SERVER:
              f"registered by an rlmcp-server, pid {row.get('pid')}, {state}",
          registry.KIND_SEEN: "the run last looked at from this machine",
      }.get(row.get("kind"), f"registered by {row.get('kind')}")
      if row.get("session_exists"):
        try:
          found = Session.open(row["session_dir"])
        except FileNotFoundError:
          continue
        return found, who
      if row.get("root_exists"):
        found = Session.find_latest(row["root"])
        if found is not None:
          if row.get("kind") == registry.KIND_SERVER:
            return found, (
                f"newest under '{row['root']}', which an rlmcp-server "
                f"(pid {row.get('pid')}, {state}) is watching"
            )
          return found, f"newest under '{row['root']}', from the registry"
  return None, ""


def _refuse_no_session(roots: List[str], origin: str) -> None:
  """Exit with a report: where the search went, and what would point it right.

  In JSON mode the refusal is a payload on stdout -- an agent that captured
  stdout must get something to parse, the same ``ok: false`` envelope every
  other failure wears -- and the exit code stays 1 either way.
  """
  from rlmcp import registry

  if origin == "default":
    looked = (
        f"under the current directory ({Path.cwd()}) -- its ./logs, "
        "./rlmcp_session, or anywhere beneath it"
    )
  else:
    looked = f"under '{roots[0]}' (from {origin})"

  rows = registry.entries()
  servers = [r for r in rows if r["kind"] == registry.KIND_SERVER and r["pid_alive"]]
  if servers:
    watching = servers[0].get("session_dir") or servers[0].get("root")
    known = (
        f"An rlmcp-server (pid {servers[0]['pid']}) is running, watching "
        f"'{watching}', but no session was found there either"
    )
  elif not rows:
    known = (
        "The registry is empty -- trainers and MCP servers announce "
        "themselves when they start, and none has yet"
    )
  elif origin != "default":
    known = (
        f"The registry knows {len(rows)} session(s) elsewhere; run plain "
        "`rlmcp sessions` (no --root) to see them"
    )
  else:
    known = (
        f"Nothing in the registry still points at a readable session "
        f"({len(rows)} stale entries)"
    )
  hint = (
      "A session is the <run_dir>/rlmcp directory a training run writes. "
      "Point at one with --session <dir> or --root <logs dir>, or set "
      "RLMCP_SESSION / RLMCP_ROOT."
  )
  if _MODE != "text":
    _emit({"ok": False, "error": "No rlmcp session found.",
           "looked": looked, "registry": known, "hint": hint})
    raise SystemExit(1)
  raise SystemExit(f"No rlmcp session found {looked}.\n{known}.\n{hint}")


# Resolved once in main(), read by _emit. Module state rather than a threaded
# parameter because _emit has ~45 call sites across four helpers, and this is a
# single-shot process that exits before anything could race on it.
_MODE = "json"
_OPEN = "auto"


def _emit(payload: Any, pretty: bool = True, *, command: Optional[str] = None) -> None:
  """The one place anything reaches stdout.

  ``command`` is the *trainer* command name, so ``rlmcp shot`` and
  ``rlmcp run screenshot`` render through the same formatter; it is ignored
  in JSON mode, where the bytes must not depend on how the call was spelled.
  """
  if _MODE != "text":
    print(json.dumps(payload, indent=2 if pretty else None, default=str))
    return
  print(cli_output.render(payload, command))
  shown, held = cli_output.show_artifacts(payload, _OPEN)
  for path, how in shown:
    if how != "inline":  # An inline draw is its own evidence.
      print(cli_output.note(f"-> opened {path} ({how})"))
  if held and _OPEN != "never":
    print(cli_output.note(f"-> {len(held)} artifact(s) not opened; --open to show"))


def _call(session: Session, cmd: str, timeout: float, **args: Any) -> int:
  """Send a command and print the response; returns a process exit code."""
  live = session.liveness_info()
  if live["state"] == "dead":
    _emit(
        {
            "ok": False,
            "error": "The training process for this session is not running.",
            "session": str(session.dir),
            **({"note": live["note"]} if "note" in live else {}),
            "hint": "Read status.json / metrics.jsonl for the final state.",
        },
        command=cmd,
    )
    return 1
  response = session.call(cmd, timeout=timeout, **args)
  _emit({"ok": response.ok, "result": response.result, "error": response.error},
        command=cmd)
  return 0 if response.ok else 1


def _parse_value(text: str) -> Any:
  """Accept JSON where possible so ranges like ``[-1,2]`` work, else a string."""
  try:
    return json.loads(text)
  except json.JSONDecodeError:
    return text


def _kv_pairs(items: Optional[List[str]]) -> Dict[str, Any]:
  out: Dict[str, Any] = {}
  for item in items or []:
    if "=" not in item:
      raise SystemExit(f"Expected key=value, got '{item}'")
    key, value = item.split("=", 1)
    out[key] = _parse_value(value)
  return out


def _default_metric_names(session: Session, limit: int = 4) -> List[str]:
  """A sensible default selection, drawn from what this run actually recorded.

  Naming metrics up front would bake one task's vocabulary into the CLI -- a
  cartpole run has no terrain level and no velocity command.
  """
  rows = session.metrics(last_n=1)
  available = {k for row in rows for k in row if k not in ("iteration", "t")}
  chosen = [
      k for k in ("Train/mean_reward", "Train/mean_episode_length") if k in available
  ]
  for key in sorted(k for k in available if k.startswith("rlmcp/")):
    if len(chosen) >= limit:
      break
    chosen.append(key)
  return chosen or sorted(available)[:limit]


def _offline_series(
    session: Session, names: List[str], last_n: Optional[int] = None
) -> Dict[str, List[List[float]]]:
  """Rebuild metric series from metrics.jsonl, for runs that have finished.

  ``last_n`` bounds the file read to its tail: rows are one per iteration, so
  the last N rows hold the last N points of every still-recorded metric.
  """
  rows = session.metrics(last_n=last_n)
  series: Dict[str, List[List[float]]] = {name: [] for name in names}
  for row in rows:
    iteration = row.get("iteration")
    for name in names:
      value = row.get(name)
      if isinstance(value, (int, float)) and iteration is not None:
        series[name].append([iteration, float(value)])
  if last_n:
    series = {k: v[-last_n:] for k, v in series.items()}
  return series


def _offline_plot(
    session: Session, names: List[str], last_n: int, smooth: int
) -> int:
  """Plot a finished run's metrics without talking to a training process."""
  try:
    from rlmcp.core.telemetry.plotter import plot_metric_series
  except ImportError as exc:
    _emit(
        {
            "ok": False,
            "error": f"Offline plotting needs matplotlib in this interpreter ({exc}).",
            "hint": "Run the CLI from the environment that has rlmcp's plotting extras.",
        }
    )
    return 1

  series = _offline_series(session, names, last_n=last_n)
  if not any(series.values()):
    available = sorted(
        {k for row in session.metrics(last_n=1) for k in row if k not in ("iteration", "t")}
    )
    _emit({"ok": False, "error": f"No data for {names}.", "available": available[:40]})
    return 1

  # Stage markers come from the event log's tail; 2000 events reach far past
  # any plotted window without re-reading a day of audit trail.
  markers = [
      (float(e["iteration"]), str(e.get("to", "")))
      for e in session.events(last_n=2000)
      if e.get("kind") == "curriculum_stage" and isinstance(e.get("iteration"), (int, float))
  ]
  png = plot_metric_series(
      {k: [tuple(p) for p in v] for k, v in series.items()},
      title=f"{session.dir.parent.name} (offline)",
      smooth_window=max(1, smooth),
      markers=markers,
  )
  path = session.artifact_path("metrics_offline.png")
  path.write_bytes(png)
  _emit({"ok": True, "result": {"image_path": str(path), "metrics": names,
                                "source": "metrics.jsonl", "live": False}})
  return 0


def _analyze_offline(
    trace_path: str, plot: bool = False, allow_legacy: bool = False
) -> int:
  """Re-run the analysis on a saved trace, e.g. after the trainer has exited."""
  try:
    from rlmcp.core.diagnostics import analyze_trace
    from rlmcp.core.telemetry.trace import load_npz
  except ImportError as exc:
    _emit({"ok": False, "error": f"Offline analysis needs numpy in this interpreter ({exc})."})
    return 1

  path = Path(trace_path)
  if not path.exists():
    _emit({"ok": False, "error": f"No trace at '{path}'."})
    return 1

  if allow_legacy:
    print(
        f"[rlmcp] --allow-legacy: unpickling '{path}' -- only do this for a "
        "trace whose origin you trust.",
        file=sys.stderr,
    )
  try:
    loaded = load_npz(path, allow_legacy=allow_legacy)
  except ValueError as exc:
    # The pickle refusal for legacy traces is by design; surface it as a
    # result rather than a traceback.
    _emit({
        "ok": False,
        "error": str(exc),
        "hint": "Re-run with --allow-legacy if you trust where this trace came from.",
    })
    return 1
  data, labels, meta = loaded["data"], loaded["labels"], loaded["meta"]
  time = data.get("time")
  dt = float(time[1] - time[0]) if time is not None and len(time) > 1 else 0.02
  report = analyze_trace(data, labels, dt=dt)

  result = {"trace_path": str(path), "meta": meta, "report": report}
  if plot:
    from rlmcp.core.telemetry.plotter import plot_trace

    image = path.with_name(path.stem + "_offline.png")
    image.write_bytes(plot_trace(data, labels, title=path.stem))
    result["image_path"] = str(image)
  _emit({"ok": True, "result": result})
  return 0


def _record_command(args: argparse.Namespace) -> int:
  """The records subcommands. No live trainer needed for any of them."""
  from rlmcp.records import open_store
  from rlmcp.records.record import Falsifier, Feedback, Weights, fold_recipe
  from rlmcp.records.store import StoreError
  from rlmcp.records.validate import check_verdict_change, validate

  store = open_store(getattr(args, "records_root", None), slots=args.slots)
  action = args.record_command

  if action == "new":
    record = store.new_record(
        args.slug,
        stage=args.stage,
        hypothesis=args.hypothesis,
        prediction=args.prediction,
        falsifier=Falsifier(
            prose=args.falsifier,
            conditions=_parse_conditions(args.falsify_when),
            check_after=args.check_after,
        ),
        change=list(args.change or []),
        task=args.task.strip(),
        headline=args.headline,
        parent=args.parent,
        weights=Weights(args.weights, args.checkpoint) if args.weights else None,
        proposed_by=args.proposed_by,
    )
    from rlmcp.records.report import write_plan

    plan = write_plan(store, record)
    payload = {"ok": True, "record": record.summary(), "plan": str(plan)}
    if record.falsifier.is_empty():
      payload["warning"] = (
          "No falsifier. If you cannot name an observation that would prove this "
          "wrong, you are not running an experiment."
      )
    _emit(payload)
    return 0

  if action == "list":
    found = store.query(stage=args.stage, verdict=args.verdict,
                        parent=args.parent, text=args.text)
    _emit({"count": len(found), "records": [r.summary() for r in found]},
          command="lab_list")
    return 0

  if action == "show":
    record = store.get_record(args.record_id)
    if record is None:
      _emit({"ok": False, "error": f"No record '{args.record_id}'."})
      return 1
    records = {r.id: r for r in store.list_records()}
    try:
      recipe = [{"id": rid, "change": ch} for rid, ch in fold_recipe(record.id, records)]
    except ValueError as exc:
      recipe = [{"error": str(exc)}]
    _emit({"ok": True, "record": record.to_dict(), "recipe": recipe})
    return 0

  if action == "close":
    record = store.get_record(args.record_id)
    if record is None:
      _emit({"ok": False, "error": f"No record '{args.record_id}'."})
      return 1

    from rlmcp.records.report import gather_evidence, write_report

    # The whole close-out is assembled inside the store's read-modify-write
    # helper: the rules judge the record as it would be persisted (re-judged
    # on a fresh read if another writer interleaves), a refusal aborts before
    # any write and leaves the file untouched, and a lost compare-and-swap is
    # retried instead of surfacing as an error.
    box: Dict[str, Any] = {}

    def close_out(fresh) -> Any:
      if args.outcome:
        fresh.outcome = args.outcome
      if args.headline:
        fresh.headline = args.headline
      fresh.metrics.extend(_parse_metrics(args.metric))

      # The falsifier is read at the run's final iteration, never before its
      # pre-registered read-point: a run killed at iteration 50 has not fired
      # a falsifier that says "do not read me before 600".
      evidence = gather_evidence(fresh.session)
      # Feedback said to the running trainer lives in the session's event log,
      # which is deleted with the logs. Fold it into the record now, skipping
      # anything already attached -- close-out is re-runnable on an
      # ``interrupted`` run, and the ledger must not double.
      seen = {(f.author, f.text) for f in fresh.feedback}
      for said in evidence.get("feedback") or []:
        if (said.get("author", "user"), said.get("text", "")) in seen:
          continue
        fresh.feedback.append(Feedback(
            text=said.get("text", ""),
            kind=said.get("kind", "steer"),
            author=said.get("author", "user"),
            at=float(said.get("at") or time.time()),
            iteration=said.get("iteration"),
            interpretation=said.get("interpretation", ""),
            response=said.get("response", ""),
            changed=bool(said.get("response")),
        ))

      result = fresh.falsifier.check(
          evidence.get("final_metrics") or {}, iteration=evidence.get("iterations")
      )
      result["evaluated_at"] = evidence.get("iterations")
      result["metrics_window"] = evidence.get("metrics_window", 0)
      if fresh.falsifier.conditions:
        fresh.metrics.append(["falsifier", _falsifier_row(result)])

      refusal = check_verdict_change(fresh, args.verdict, store.list_records())
      if refusal:
        box["refusal"] = refusal
        return False  # Abort without writing.
      box["evidence"], box["result"] = evidence, result

      fresh.verdict = args.verdict
      fresh.lease = None

    try:
      closed = store.update_record(record.id, close_out)
    except StoreError as exc:
      _emit({"ok": False, "error": str(exc)})
      return 1
    if closed is None:
      _emit({"ok": False, "error": box["refusal"]})
      return 1
    report = write_report(store, closed, box["result"], evidence=box["evidence"])
    _emit({"ok": True, "record": closed.summary(), "falsifier": box["result"],
           "report": str(report)})
    return 0

  if action == "headline":
    text = (args.text or "").strip()

    def set_headline(fresh) -> None:
      fresh.headline = text

    try:
      updated = store.update_record(args.record_id, set_headline)
    except StoreError as exc:
      _emit({"ok": False, "error": str(exc)})
      return 1
    if updated is None:
      _emit({"ok": False, "error": f"No record '{args.record_id}'."})
      return 1
    _emit({"ok": True, "record": updated.id, "headline": updated.one_line(),
           "derived": not updated.headline})
    return 0

  if action == "feedback":
    record = store.get_record(args.record_id)
    if record is None:
      _emit({"ok": False, "error": f"No record '{args.record_id}'."})
      return 1
    iteration = args.iteration
    if iteration is None and record.session:
      # The iteration is the one thing only the session knows, and it is what
      # places the remark on the run's timeline. Missing is fine -- feedback on
      # a finished write-up genuinely has no iteration -- but guessing is not.
      try:
        iteration = Session.open(record.session).status().get("iteration")
      except (FileNotFoundError, OSError):
        iteration = None
    entry = Feedback(
        text=args.text,
        kind=args.kind,
        author=args.author,
        iteration=iteration,
        interpretation=args.interpretation,
        response=args.response,
        changed=bool(args.response) and not args.no_change,
        affects=list(args.affects or []),
        artifacts=list(args.artifacts or []),
    )
    try:
      updated = store.add_feedback(record.id, entry)
    except StoreError as exc:
      _emit({"ok": False, "error": str(exc)})
      return 1
    index = len(updated.feedback) - 1
    payload = {"ok": True, "record": updated.id, "index": index,
               "feedback": updated.feedback[index].to_dict()}
    if updated.feedback[index].outstanding:
      payload["reminder"] = (
          f"Unanswered until you record what you did: "
          f"rlmcp record answer {updated.id} {index} \"...\""
      )
    _emit(payload)
    return 0

  if action == "answer":
    try:
      updated = store.answer_feedback(
          args.record_id, args.index, args.response, changed=not args.no_change)
    except StoreError as exc:
      _emit({"ok": False, "error": str(exc)})
      return 1
    _emit({"ok": True, "record": updated.id, "index": args.index,
           "feedback": updated.feedback[args.index].to_dict()})
    return 0

  if action == "timeline":
    rows = store.feedback_timeline(kind=args.kind, author=args.author,
                                   outstanding=args.outstanding, limit=args.limit)
    if not args.markdown:
      _emit({"count": len(rows), "feedback": rows})
      return 0
    from rlmcp.records.report import render_feedback_ledger

    text = render_feedback_ledger(rows)
    if args.out:
      out = Path(args.out)
      out.write_text(text)
      _emit({"ok": True, "path": str(out), "count": len(rows)})
    else:
      print(text)
    return 0

  if action == "asset":
    record = store.get_record(args.record_id)
    if record is None:
      _emit({"ok": False, "error": f"No record '{args.record_id}'."})
      return 1
    try:
      key = store.media.put(record.id, args.path, args.caption, args.kind)

      def add_asset(fresh) -> None:
        fresh.assets.setdefault(args.kind, []).append([key, args.caption])

      # The retry helper, so a concurrent write (a heartbeat, the reaper)
      # cannot make recording an asset fail on a lost compare-and-swap.
      store.update_record(record.id, add_asset)
    except StoreError as exc:
      _emit({"ok": False, "error": str(exc)})
      return 1
    _emit({"ok": True, "key": key, "kind": args.kind})
    return 0

  if action == "check":
    store.reap_expired()
    records = store.list_records()
    report = validate(
        records, slots=store.slots,
        media_exists={
            entry[0]: store.media.exists(entry[0])
            for r in records for entries in (r.assets or {}).values()
            for entry in entries if entry
        },
    )
    # Additive to the payload, never a reshape: an agent parsing `ok`,
    # `errors` and `warnings` sees exactly what it saw before. "2 unanswered"
    # is the number worth reading at a glance, because feedback nobody
    # answered is the one problem the records cannot fix by themselves.
    outstanding = sum(len(r.outstanding_feedback()) for r in records)
    _emit({
        "records": len(records),
        **report.to_dict(),
        "feedback": {
            "total": sum(len(r.feedback) for r in records),
            "unanswered": outstanding,
            "runs_with_unanswered": sorted(
                r.id for r in records if r.outstanding_feedback()),
        },
    })
    return 0 if report.ok else 1

  if action == "graph":
    from rlmcp.records.views import plot_records, render_records_html
    from rlmcp.records.graph import build, summarize
    from rlmcp.records.poster import ensure_posters

    records = store.list_records()
    if not records:
      _emit({"ok": False, "error": "No records to draw."})
      return 1
    posters: Dict[str, str] = {}
    if args.png:
      out = Path(args.out or (store.root / "records.png"))
      out.write_bytes(plot_records(records, title=args.title))
    else:
      out = Path(args.out or (store.root / "records.html"))
      # A still per filmed run, cached in the media store beside the clip it
      # came from, so the tree can show what each run looked like. Nothing here
      # can fail the render: a store with no video, no imageio or no room to
      # write one gets a page without thumbnails.
      posters = ensure_posters(records, store.media_root,
                               derive=not args.no_posters)
      # Media is referenced relatively, so the page works from a file:// path
      # with no server -- the same single-origin trick, minus the origin.
      try:
        media_base = os.path.relpath(store.media_root, out.parent) + "/"
      except ValueError:
        media_base = str(store.media_root) + "/"
      out.write_text(
          render_records_html(records, title=args.title, media_base=media_base,
                              engine=args.engine, posters=posters)
      )
    _emit({"ok": True, "path": str(out), "posters": len(posters),
           **summarize(build(records))})
    return 0

  if action == "code":
    from rlmcp.records import snapshot

    record = store.get_record(args.record_id)
    if record is None:
      _emit({"ok": False, "error": f"No record '{args.record_id}'."})
      return 1
    code = record.code or {}
    if code.get("kind") != "git":
      _emit({"ok": False, "record": record.id, "code": code,
             "error": code.get("reason") or
             "This run recorded no code snapshot. Launch with --code-root, or "
             "`rlmcp.wrap(code_root=...)`, to stamp the package."})
      return 1

    payload: Dict[str, Any] = {"ok": True, "record": record.id, "code": code}
    if args.restore:
      payload["restored"] = str(
          snapshot.restore(code["repo"], code["tree"], args.restore))
    if args.against:
      other = store.get_record(args.against)
      if other is None or (other.code or {}).get("kind") != "git":
        _emit({"ok": False,
               "error": f"'{args.against}' has no code snapshot to compare with."})
        return 1
      payload["against"] = other.id
      payload["diff"] = snapshot.diff(
          code["repo"], other.code["tree"], code["tree"])
      if args.patch:
        payload["patch"] = snapshot.patch(
            code["repo"], other.code["tree"], code["tree"])
    elif args.patch and code.get("head"):
      payload["patch"] = snapshot.patch(
          code["repo"], f"{code['head']['commit']}^{{tree}}", code["tree"],
          scope=code.get("root", ""))
    _emit(payload)
    return 0

  if action == "compare":
    from rlmcp.records.views import plot_run_comparison

    series, missing = {}, []
    for rid in args.ids:
      record = store.get_record(rid)
      if record is None or not record.session:
        missing.append(rid)
        continue
      try:
        session = Session.open(record.session)
      except FileNotFoundError:
        missing.append(rid)
        continue
      names = args.metrics or _default_metric_names(session)
      series[f"{rid} {record.slug}"] = _offline_series(session, names)
    if not series:
      _emit({"ok": False, "error": f"No readable sessions for {args.ids}.",
             "missing": missing})
      return 1
    metrics = args.metrics or sorted({m for s in series.values() for m in s})
    png = plot_run_comparison(series, metrics, at_iteration=args.at_iteration)
    out = Path(args.out or (store.root / "comparison.png"))
    out.write_bytes(png)
    _emit({"ok": True, "path": str(out), "runs": sorted(series),
           "metrics": metrics, "at_iteration": args.at_iteration,
           "unreadable": missing})
    return 0

  if action == "import":
    from rlmcp.records.importer import import_records

    result = import_records(store, args.source, dry_run=args.dry_run)
    _emit(result)
    return 0 if result.get("ok") else 1

  if action == "reindex":
    _emit({"ok": True, "indexed": store.reindex()})
    return 0

  if action == "claim":
    try:
      record = store.claim(args.record_id, slot=args.slot, ttl_seconds=args.ttl)
    except StoreError as exc:
      _emit({"ok": False, "error": str(exc)})
      return 1
    _emit({"ok": True, "record": record.id, "slot": args.slot})
    return 0

  if action == "release":
    record = store.release(args.record_id)
    _emit({"ok": bool(record), "record": args.record_id})
    return 0 if record else 1

  raise SystemExit(f"Unhandled record command '{action}'")


def _parse_conditions(tokens: Optional[List[str]]):
  """``metric op value`` triples into Conditions."""
  from rlmcp.core.curriculum import Condition

  if not tokens:
    return []
  if len(tokens) % 3:
    raise SystemExit(
        "--falsify-when takes METRIC OP VALUE triples, "
        "e.g. --falsify-when rlmcp/episode_length_frac '<=' 0.2"
    )
  return [
      Condition(tokens[i], tokens[i + 1], float(tokens[i + 2]))
      for i in range(0, len(tokens), 3)
  ]


def _falsifier_row(result: Dict[str, Any]) -> str:
  """The falsifier's one-line verdict, recorded against the run's metrics."""
  if result.get("too_early"):
    return (
        f"not evaluable yet (closed at iteration {result.get('evaluated_at')} "
        f"< check_after {result.get('check_after')})"
    )
  state = "FIRED" if result.get("fired") else (
      "undecidable" if result.get("undecidable") else "held")
  evaluated_at = result.get("evaluated_at")
  return state if evaluated_at is None else f"{state} (at iteration {evaluated_at})"


def _parse_metrics(items: Optional[List[str]]) -> List[List[str]]:
  """``name=value`` pairs, kept as strings on purpose."""
  out = []
  for item in items or []:
    if "=" not in item:
      raise SystemExit(f"Expected name=value, got '{item}'")
    name, value = item.split("=", 1)
    out.append([name, value])
  return out


def _play_command(args: argparse.Namespace) -> int:
  """``rlmcp play`` -- a checkpoint, not a session.

  The only command here that builds an environment of its own, because the run
  it is looking at has already exited. It needs no trainer and no session; a
  bare ``rlmcp play`` still means "the run I was just working on", which is the
  newest session's run directory.
  """
  from rlmcp.core.replay import parse_overrides
  from rlmcp.play import PlayError, config_from_args, run_play

  target = args.checkpoint
  if not target:
    target = str(_resolve_session(args).dir)

  try:
    config = config_from_args(args, parse_overrides(args.set))
    config.checkpoint = target
    result = run_play(config)
  except (PlayError, KeyError, ValueError) as exc:
    # KeyError is read_conditions refusing a stage this run never entered, and
    # ValueError a malformed --set; both are the caller's to fix, and neither
    # is worth a traceback.
    _emit({"ok": False, "error": str(exc).strip("'")}, command="play")
    return 1
  _emit({"ok": True, "result": result}, command="play")
  return 0


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(prog="rlmcp", description=__doc__.splitlines()[0])
  parser.add_argument("--session", help="Path to a session directory")
  parser.add_argument(
      "--root",
      help="Directory to search for the newest session "
           "(default: $RLMCP_ROOT, then ./logs, ./rlmcp_session, .)",
  )
  parser.add_argument("--timeout", type=float, default=120.0)
  # Output mode is guessed from stdout (pipe -> json, terminal -> text); these
  # are for the cases the guess cannot see, like a pretty run teed to a file.
  out = parser.add_mutually_exclusive_group()
  out.add_argument("--json", dest="output", action="store_const", const="json",
                   help="Force raw JSON output (the default when piped)")
  out.add_argument("--text", dest="output", action="store_const", const="text",
                   help="Force formatted output (the default in a terminal)")
  show = parser.add_mutually_exclusive_group()
  show.add_argument("--open", dest="open_policy", action="store_const",
                    const="always", help="Open any image or clip in the result")
  show.add_argument("--no-open", dest="open_policy", action="store_const",
                    const="never", help="Never open artifacts, just print paths")
  sub = parser.add_subparsers(dest="command", required=True)

  # Launchers. Their arguments are passed through untouched -- see main().
  sub.add_parser("train", help="Launch a training run (everything after is the trainer's)",
                 add_help=False)
  sub.add_parser("serve", help="Run the MCP server over stdio", add_help=False)

  sub.add_parser("sessions", help="List known sessions and whether they are live")

  p = sub.add_parser("tasks", help="List task ids this environment can drive")
  p.add_argument("--task-package", action="append", default=[], metavar="MODULE",
                 help="Import this module first, so its tasks register. Repeatable.")
  p.add_argument("--contains", default="",
                 help="Only ids containing this text, case-insensitively")

  p = sub.add_parser("check", help="Verify a task before training on it")
  from rlmcp import check as check_module
  check_module.add_arguments(p)
  sub.add_parser("status", help="Show live training status (reads status.json)")
  sub.add_parser("help", help="List the commands the running trainer accepts")
  sub.add_parser("info", help="Show static session info")

  p = sub.add_parser("params", help="List tunable parameters")
  p.add_argument("--contains")
  p.add_argument("--category")
  p.add_argument("--live", action="store_true", help="Ask the trainer instead of reading params.json")

  p = sub.add_parser("get", help="Read one parameter")
  p.add_argument("key")

  p = sub.add_parser("set", help="Set one parameter")
  p.add_argument("key")
  p.add_argument("value")
  p.add_argument("--why", default="", help="Rationale recorded in the event log")

  p = sub.add_parser("reset", help="Restore parameters to their startup values")
  p.add_argument("keys", nargs="*")

  p = sub.add_parser(
      "reset-envs",
      help="Start fresh episodes in some or all environments",
      description="Restarts episodes, not parameter values -- 'reset' is the "
                  "one that puts the knobs back. With no arguments every "
                  "environment restarts.",
  )
  p.add_argument("--env-id", type=int, action="append", dest="env_ids", metavar="N",
                 help="Reset only this environment. Repeatable.")
  p.add_argument("--where", nargs="*", metavar="KEY=VALUE",
                 help="Reset the envs matching a description, e.g. terrain=pyramid_stairs")
  p.add_argument("--why", default="", help="Rationale recorded in the event log")

  p = sub.add_parser("metrics", help="Show recent metric values")
  p.add_argument("names", nargs="*")
  p.add_argument("--last-n", type=int, default=30)
  p.add_argument("--list", action="store_true", help="List metric names only")
  p.add_argument("--contains")
  p.add_argument("--offline", action="store_true",
                 help="Read metrics.jsonl instead of asking the trainer")

  p = sub.add_parser("plot", help="Plot metrics to a PNG")
  p.add_argument("names", nargs="*")
  p.add_argument("--last-n", type=int, default=400)
  p.add_argument("--smooth", type=int, default=5)
  p.add_argument("--offline", action="store_true",
                 help="Plot from metrics.jsonl instead of asking the trainer")

  p = sub.add_parser("shot", help="Screenshot one environment")
  p.add_argument("--env-id", type=int)
  p.add_argument("--where", nargs="*", metavar="KEY=VALUE",
                 help="Pick an env by description, e.g. terrain=pyramid_stairs level=2")

  p = sub.add_parser(
      "video", help="Record a short clip of training, or set the clip schedule",
      description="With no --every, records one clip now. With --every N, "
                  "changes the automatic schedule the run is already taking "
                  "clips on ('--every off' stops it) and reports it.",
  )
  p.add_argument("--seconds", type=float, default=4.0)
  p.add_argument("--env-id", type=int)
  p.add_argument("--where", nargs="*", metavar="KEY=VALUE",
                 help="Pick an env by description, e.g. terrain=pyramid_stairs level=2")
  p.add_argument("--every", metavar="CADENCE",
                 help="Change the automatic cadence: 'double' (0, 50, 100, 200, "
                      "400 ...), 'double:<first>:<cap>', a flat interval like "
                      "'200', or 'off' for no clips ('none', 'never' and '0' "
                      "mean the same)")
  p.add_argument("--budget-mb", type=float,
                 help="Disk the progress clips may use before they stop")
  p.add_argument("--schedule", action="store_true",
                 help="Report the automatic clip schedule without recording")

  p = sub.add_parser(
      "view",
      help="Watch the run live in a browser, over viser",
      description="A run trains with one of these already attached, so with "
                  "no flags this reports where it is. The rest re-point it, "
                  "stop it paying for itself, or give the port back -- none "
                  "of which restarts or pauses the run itself.",
  )
  p.add_argument("--on", dest="on", action="store_true",
                 help="Attach the view and print its URL")
  p.add_argument("--off", dest="off", action="store_true",
                 help="Detach the view and give the port back")
  p.add_argument("--pause", dest="paused", action="store_true", default=None,
                 help="Stop feeding the view without detaching it: the tab "
                      "keeps the frame it has and the run goes back to full "
                      "speed. The same thing the button in the tab does")
  p.add_argument("--resume", dest="paused", action="store_false",
                 help="Start feeding it again")
  p.add_argument("--port", type=int,
                 help="First port to try (default 8740; busy ports are skipped)")
  p.add_argument("--host", help="Interface to bind (default 0.0.0.0)")
  p.add_argument("--fps", type=float,
                 help="Frames per second pushed while somebody is watching")
  p.add_argument("--realtime", dest="realtime", action="store_true", default=None,
                 help="Play a buffered window back at the speed the robot "
                      "actually moves, with pause and a speed control in the tab")
  p.add_argument("--live", dest="realtime", action="store_false",
                 help="The opposite: show the current step, at the run's pace")
  p.add_argument("--buffer-seconds", type=float,
                 help="Sim time one realtime window holds (default 4)")
  p.add_argument("--env-id", type=int, help="Which environment to show")
  p.add_argument("--where", nargs="*", metavar="KEY=VALUE",
                 help="Pick the environment by description instead, e.g. "
                      "terrain=pyramid_stairs level=2")

  p = sub.add_parser(
      "play",
      help="Play a task: render a clip, or open a viewer",
      description="Replay a checkpoint under the conditions it was trained on. "
                  "Needs no live trainer -- this is how a finished run gets "
                  "looked at. With --policy zero or random it needs no "
                  "checkpoint either, which is how a task being written gets "
                  "looked at.",
  )
  # Cheap to import: everything in rlmcp.play that needs a simulator, torch or
  # an encoder imports it inside a function, so the CLI stays standard-library
  # only until somebody actually asks to play something.
  from rlmcp.play import add_arguments as play_arguments

  play_arguments(p)

  p = sub.add_parser("trace", help="Record per-step joint signals")
  p.add_argument("--seconds", type=float, default=4.0)
  p.add_argument("--env-id", type=int)
  p.add_argument("--where", nargs="*", metavar="KEY=VALUE",
                 help="Pick an env by description, e.g. terrain=pyramid_stairs")

  p = sub.add_parser("diagnose", help="Record a trace, analyse it, and plot it")
  p.add_argument("--seconds", type=float, default=4.0)
  p.add_argument("--env-id", type=int)
  p.add_argument("--where", nargs="*", metavar="KEY=VALUE",
                 help="Pick an env by description, e.g. terrain=pyramid_stairs")

  p = sub.add_parser("analyze", help="Re-analyse a saved trace .npz without a live run")
  p.add_argument("trace_path")
  p.add_argument("--plot", action="store_true", help="Also write a PNG next to it")
  p.add_argument("--allow-legacy", action="store_true",
                 help="Unpickle a pre-JSON trace; only for files whose origin you trust")

  p = sub.add_parser("plot-trace", help="Re-plot the last trace")
  p.add_argument("--channels", nargs="*")
  p.add_argument("--components", nargs="*", help="Filter joints by substring")

  p = sub.add_parser("curriculum", help="Inspect or drive the curriculum")
  p.add_argument(
      "action", nargs="?", default="status",
      choices=["status", "advance", "goto", "auto-on", "auto-off"],
  )
  p.add_argument("stage", nargs="?")
  p.add_argument("--why", default="manual")

  sub.add_parser("commands", help="List every command this run accepts")

  p = sub.add_parser("extensions", help="List installed capability extensions")
  p.add_argument("--available", action="store_true",
                 help="Only those the running session actually loaded")

  sub.add_parser("pause", help="Pause training")
  sub.add_parser("resume", help="Resume training")
  sub.add_parser("step-once", help="Advance one iteration while paused")

  p = sub.add_parser("checkpoint", help="Save a checkpoint")
  p.add_argument("tag", nargs="?", default="")
  p.add_argument("--note", default="")

  sub.add_parser("checkpoints", help="List checkpoints saved in this session")

  p = sub.add_parser("load", help="Roll back to a checkpoint")
  p.add_argument("path")

  p = sub.add_parser("note", help="Record a note in the event log")
  p.add_argument("text")

  p = sub.add_parser(
      "feedback",
      help="Record what a human said about the live run (stamped with the iteration)")
  p.add_argument("text", help="Verbatim. What was actually said.")
  p.add_argument("--kind", default="steer", choices=list(FEEDBACK_KINDS),
                 help="What they were doing: steering, correcting, rejecting, "
                      "approving, observing, or setting a standing rule")
  p.add_argument("--author", default="user")
  p.add_argument("--read-as", dest="interpretation", default="",
                 help="What you took it to mean, if that is not obvious")

  p = sub.add_parser("events", help="Show recent session events")
  p.add_argument("--last-n", type=int, default=25)
  p.add_argument("--interventions", action="store_true",
                 help="Only what was done to the run -- parameter edits, stage "
                      "changes, checkpoints, notes -- each with its reason")

  p = sub.add_parser("stop", help="Stop a run -- or a play session -- cleanly")
  p.add_argument("--why", default="")

  rec = sub.add_parser("record", help="Run records: plans, outcomes, ancestry")
  rec.add_argument("--records-root", help="Records directory (default: $RLMCP_RECORDS or ./records)")
  rec.add_argument("--slots", type=int, default=1,
                   help="How many runs may hold a lease at once (default: 1)")
  record_sub = rec.add_subparsers(dest="record_command", required=True)

  q = record_sub.add_parser("new", help="Pre-register a run, before launching it")
  q.add_argument("slug")
  q.add_argument("--headline", default="",
                 help="One sentence for the tree. Derived from the outcome if omitted")
  q.add_argument("--hypothesis", default="")
  q.add_argument("--prediction", default="")
  q.add_argument("--falsifier", default="", help="What would prove this wrong")
  q.add_argument("--falsify-when", nargs="*", metavar="METRIC OP VALUE",
                 help="Machine-checkable falsifier, e.g. rlmcp/episode_length_frac '<=' 0.2")
  q.add_argument("--check-after", type=int, default=0,
                 help="Iteration before which the falsifier means nothing "
                      "(every policy is bad at iteration zero)")
  q.add_argument("--change", nargs="*", help="What differs from the parent")
  q.add_argument("--parent", help="Config ancestry: the run this one changes")
  q.add_argument("--weights", help="Warm start from this run id")
  q.add_argument("--checkpoint", default="", help="Checkpoint within --weights")
  q.add_argument("--stage", default="default")
  q.add_argument("--task", default="",
                 help="Which problem this run is about (the environment id). "
                      "Filled in from the live session at launch if omitted")
  q.add_argument("--proposed-by", default="human")

  q = record_sub.add_parser("list", help="List records")
  q.add_argument("--stage")
  q.add_argument("--verdict")
  q.add_argument("--parent")
  q.add_argument("--text", help="Match slug, hypothesis or outcome")

  q = record_sub.add_parser("show", help="Show one record and its computed recipe")
  q.add_argument("record_id")

  q = record_sub.add_parser("close", help="Close a run with a verdict")
  q.add_argument("record_id")
  q.add_argument("verdict")
  q.add_argument("--headline", default="",
                 help="One sentence for the tree. Derived from the outcome if omitted")
  q.add_argument("--outcome", default="")
  q.add_argument("--metric", nargs="*", metavar="NAME=VALUE",
                 help="The measurements that establish the outcome")

  q = record_sub.add_parser(
      "headline", help="Set the one-sentence summary shown on the tree")
  q.add_argument("record_id")
  q.add_argument("text", nargs="?",
                 help="Omit to clear it and fall back to the derived sentence")

  q = record_sub.add_parser(
      "feedback", help="Attach what a human said to a run record")
  q.add_argument("record_id")
  q.add_argument("text", help="Verbatim. What was actually said.")
  q.add_argument("--kind", default="steer", choices=list(FEEDBACK_KINDS))
  q.add_argument("--author", default="user")
  q.add_argument("--read-as", dest="interpretation", default="",
                 help="What you took it to mean")
  q.add_argument("--did", dest="response", default="",
                 help="What you did about it, if you already have")
  q.add_argument("--no-change", action="store_true",
                 help="With --did: you answered it, but nothing changed")
  q.add_argument("--affects", nargs="*", default=[], metavar="RUN_ID",
                 help="Other runs this remark changed")
  q.add_argument("--artifact", nargs="*", default=[], dest="artifacts",
                 metavar="PATH", help="Paths that exist because of it")
  q.add_argument("--iteration", type=int,
                 help="Training iteration (read from the run's session if omitted)")

  q = record_sub.add_parser(
      "answer", help="Record what was done about a piece of feedback")
  q.add_argument("record_id")
  q.add_argument("index", type=int, help="Position in the run's feedback list")
  q.add_argument("response")
  q.add_argument("--no-change", action="store_true",
                 help="You answered it, but nothing changed as a result")

  q = record_sub.add_parser(
      "timeline", help="Every remark across the records, oldest first")
  q.add_argument("--kind", choices=list(FEEDBACK_KINDS))
  q.add_argument("--author")
  q.add_argument("--outstanding", action="store_true",
                 help="Only instructions with no recorded response")
  q.add_argument("--limit", type=int)
  q.add_argument("--markdown", action="store_true",
                 help="Render the ledger as Markdown instead of JSON")
  q.add_argument("--out", help="With --markdown: write here instead of stdout")

  q = record_sub.add_parser("asset", help="Record an artifact against a run")
  q.add_argument("record_id")
  q.add_argument("path")
  q.add_argument("--caption", default="")
  q.add_argument("--kind", default="plots", choices=["plots", "videos"])

  q = record_sub.add_parser("graph", help="Render the record graph")
  q.add_argument("--out", help="Where to write (default: <records>/records.html)")
  q.add_argument("--png", action="store_true", help="Write a PNG instead of the viewer")
  q.add_argument("--title", default="rlmcp run tree")
  q.add_argument("--no-posters", action="store_true",
                 help="Do not derive missing clip stills (read-only media)")
  q.add_argument("--engine", default="auto", choices=["auto", "cytoscape", "simple"],
                 help="Interactive graph view, or a dependency-free fallback")

  q = record_sub.add_parser(
      "code", help="What the task package was when a run launched",
      description="The other half of a recipe. With --against, the code diff "
                  "between two runs -- committed or not.")
  q.add_argument("record_id")
  q.add_argument("--against", metavar="RECORD_ID",
                 help="Diff this run's package against another run's")
  q.add_argument("--patch", action="store_true", help="Print the unified diff")
  q.add_argument("--restore", metavar="DIR",
                 help="Write the package back out as that run had it")

  q = record_sub.add_parser("compare", help="Overlay runs on the same metrics")
  q.add_argument("ids", nargs="+")
  q.add_argument("--metrics", nargs="*")
  q.add_argument("--at-iteration", type=int,
                 help="Truncate every run to this iteration; 'better at the end' "
                      "often just means 'trained longer'")
  q.add_argument("--out")

  q = record_sub.add_parser("import", help="Import records from another store")
  q.add_argument("source", help="Directory holding <id>/meta.json or legacy *.json")
  q.add_argument("--dry-run", action="store_true")

  record_sub.add_parser("check", help="Validate every record")
  record_sub.add_parser("reindex", help="Rebuild the index from the files")

  q = record_sub.add_parser("claim", help="Take a slot for a run")
  q.add_argument("record_id")
  q.add_argument("--slot", default="gpu0")
  q.add_argument("--ttl", type=float, default=900.0)

  q = record_sub.add_parser("release", help="Give a slot back")
  q.add_argument("record_id")

  p = sub.add_parser("run", help="Run any command this session accepts")
  p.add_argument("cmd")
  p.add_argument("args", nargs="*", help="key=value pairs (values parsed as JSON)")

  p = sub.add_parser("raw", help="Alias for 'run'")
  p.add_argument("cmd")
  p.add_argument("args", nargs="*", help="key=value pairs (values parsed as JSON)")

  return parser


def main(argv: Optional[List[str]] = None) -> int:
  global _MODE, _OPEN

  # `train` and `serve` launch a process rather than talking to one, and their
  # flags are their own. Hand the rest of the line over before argparse claims
  # it. Imported lazily: `rlmcp status` should not pay for mjlab or torch.
  head = list(sys.argv[1:] if argv is None else argv)
  if head and head[0] == "train":
    from rlmcp.train import main as train_main
    return train_main(head[1:])
  if head and head[0] == "serve":
    from rlmcp.server.mcp_server import main as serve_main
    return serve_main(head[1:])

  args = build_parser().parse_args(argv)
  cmd = args.command
  timeout = args.timeout
  _MODE = cli_output.resolve_mode(getattr(args, "output", None))
  _OPEN = cli_output.resolve_open(getattr(args, "open_policy", None))

  if cmd == "tasks":
    # The one command with no session: it answers what *could* run, which is
    # the question a task being built has no session to answer. Importing a
    # simulator costs seconds, so it happens here rather than at module load.
    from rlmcp import tasks as task_registry

    payload = task_registry.registered(args.task_package, args.contains)
    _emit(payload, command="tasks")

    # A package that would not import is the usual reason an id is missing, and
    # it is invisible in the list itself -- the task is simply not there.
    failed = [e for e in payload["packages"] if not e["imported"]]
    for entry in failed:
      print(cli_output.note(
          f"[rlmcp] '{entry['module']}' did not import, so its tasks are "
          f"missing from this list: {entry['error']}"), file=sys.stderr)

    if payload["tasks"]:
      return 1 if failed else 0

    # Nothing to show, and three different reasons for it.
    total = sum(b["tasks"] for b in payload["backends"])
    live = [b for b in payload["backends"] if b["available"]]
    if not live:
      print(cli_output.note(
          "[rlmcp] no backend here can list tasks: "
          + "; ".join(f"{b['backend']} {b['reason']}" for b in payload["backends"])),
          file=sys.stderr)
    elif args.contains and total:
      print(cli_output.note(
          f"[rlmcp] no task id contains '{args.contains}'; {total} registered "
          "here. Drop --contains to see them."), file=sys.stderr)
    else:
      print(cli_output.note(
          "[rlmcp] nothing imported here registers a task. Pass "
          f"--task-package <module>, or set ${task_registry.TASK_PACKAGES_ENV}, "
          "naming the package whose import registers yours -- the same package "
          "`rlmcp train --task-package` is given."), file=sys.stderr)
    return 1
  if cmd == "check":
    # No session: this runs before anything has trained, which is the point.
    # The envelope is `play`'s, for the same reason -- a command that stands in
    # for a trainer that is not there yet.
    from rlmcp import check as check_module

    try:
      answer = check_module.run_check(check_module.config_from_args(args))
    except check_module.CheckError as exc:
      # 1, not 2: the same refusal envelope `play` and `analyze` return when
      # they cannot start, and the same exit code.
      _emit({"ok": False, "error": str(exc)}, command="check")
      return 1
    _emit({"ok": True, "result": answer}, command="check")
    # The verdict is the exit code, so `rlmcp check && rlmcp train` is a
    # sentence somebody can write.
    return 0 if answer["passed"] else 1

  if cmd == "sessions":
    from rlmcp import registry
    from rlmcp.session import PLAY_SESSION_KIND

    roots, origin = _search_roots(args)
    seen, rows = set(), []
    newest: Optional[Session] = None
    newest_started = 0.0

    def add(session: Session) -> None:
      nonlocal newest, newest_started
      if str(session.dir) in seen:
        return
      seen.add(str(session.dir))
      info = session.info()
      live = session.liveness_info()
      rows.append(
          {
              "session": str(session.dir),
              "task": info.get("task"),
              "num_envs": info.get("num_envs"),
              "started_at": info.get("started_at"),
              "state": live["state"],
              "alive": live["pid_alive"],
              "iteration": session.status().get("iteration"),
              "heartbeat_age_s": live["heartbeat_age_s"],
          }
      )
      started = info.get("started_at") or 0.0
      if info.get("kind") != PLAY_SESSION_KIND and (
          newest is None or started > newest_started):
        newest, newest_started = session, started

    for root in roots:
      # Everything here, play sessions included: this is the command for
      # finding out what exists, so it should not hide anything.
      for session in iter_sessions(root, include_play=True):  # Newest first.
        add(session)

    # The registry widens the answer beyond the cwd -- but only the default
    # search: an explicit --root or RLMCP_ROOT is a scoped question, and a
    # scoped question deserves a scoped answer.
    reg_rows = registry.entries() if origin == "default" else []
    for row in reg_rows:
      if row.get("session_exists"):
        try:
          add(Session.open(row["session_dir"]))
        except FileNotFoundError:
          continue
      elif row.get("root_exists"):
        for session in iter_sessions(row["root"], include_play=True):
          add(session)
    if reg_rows:
      # Merged sources arrive in source order; restore newest-first.
      rows.sort(key=lambda r: -(r.get("started_at") or 0.0))

    _emit(rows, command="sessions")
    if newest is not None:
      # What "the run I was just working on" resolves to next time, anywhere.
      _remember(newest)
    for server in (r for r in reg_rows
                   if r["kind"] == registry.KIND_SERVER and r["pid_alive"]):
      where = server.get("session_dir") or server.get("root")
      print(cli_output.note(
          f"[rlmcp] rlmcp-server pid {server['pid']} is running, "
          f"watching '{where}'"), file=sys.stderr)
    if not rows:
      searched = ", ".join(str(r) for r in roots)
      where = (f"searched {searched} under {Path.cwd()}"
               if origin == "default" else
               f"searched {roots[0]} (from {origin})")
      extra = ("; the registry adds nothing" if origin == "default" else "")
      print(cli_output.note(f"[rlmcp] {where}{extra}"), file=sys.stderr)
      print(cli_output.note(
          "[rlmcp] a training run registers itself when it starts; "
          "--root <dir> or RLMCP_ROOT points the search somewhere specific"),
          file=sys.stderr)
    return 0

  if cmd == "analyze":
    return _analyze_offline(args.trace_path, plot=args.plot,
                            allow_legacy=args.allow_legacy)

  if cmd == "record":
    return _record_command(args)

  if cmd == "play":
    return _play_command(args)

  if cmd == "extensions" and not args.available:
    # A static question about what is installed; no session needed.
    from rlmcp.extensions import catalog

    _emit({"installed": catalog()})
    return 0

  session = _resolve_session(args)

  # Read-only commands answer straight from disk, so they work even after the
  # trainer has exited.
  if cmd == "status":
    live = session.liveness_info()
    _emit({
        "session": str(session.dir),
        "state": live["state"],
        "alive": live["pid_alive"],
        "heartbeat_age_s": live["heartbeat_age_s"],
        **({"liveness_note": live["note"]} if "note" in live else {}),
        **session.status(),
    }, command="status")
    return 0
  if cmd == "info":
    _emit({"session": str(session.dir), **session.info()}, command="info")
    return 0
  if cmd == "events":
    if args.interventions:
      from rlmcp.records.interventions import from_events

      chosen = from_events(session.events())
      _emit({"count": len(chosen), "interventions": chosen[-args.last_n:]})
      return 0
    _emit(session.events(last_n=args.last_n), command="events")
    return 0
  if cmd == "params" and not args.live:
    schema = session.params()
    items = {
        k: v
        for k, v in schema.items()
        if (not args.contains or args.contains.lower() in k.lower())
        and (not args.category or v.get("category") == args.category)
    }
    _emit({"count": len(items), "parameters": items}, command="list_parameters")
    return 0
  # A finished run still has everything on disk; fall back automatically rather
  # than reporting a dead process for questions the files can answer. The pid
  # alone is not trusted here: liveness() treats a day-stale heartbeat behind a
  # "live" pid as a recycled pid, so a long-dead run still gets its fallback.
  # Default metric names are resolved lazily -- only these offline paths need
  # them, and computing them reads the metrics file.
  if cmd == "plot" and (args.offline or session.liveness() == "dead"):
    return _offline_plot(session, args.names or _default_metric_names(session),
                         args.last_n, args.smooth)
  if cmd == "metrics" and not args.list and (args.offline or session.liveness() == "dead"):
    names = args.names or _default_metric_names(session)
    series = _offline_series(session, names, last_n=args.last_n)
    try:
      from rlmcp.core.diagnostics import summarize_metric_history

      # The summary's window math looks at the last ~40 points; a bounded tail
      # feeds it fully without re-reading a day-old file end to end.
      summary = summarize_metric_history(session.metrics(last_n=max(args.last_n, 200)), names)
    except ImportError:
      summary = {}
    _emit({"ok": True, "result": {"metrics": series, "summary": summary, "live": False}},
          command="get_metrics")
    return 0

  if cmd == "metrics" and args.list:
    rows = session.metrics(last_n=1)
    names = sorted({k for row in rows for k in row if k not in ("iteration", "t")})
    if args.contains:
      names = [n for n in names if args.contains.lower() in n.lower()]
    _emit({"count": len(names), "metrics": names}, command="list_metrics")
    return 0

  dispatch = {
      "help": ("help", {}),
      "params": ("list_parameters", {"contains": getattr(args, "contains", None),
                                     "category": getattr(args, "category", None)}),
      "get": ("get_parameter", {"key": getattr(args, "key", None)}),
      "commands": ("help", {}),
      "extensions": ("status", {}),
      "pause": ("pause", {}),
      "resume": ("resume", {}),
      "step-once": ("step_once", {}),
      "checkpoints": ("list_checkpoints", {}),
  }
  if cmd in dispatch:
    name, kwargs = dispatch[cmd]
    return _call(session, name, timeout, **{k: v for k, v in kwargs.items() if v is not None})

  if cmd == "set":
    return _call(session, "set_parameter", timeout, key=args.key,
                 value=_parse_value(args.value), rationale=args.why)
  if cmd == "reset":
    return _call(session, "reset_parameters", timeout, keys=args.keys or None)
  if cmd == "reset-envs":
    return _call(session, "reset_envs", timeout, env_ids=args.env_ids or None,
                 where=_kv_pairs(args.where) or None, rationale=args.why)
  if cmd == "metrics":
    return _call(session, "get_metrics", timeout, names=args.names or None,
                 last_n=args.last_n)
  if cmd == "plot":
    return _call(session, "plot_metrics", timeout, names=args.names or None,
                 last_n=args.last_n, smooth=args.smooth)
  if cmd == "shot":
    return _call(session, "screenshot", timeout, env_id=args.env_id,
                 where=_kv_pairs(args.where) or None)
  if cmd == "video":
    if args.every is not None or args.budget_mb is not None or args.schedule:
      return _call(session, "progress_video", timeout, every=args.every,
                   seconds=args.seconds if args.every is not None else None,
                   env_id=args.env_id, budget_mb=args.budget_mb)
    return _call(session, "record_video", max(timeout, args.seconds * 20 + 60),
                 seconds=args.seconds, env_id=args.env_id,
                 where=_kv_pairs(args.where) or None)
  if cmd == "view":
    if args.off:
      enabled = False
    elif args.on:
      enabled = True
    else:
      # Any setting on its own means "make it so": somebody asking for another
      # environment or another rate wants to be looking at one, and refusing
      # until they also type --on would be pedantry.
      # --pause is not in this list on purpose: "stop paying for the view"
      # is the one setting that must never be the thing that starts one.
      enabled = True if (args.port is not None or args.fps is not None
                         or args.env_id is not None or args.where
                         or args.host is not None or args.realtime is not None
                         or args.buffer_seconds is not None) else None
    return _call(session, "live_view", timeout, enabled=enabled, port=args.port,
                 host=args.host, fps=args.fps, env_id=args.env_id,
                 realtime=args.realtime, buffer_seconds=args.buffer_seconds,
                 paused=args.paused, where=_kv_pairs(args.where) or None)
  if cmd in ("trace", "diagnose"):
    name = "record_trace" if cmd == "trace" else "diagnose"
    return _call(session, name, max(timeout, args.seconds * 20 + 60),
                 seconds=args.seconds, env_id=args.env_id,
                 where=_kv_pairs(args.where) or None)
  if cmd == "plot-trace":
    return _call(session, "plot_trace", timeout, channels=args.channels,
                 components=args.components)
  if cmd == "curriculum":
    if args.action == "status":
      return _call(session, "curriculum_status", timeout)
    if args.action == "advance":
      return _call(session, "curriculum_advance", timeout, reason=args.why)
    if args.action == "goto":
      if not args.stage:
        raise SystemExit("curriculum goto needs a stage name")
      return _call(session, "curriculum_goto", timeout, stage=args.stage, reason=args.why)
    return _call(session, "curriculum_auto", timeout, enabled=args.action == "auto-on")
  if cmd == "checkpoint":
    return _call(session, "save_checkpoint", max(timeout, 300.0), tag=args.tag,
                 note=args.note)
  if cmd == "load":
    return _call(session, "load_checkpoint", max(timeout, 300.0), path=args.path)
  if cmd == "note":
    return _call(session, "note", timeout, text=args.text)
  if cmd == "feedback":
    return _call(session, "feedback", timeout, text=args.text, kind=args.kind,
                 author=args.author, interpretation=args.interpretation)
  if cmd == "stop":
    return _call(session, "stop_training", timeout, reason=args.why)
  if cmd in ("run", "raw"):
    return _call(session, args.cmd, max(timeout, 300.0), **_kv_pairs(args.args))

  raise SystemExit(f"Unhandled command '{cmd}'")


if __name__ == "__main__":
  sys.exit(main())
