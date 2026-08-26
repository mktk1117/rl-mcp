"""The run record: one experiment, as data.

A record is written twice. Before the run it carries a hypothesis, a prediction
and a falsifier; after the run it carries an outcome, measurements and a verdict.
A run you cannot falsify is a run you cannot learn from, so the falsifier is not
optional decoration -- it is the field that makes the rest an experiment.

Three properties are worth stating up front, because they are what the schema
is shaped around.

**The recipe is never stored.** ``parent`` is the config lineage; the recipe at
any node is the fold of ``change`` from the root down. Computed, never written,
so provenance and the current recipe cannot drift apart.

**A warm start is a different claim.** ``weights`` records that a run inherited a
policy, which proves a change *preserves* a behaviour rather than *creates* one.
The validator refuses to let such a run be called ``validated``.

**Feedback is append-only, and owes an answer.** ``feedback`` is what a human
said about the run -- steer, correct, reject, approve. It is never edited, only
appended to, and each entry carries a ``response`` slot for what was done about
it. An instruction with no recorded response is the shape of feedback that was
heard and dropped, and the point of storing it is to make that visible rather
than deniable.

Stdlib only, like :mod:`rlmcp.session` -- a record must be readable in an
interpreter that has no simulator, and eventually by a service that has no GPU.
"""

from __future__ import annotations

import os
import re
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rlmcp.core.curriculum import Condition

SCHEMA_VERSION = 2
"""The shape records are written in today.

Version 2 added ``feedback``, ``headline`` and ``task``. Every one of them is
optional and defaults to empty, so a version 1 record loads unchanged and keeps
its own stamp until something rewrites it -- nothing in this package branches on
the number, and a store written by an older build stays readable by a newer one.
"""

VERDICTS = (
    "planned",
    "running",
    "validated",
    "falsified",
    "provisional",
    "control",
    "superseded",
    "best",
    "interrupted",
)
"""The states a record may hold.

``falsified`` is a good outcome: a run that killed its hypothesis in twenty
minutes is information-dense. ``provisional`` means "measured only as a delta on
inherited weights", promoted to ``validated`` by a from-scratch run.
``interrupted`` is everything the reaper leaves behind -- an expired lease, a
runner whose process died, or a ``running`` record that went silent holding no
lease at all (``links["reaped"]`` says which). A job that died is not a result,
and it must not sit at ``running`` forever holding a slot; it still owes a real
close-out, which is why ``interrupted`` alone among the terminal verdicts can
be closed again.
"""

OPEN_VERDICTS = ("planned", "running")
"""Verdicts that mean the run has not produced a result yet."""

TERMINAL_VERDICTS = tuple(v for v in VERDICTS if v not in OPEN_VERDICTS)

_SLUG_RE = re.compile(r"[^a-z0-9]+")

MAX_SLUG_LENGTH = 80
"""Slugs become directory names; a pasted paragraph must not."""


def slugify(text: str) -> str:
  """A filesystem- and url-safe slug, for directory names and display."""
  slug = _SLUG_RE.sub("_", text.strip().lower()).strip("_")
  return slug[:MAX_SLUG_LENGTH].rstrip("_") or "run"


def _utc_today() -> str:
  """The record's date, host-timezone-independent."""
  return datetime.now(timezone.utc).date().isoformat()


FEEDBACK_KINDS = (
    "steer",
    "correct",
    "reject",
    "approve",
    "observe",
    "constrain",
)
"""What a human was doing when they said it.

Kept small on purpose -- a taxonomy nobody can hold in their head gets filled
in at random, and then the label carries no information. ``steer`` points at
what to do next, ``correct`` says something already done is wrong, ``reject``
says stop, ``approve`` says keep going, ``observe`` is a human eye on the
behaviour that no metric caught, and ``constrain`` is a standing rule that
outlives this run.
"""

OWED_A_RESPONSE = ("steer", "correct", "reject", "constrain")
"""Kinds that asked for something, and so are unanswered until they get it.

``approve`` and ``observe`` are information rather than instructions; leaving
them without a response is not a dropped ball.
"""


@dataclass
class Feedback:
  """One thing a human said about this run, and what was done about it.

  Identity is position in ``RunRecord.feedback``: entries are appended and never
  reordered or removed, so index ``3`` keeps meaning the same remark forever.
  ``response`` is the one mutable part, because the answer usually arrives after
  the remark.
  """

  text: str
  """Verbatim. What was actually said, not a paraphrase -- a paraphrase is
  already an interpretation, and the two are worth keeping apart."""
  kind: str = "steer"
  author: str = "user"
  at: float = field(default_factory=time.time)
  iteration: Optional[int] = None
  """Training iteration when it was said, when the run was live to hear it."""

  interpretation: str = ""
  """What it was taken to mean. Separate from ``text`` because this is where a
  misreading becomes visible afterwards -- and separate from ``response``
  because understanding a remark and acting on it are different failures."""
  response: str = ""
  """What was done about it. Empty means nothing is recorded yet."""
  changed: bool = False
  """Whether anything actually changed as a result.

  Not derivable from ``response``: "investigated, and the answer was already
  right" is a real response that changed nothing. Recording it honestly is what
  keeps the ledger from reading as though every remark moved the project.
  """

  affects: List[str] = field(default_factory=list)
  """Other run ids this remark changed. The entry lives on the run that was
  live when it was said; this is its blast radius."""
  artifacts: List[str] = field(default_factory=list)
  """Paths that exist because of it."""

  @property
  def answered(self) -> bool:
    return bool(self.response.strip())

  @property
  def outstanding(self) -> bool:
    """An instruction with no recorded response."""
    return self.kind in OWED_A_RESPONSE and not self.answered

  def to_dict(self) -> Dict[str, Any]:
    return {
        "text": self.text,
        "kind": self.kind,
        "author": self.author,
        "at": self.at,
        "iteration": self.iteration,
        "interpretation": self.interpretation,
        "response": self.response,
        "changed": self.changed,
        "affects": list(self.affects),
        "artifacts": list(self.artifacts),
    }

  @staticmethod
  def from_dict(d: Any) -> "Feedback":
    # Tolerate a bare string: that is what a hand-edited record and a rough
    # import both produce, and losing the remark would be worse than defaulting
    # its kind.
    if isinstance(d, str):
      return Feedback(text=d)
    iteration = d.get("iteration")
    return Feedback(
        text=d.get("text", ""),
        kind=d.get("kind", "steer"),
        author=d.get("author", "user"),
        at=float(d.get("at", time.time())),
        iteration=None if iteration is None else int(iteration),
        interpretation=d.get("interpretation", ""),
        response=d.get("response", ""),
        changed=bool(d.get("changed", False)),
        affects=list(d.get("affects") or []),
        artifacts=list(d.get("artifacts") or []),
    )


@dataclass
class Weights:
  """A warm start: the run whose policy this one inherited."""

  run: str
  checkpoint: str = ""

  def to_dict(self) -> Dict[str, Any]:
    return {"run": self.run, "checkpoint": self.checkpoint}

  @staticmethod
  def from_dict(d: Dict[str, Any]) -> "Weights":
    return Weights(run=d["run"], checkpoint=d.get("checkpoint", ""))

  def describe(self) -> str:
    return f"{self.run} @ {self.checkpoint}" if self.checkpoint else self.run


@dataclass
class Falsifier:
  """What observation would prove the hypothesis wrong.

  ``prose`` is for the human reading the record. ``conditions`` are for whatever
  closes the run -- an orchestrator running a hundred jobs cannot read a
  sentence, and "the falsifier fired" should be a fact rather than a judgement.

  A condition here reads the opposite way round from a curriculum promotion: it
  is true when the hypothesis is *dead*.
  """

  prose: str = ""
  conditions: List[Condition] = field(default_factory=list)
  check_after: int = 0
  """Iteration before which the falsifier means nothing.

  Every policy is bad at iteration zero, so a falsifier evaluated immediately
  fires on every run and stops carrying information. This is the pre-registered
  read-point: "do not read this run before N".
  """

  def check(self, metrics: Dict[str, float], iteration: Optional[int] = None) -> Dict[str, Any]:
    """Evaluate every condition against a metric snapshot.

    Returns ``fired`` (any condition met), ``undecidable`` (a condition whose
    metric is absent, which is not the same as "did not fire"), and the
    per-condition detail.
    """
    if iteration is not None and iteration < self.check_after:
      return {
          "fired": False,
          "undecidable": False,
          "too_early": True,
          "check_after": self.check_after,
          "checks": [],
          "prose": self.prose,
      }

    checks: List[Dict[str, Any]] = []
    fired = False
    undecidable = False
    for condition in self.conditions:
      met, current = condition.check(metrics)
      if current is None:
        undecidable = True
      fired = fired or met
      checks.append(
          {
              "condition": condition.describe(),
              "current": None if current is None else round(current, 6),
              "fired": met,
              "measurable": current is not None,
          }
      )
    return {
        "fired": fired,
        "undecidable": undecidable and not fired,
        "too_early": False,
        "check_after": self.check_after,
        "checks": checks,
        "prose": self.prose,
    }

  def is_empty(self) -> bool:
    return not self.prose.strip() and not self.conditions

  def to_dict(self) -> Dict[str, Any]:
    return {
        "prose": self.prose,
        "conditions": [c.to_dict() for c in self.conditions],
        "check_after": self.check_after,
    }

  @staticmethod
  def from_dict(d: Any) -> "Falsifier":
    # Tolerate a bare string, which is what a human writes first.
    if isinstance(d, str):
      return Falsifier(prose=d)
    if not d:
      return Falsifier()
    return Falsifier(
        prose=d.get("prose", ""),
        conditions=[Condition.from_dict(c) for c in d.get("conditions") or []],
        check_after=int(d.get("check_after", 0)),
    )


@dataclass
class Lease:
  """A claim on a scarce resource, held by a running job.

  This replaces the single-``running``-record mutex, which only works when there
  is one GPU. A lease expires unless the job renews it, so a machine that dies
  releases its slot instead of blocking the next launch forever.

  ``owner`` says what kind of claim this is. A ``runner`` lease is held by the
  trainer process itself: its pid is meaningful, so pid-death releases the slot
  immediately. A ``manual`` lease is a reservation made from a short-lived
  process (the CLI) on behalf of a human; recording that process's pid would
  hand the slot back the moment the CLI exited, so a manual lease answers to
  its TTL alone.
  """

  slot: str
  holder: str = ""
  host: str = field(default_factory=socket.gethostname)
  pid: int = 0
  acquired_at: float = field(default_factory=time.time)
  renewed_at: float = field(default_factory=time.time)
  ttl_seconds: float = 900.0
  owner: str = "runner"

  def expired(self, now: Optional[float] = None) -> bool:
    return (now or time.time()) - self.renewed_at > self.ttl_seconds

  def holder_is_gone(self) -> bool:
    """True when the holding process was on this machine and has died.

    A TTL alone means a crashed trainer keeps its slot for the whole timeout.
    When the lease was taken on this host the answer is available immediately,
    and it is the same check :meth:`rlmcp.session.Session.is_alive` uses. On any
    other host this returns False and the TTL remains the only authority.
    A manual reservation has no holding process, so it is never "gone".
    """
    if self.owner == "manual":
      return False
    if not self.pid or self.host != socket.gethostname():
      return False
    try:
      os.kill(self.pid, 0)
    except ProcessLookupError:
      return True
    except PermissionError:
      return False
    except OSError:
      return False
    return False

  def dead(self, now: Optional[float] = None) -> bool:
    """Expired, or held by a process that is demonstrably gone."""
    return self.expired(now) or self.holder_is_gone()

  def age_seconds(self, now: Optional[float] = None) -> float:
    return (now or time.time()) - self.acquired_at

  def to_dict(self) -> Dict[str, Any]:
    return {
        "slot": self.slot,
        "holder": self.holder,
        "host": self.host,
        "pid": self.pid,
        "acquired_at": self.acquired_at,
        "renewed_at": self.renewed_at,
        "ttl_seconds": self.ttl_seconds,
        "owner": self.owner,
    }

  @staticmethod
  def from_dict(d: Dict[str, Any]) -> "Lease":
    return Lease(
        slot=d["slot"],
        holder=d.get("holder", ""),
        host=d.get("host", socket.gethostname()),
        pid=int(d.get("pid", 0)),
        acquired_at=d.get("acquired_at", time.time()),
        renewed_at=d.get("renewed_at", d.get("acquired_at", time.time())),
        ttl_seconds=d.get("ttl_seconds", 900.0),
        owner=d.get("owner", "runner"),
    )


@dataclass
class RunRecord:
  """One experiment.

  ``id`` and ``seq`` are assigned by the store, never chosen by the caller: two
  jobs on two machines would otherwise pick the same number.
  """

  id: str
  slug: str
  seq: int = 0
  stage: str = "default"
  task: str = ""
  """Which problem the run was working on -- the environment id it trained on.

  A store accumulates unrelated problems, and runs from two of them share no
  lineage and no meaning: drawing them in one tree invents a relationship that
  does not exist. This is the field a view splits on.

  Filled in automatically by :meth:`rlmcp.records.link.RecordLink.start` from
  the live session's task, because a field only a flag can set is empty on most
  records. Empty means unknown -- every record written before this field existed
  loads that way, and those runs are grouped as such rather than dropped.
  """
  date: str = field(default_factory=_utc_today)
  verdict: str = "planned"

  # Written before the run.
  headline: str = ""
  """One sentence, for a reader who will not open the record.

  Optional: :meth:`one_line` falls back to the first sentence of ``outcome``
  (or ``hypothesis`` while the run is still open), so every run has a summary
  whether or not anyone wrote one. Set it when the derived sentence buries the
  point -- an outcome that opens with a metric dump, say.
  """
  hypothesis: str = ""
  prediction: str = ""
  falsifier: Falsifier = field(default_factory=Falsifier)
  change: List[str] = field(default_factory=list)

  # Written after the run.
  outcome: str = ""
  metrics: List[List[str]] = field(default_factory=list)
  """Free ``[name, value]`` string pairs.

  Deliberately unstructured: this is the slot that has to hold a number, a fired
  falsifier, and "9.5 h of GPU with no run record" equally well.
  """

  feedback: List[Feedback] = field(default_factory=list)
  """What humans said about this run, oldest first. Append-only.

  Written whenever it arrives rather than at either of the two write points:
  feedback lands mid-run, and postponing it to the close-out is how it gets
  lost. Position is identity -- see :class:`Feedback`.
  """

  # Lineage.
  parent: Optional[str] = None
  """Config lineage. The recipe is the fold of ``change`` from the root."""
  weights: Optional[Weights] = None
  """Warm start, or None for from scratch. A different edge from ``parent``."""
  prior: Optional[str] = None
  """Resource dependency -- a motion prior, a pretrained encoder."""

  # Provenance.
  proposed_by: str = "human"
  """``human``, ``orchestrator``, or the name of the policy that proposed it."""
  session: Optional[str] = None
  """The rlmcp session directory, so metrics and events stay recoverable."""
  config: Dict[str, Any] = field(default_factory=dict)
  """Resolved parameter snapshot at launch -- not the source file."""
  links: Dict[str, str] = field(default_factory=dict)
  assets: Dict[str, List[List[str]]] = field(default_factory=dict)
  """``{"videos": [[path, caption]], "plots": [[path, caption]]}``."""
  lease: Optional[Lease] = None
  tags: List[str] = field(default_factory=list)

  created_at: float = field(default_factory=time.time)
  updated_at: float = field(default_factory=time.time)
  schema_version: int = SCHEMA_VERSION
  version: int = 0
  """Write counter, bumped by the store on every persist.

  This is what makes a lost update detectable: :meth:`FileStore.put_record`
  compares it against the file before overwriting. Records written before this
  field existed load as 0 and join the protocol on their first write.
  """

  # Derived.

  @property
  def is_open(self) -> bool:
    return self.verdict in OPEN_VERDICTS

  @property
  def from_scratch(self) -> bool:
    return self.weights is None

  def display(self) -> str:
    return f"{self.id}-{self.slug}"

  def one_line(self, limit: int = 200) -> str:
    """The written headline, or the first sentence of what the run said.

    ``headline`` wins when somebody set it, because a written sentence is the
    record and a derived one is only a reading of it. Otherwise a closed run is
    summarised by its outcome and an open one by its hypothesis, since that is
    the most recent claim each has made. This is what a tile in the story view
    or a label on a node shows: an id and a slug say which run it is, and this
    says why anyone should care.
    """
    if self.headline.strip():
      return self.headline.strip()
    source = (self.outcome or self.hypothesis or "").strip()
    if not source:
      return ""
    # Sentence-ish: a period followed by a space, but not the one inside
    # "0.25" or "it3000." at a line end that has no successor.
    match = re.search(r"(?<=[.!?])\s+(?=[A-Z(])", source)
    sentence = source[: match.start()] if match else source
    sentence = " ".join(sentence.split())
    if len(sentence) > limit:
      sentence = sentence[: limit - 1].rsplit(" ", 1)[0] + "…"
    return sentence

  def outstanding_feedback(self) -> List[Tuple[int, "Feedback"]]:
    """Instructions with no recorded response, with their indices.

    The index travels with the entry because it is how a response is later
    attached (``store.answer_feedback(run, index, ...)``).
    """
    return [(i, f) for i, f in enumerate(self.feedback) if f.outstanding]

  def feedback_kinds(self) -> Dict[str, int]:
    """``{kind: count}``, for a badge that says what kind without opening it."""
    counts: Dict[str, int] = {}
    for entry in self.feedback:
      counts[entry.kind] = counts.get(entry.kind, 0) + 1
    return counts

  def summary(self) -> Dict[str, Any]:
    """The row form, for listings and index tables.

    ``headline`` here is the *display* sentence -- :meth:`one_line`, so written
    when there is one and derived when there is not. The raw field is in
    :meth:`to_dict`, which is where a caller that must tell the two apart looks.
    """
    return {
        "id": self.id,
        "slug": self.slug,
        "seq": self.seq,
        "stage": self.stage,
        "task": self.task,
        "date": self.date,
        "verdict": self.verdict,
        "parent": self.parent,
        "weights": self.weights.run if self.weights else None,
        "from_scratch": self.from_scratch,
        "proposed_by": self.proposed_by,
        "headline": self.one_line(),
        "hypothesis": self.hypothesis,
        "outcome": self.outcome,
        "feedback": len(self.feedback),
        "feedback_outstanding": len(self.outstanding_feedback()),
    }

  def touch(self) -> None:
    self.updated_at = time.time()

  # Serialisation.

  def to_dict(self) -> Dict[str, Any]:
    return {
        "schema_version": self.schema_version,
        "id": self.id,
        "slug": self.slug,
        "seq": self.seq,
        "stage": self.stage,
        "task": self.task,
        "date": self.date,
        "verdict": self.verdict,
        "headline": self.headline,
        "hypothesis": self.hypothesis,
        "prediction": self.prediction,
        "falsifier": self.falsifier.to_dict(),
        "change": list(self.change),
        "outcome": self.outcome,
        "metrics": [list(pair) for pair in self.metrics],
        "feedback": [f.to_dict() for f in self.feedback],
        "parent": self.parent,
        "weights": self.weights.to_dict() if self.weights else None,
        "prior": self.prior,
        "proposed_by": self.proposed_by,
        "session": self.session,
        "config": self.config,
        "links": self.links,
        "assets": self.assets,
        "lease": self.lease.to_dict() if self.lease else None,
        "tags": list(self.tags),
        "created_at": self.created_at,
        "updated_at": self.updated_at,
        "version": self.version,
    }

  @staticmethod
  def from_dict(d: Dict[str, Any]) -> "RunRecord":
    weights = d.get("weights")
    lease = d.get("lease")
    return RunRecord(
        id=d["id"],
        slug=d.get("slug", "run"),
        seq=int(d.get("seq", 0)),
        stage=d.get("stage", "default"),
        task=d.get("task") or "",
        date=d.get("date", _utc_today()),
        verdict=d.get("verdict", "planned"),
        headline=d.get("headline", ""),
        hypothesis=d.get("hypothesis", ""),
        prediction=d.get("prediction", ""),
        falsifier=Falsifier.from_dict(d.get("falsifier")),
        change=list(d.get("change") or []),
        outcome=d.get("outcome", ""),
        metrics=[list(pair) for pair in (d.get("metrics") or [])],
        feedback=[Feedback.from_dict(f) for f in (d.get("feedback") or [])],
        parent=d.get("parent"),
        weights=Weights.from_dict(weights) if weights else None,
        prior=d.get("prior"),
        proposed_by=d.get("proposed_by", "human"),
        session=d.get("session"),
        config=dict(d.get("config") or {}),
        links=dict(d.get("links") or {}),
        assets=dict(d.get("assets") or {}),
        lease=Lease.from_dict(lease) if lease else None,
        tags=list(d.get("tags") or []),
        created_at=d.get("created_at", time.time()),
        updated_at=d.get("updated_at", time.time()),
        schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
        version=int(d.get("version", 0)),
    )


def fold_recipe(
    record_id: str, records: Dict[str, RunRecord]
) -> List[Tuple[str, List[str]]]:
  """The recipe at a node: every ``change`` from the root down to it.

  Nothing stores this. Walking the config edges is what keeps provenance and the
  current recipe from drifting apart -- the property the whole schema exists for.

  Raises:
    ValueError: if the ``parent`` chain contains a cycle.
  """
  chain: List[Tuple[str, List[str]]] = []
  seen: List[str] = []
  current = records.get(record_id)
  while current is not None:
    if current.id in seen:
      raise ValueError(
          f"Cycle in the parent chain at '{current.id}': {' -> '.join(seen)}"
      )
    seen.append(current.id)
    chain.append((current.id, list(current.change)))
    current = records.get(current.parent) if current.parent else None
  chain.reverse()
  return chain


def ancestors(record_id: str, records: Dict[str, RunRecord]) -> List[str]:
  """Ids from the root down to (and including) ``record_id``."""
  return [rid for rid, _ in fold_recipe(record_id, records)]


def children(record_id: str, records: Sequence[RunRecord]) -> List[RunRecord]:
  """Records whose config parent is ``record_id``."""
  return [r for r in records if r.parent == record_id]
