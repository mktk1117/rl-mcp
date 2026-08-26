"""Laying the tree out, and folding the recipe along it.

Two edges, two meanings, and the layout has to keep them apart:

* ``parent`` is the **config** edge. The recipe at any node is the fold of every
  ``change`` from the root down to it, which is why the recipe is never stored.
* ``weights`` is the **warm-start** edge. It says a run inherited a policy, and
  it frequently points somewhere other than the config parent -- a run can take
  its config from its predecessor and its weights from six runs back.

The method this ports draws the warm-start edge from ``parent`` rather than from
``weights.run``, so a record with a config parent of 032 and weights from 011 is
drawn wrong. Here they are separate edges, because they are separate claims.

Lanes are assigned the way ``git log --graph`` does: a node keeps its parent's
lane when it is the first child and takes a free one otherwise, so a fork is
visible as a branch rather than as a longer list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rlmcp.records.record import RunRecord, fold_recipe

VERDICT_ORDER = (
    "best", "validated", "provisional", "running", "planned",
    "control", "falsified", "superseded", "interrupted",
)


@dataclass
class Node:
  """One record, placed."""

  record: RunRecord
  row: int
  lane: int
  depth: int
  children: List[str] = field(default_factory=list)
  warm_children: List[str] = field(default_factory=list)

  @property
  def id(self) -> str:
    return self.record.id


@dataclass
class Graph:
  """The placed tree, plus the edges to draw."""

  nodes: Dict[str, Node]
  order: List[str]
  config_edges: List[Tuple[str, str]] = field(default_factory=list)
  warm_edges: List[Tuple[str, str]] = field(default_factory=list)
  prior_edges: List[Tuple[str, str]] = field(default_factory=list)

  @property
  def lanes(self) -> int:
    return max((n.lane for n in self.nodes.values()), default=0) + 1

  @property
  def rows(self) -> int:
    return len(self.order)

  def recipe(self, record_id: str) -> List[Tuple[str, List[str]]]:
    """The fold of changes from the root to ``record_id``."""
    return fold_recipe(record_id, {k: n.record for k, n in self.nodes.items()})


def build(records: Sequence[RunRecord]) -> Graph:
  """Place records into rows and lanes, and collect the three edge kinds."""
  ordered = sorted(records, key=lambda r: (r.seq, r.id))
  by_id = {r.id: r for r in ordered}

  depths: Dict[str, int] = {}

  def depth_of(record: RunRecord, seen: Optional[set] = None) -> int:
    seen = seen or set()
    if record.id in depths:
      return depths[record.id]
    if record.id in seen or not record.parent or record.parent not in by_id:
      depths[record.id] = 0
      return 0
    seen.add(record.id)
    depths[record.id] = depth_of(by_id[record.parent], seen) + 1
    return depths[record.id]

  # Lane packing, close to what git log --graph does.
  #
  # A lane is held while any node placed in it still has unplaced children, and
  # released the moment the last one lands. Two rules together give a readable
  # tree: the first child continues its parent's lane, and any later sibling
  # forks into a lane that nothing else is still using. Without the release step
  # a long linear history drifts diagonally forever; without the first-child
  # rule every run forks.
  pending: Dict[str, int] = {r.id: 0 for r in ordered}
  for record in ordered:
    if record.parent in pending:
      pending[record.parent] += 1

  nodes: Dict[str, Node] = {}
  lane_of: Dict[str, int] = {}
  lane_debt: Dict[int, set] = {}
  continued: set = set()

  def free_lane() -> int:
    lane = 0
    while lane_debt.get(lane):
      lane += 1
    return lane

  for row, record in enumerate(ordered):
    parent = record.parent if record.parent in lane_of else None
    if parent is not None and parent not in continued and pending.get(parent, 0) > 0:
      lane = lane_of[parent]
      continued.add(parent)
    else:
      lane = free_lane()

    if parent is not None:
      pending[parent] -= 1
      if pending[parent] <= 0:
        lane_debt.get(lane_of[parent], set()).discard(parent)

    lane_of[record.id] = lane
    if pending.get(record.id, 0) > 0:
      lane_debt.setdefault(lane, set()).add(record.id)
    nodes[record.id] = Node(record=record, row=row, lane=lane, depth=depth_of(record))

  graph = Graph(nodes=nodes, order=[r.id for r in ordered])

  for record in ordered:
    if record.parent and record.parent in nodes:
      graph.config_edges.append((record.parent, record.id))
      nodes[record.parent].children.append(record.id)
    # The warm-start edge starts at the run whose weights were taken, which is
    # not always the config parent.
    if record.weights and record.weights.run in nodes:
      graph.warm_edges.append((record.weights.run, record.id))
      nodes[record.weights.run].warm_children.append(record.id)
    if record.prior and record.prior in nodes:
      graph.prior_edges.append((record.prior, record.id))

  return graph


def summarize(graph: Graph) -> Dict[str, Any]:
  """Counts an agent can read without rendering anything."""
  counts: Dict[str, int] = {}
  for node in graph.nodes.values():
    counts[node.record.verdict] = counts.get(node.record.verdict, 0) + 1
  roots = [i for i, n in graph.nodes.items() if not n.record.parent]
  warm = [i for i, n in graph.nodes.items() if n.record.weights is not None]
  leaves = [i for i, n in graph.nodes.items() if not n.children]
  return {
      "records": len(graph.nodes),
      "verdicts": {v: counts[v] for v in VERDICT_ORDER if v in counts},
      "roots": roots,
      "leaves": leaves,
      "warm_started": len(warm),
      "from_scratch": len(graph.nodes) - len(warm),
      "config_edges": len(graph.config_edges),
      "warm_edges": len(graph.warm_edges),
      "lanes": graph.lanes,
  }


def frontier(graph: Graph, max_children: int = 1) -> List[str]:
  """Nodes worth expanding: a result, and not yet branched from.

  The query an orchestrator asks when it is choosing where to spend the next
  GPU-hour. Kept here rather than in the store because it is a property of the
  tree, not of how the tree is persisted.
  """
  out = []
  for node_id, node in graph.nodes.items():
    verdict = node.record.verdict
    if verdict in ("validated", "provisional", "best") and len(node.children) < max_children:
      out.append(node_id)
  return sorted(out, key=lambda i: graph.nodes[i].row)


def headline_video(record: RunRecord) -> Optional[List[str]]:
  """The clip that stands for the run: the last video attached to it.

  Last rather than first because attachments arrive in time order, so the last
  one is the most recent look at the policy -- which is the one a reader of the
  tree wants. Everything that shows a run as a picture agrees to ask here, so
  the node, the story tile and the poster cache cannot pick different clips.
  """
  videos = (record.assets or {}).get("videos") or []
  return list(videos[-1]) if videos else None


def _clip_of(record: RunRecord,
             posters: Optional[Dict[str, str]] = None) -> Optional[Dict[str, str]]:
  """The headline clip with its still, or None for a run nobody filmed.

  ``posters`` maps a video key to a poster key -- see
  :func:`rlmcp.records.poster.ensure_posters`. A record may also carry the
  poster itself as a third element of the asset entry, which wins: a still
  written down at attach time is a fact about the record, and a derived one is
  a cache.
  """
  entry = headline_video(record)
  if not entry:
    return None
  src = entry[0] if entry else ""
  poster = entry[2] if len(entry) > 2 and entry[2] else (posters or {}).get(src, "")
  return {
      "src": src,
      "caption": entry[1] if len(entry) > 1 else "",
      "poster": poster,
  }


def to_payload(graph: Graph,
               posters: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
  """Everything a viewer needs, with the derived bits derived here."""
  payload = []
  for node_id in graph.order:
    node = graph.nodes[node_id]
    record = node.record
    try:
      recipe = [{"id": rid, "change": ch} for rid, ch in graph.recipe(node_id)]
    except ValueError as exc:
      recipe = [{"id": node_id, "change": [f"(lineage error: {exc})"]}]
    payload.append(
        {
            "id": record.id,
            "slug": record.slug,
            "stage": record.stage,
            "date": record.date,
            "verdict": record.verdict,
            "row": node.row,
            "lane": node.lane,
            "depth": node.depth,
            "parent": record.parent,
            "warm": record.weights.describe() if record.weights else None,
            "warm_run": record.weights.run if record.weights else None,
            "prior": record.prior,
            "scratch": record.weights is None,
            "proposed_by": record.proposed_by,
            "summary": record.one_line(),
            # Where the sentence came from, because "what this run found" and
            # "what this run set out to find" are different claims and the
            # story view must not let them read as the same one. A written
            # headline is a third source: it is the record rather than a
            # reading of it, and ``one_line`` prefers it.
            "summary_source": ("headline" if record.headline.strip()
                               else "outcome" if record.outcome.strip()
                               else "hypothesis" if record.hypothesis.strip()
                               else ""),
            "hypothesis": record.hypothesis,
            "prediction": record.prediction,
            "falsifier": record.falsifier.prose,
            "falsifier_conditions": [c.describe() for c in record.falsifier.conditions],
            "change": list(record.change),
            "outcome": record.outcome,
            "metrics": [list(m) for m in record.metrics],
            "multi": len(record.change) > 1
                     and record.verdict not in ("planned", "superseded"),
            "recipe": recipe,
            "session": record.session,
            "assets": record.assets,
            # The clip that stands for the run, hoisted out of assets so a
            # node, a story tile and a thumbnail do not each re-derive it.
            "clip": _clip_of(record, posters),
        }
    )
  return payload
