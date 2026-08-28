"""Laying out the tree, and the two edges that mean different things."""

from __future__ import annotations

import json

import pytest

from rlmcp.records.graph import build, frontier, summarize, to_payload
from rlmcp.records.record import RunRecord, Weights
from rlmcp.records.views import (
  plot_records,
  plot_run_comparison,
  render_records_html,
  vendored,
)

ENGINES = ["simple"] + (["cytoscape"] if vendored() else [])
"""Both renderers must escape identically; cytoscape only when it is vendored."""


def _assert_nothing_is_fetched(page: str) -> None:
  """No element loads from a network.

  Checked at the attribute level rather than by scanning for "http", because the
  vendored libraries carry MIT licence URLs in their comments -- a substring
  match calls those a network dependency, which they are not.
  """
  for pattern in ('src="http', "src='http", 'src="//', 'href="http', "href='http",
                  'href="//', "@import", "importScripts("):
    assert pattern not in page, f"page would fetch: {pattern}"


def _r(rid: str, seq: int, parent=None, weights=None, verdict="provisional", **kw):
  return RunRecord(id=rid, slug=f"run_{rid}", seq=seq, parent=parent,
                   weights=weights, verdict=verdict,
                   change=kw.pop("change", [f"change {rid}"]), **kw)


def test_a_linear_history_stays_in_one_lane():
  """Without releasing lanes a long chain drifts diagonally off the page."""
  records = [_r("001", 1)]
  records += [_r(f"{i:03d}", i, parent=f"{i-1:03d}") for i in range(2, 12)]

  graph = build(records)

  assert graph.lanes == 1
  assert [graph.nodes[r.id].lane for r in records] == [0] * 11


def test_a_fork_opens_a_second_lane_and_closes_it_again():
  records = [
      _r("001", 1),
      _r("002", 2, parent="001"),   # continues the branch
      _r("003", 3, parent="001"),   # forks
      _r("004", 4, parent="003"),   # continues the fork
  ]

  graph = build(records)
  lanes = {r: graph.nodes[r].lane for r in ("001", "002", "003", "004")}

  assert lanes["001"] == lanes["002"] == 0
  assert lanes["003"] == lanes["004"] == 1
  assert graph.lanes == 2


def test_rows_follow_sequence_not_insertion():
  graph = build([_r("002", 2), _r("001", 1)])
  assert graph.order == ["001", "002"]
  assert graph.nodes["001"].row == 0


def test_the_warm_edge_starts_where_the_weights_came_from():
  """The method this ports draws it from the config parent, which is wrong."""
  records = [
      _r("011", 1),
      _r("032", 2, parent="011"),
      _r("034", 3, parent="032", weights=Weights("011", "model_17999.pt")),
  ]

  graph = build(records)

  assert ("032", "034") in graph.config_edges
  assert ("011", "034") in graph.warm_edges
  assert ("032", "034") not in graph.warm_edges


def test_depth_counts_config_edges_only():
  records = [
      _r("001", 1),
      _r("002", 2, parent="001"),
      _r("003", 3, parent="002", weights=Weights("001")),
  ]
  graph = build(records)
  assert [graph.nodes[i].depth for i in ("001", "002", "003")] == [0, 1, 2]


def test_the_recipe_is_the_fold_along_the_config_edges():
  records = [
      _r("001", 1, change=["baseline"]),
      _r("002", 2, parent="001", change=["add slip penalty"]),
      _r("003", 3, parent="002", weights=Weights("001"), change=["raise clearance"]),
  ]
  graph = build(records)

  assert graph.recipe("003") == [
      ("001", ["baseline"]),
      ("002", ["add slip penalty"]),
      ("003", ["raise clearance"]),
  ]


def test_summary_counts_what_a_reader_asks_first():
  records = [
      _r("001", 1, verdict="validated"),
      _r("002", 2, parent="001", weights=Weights("001"), verdict="provisional"),
      _r("003", 3, parent="001", verdict="falsified"),
  ]

  stats = summarize(build(records))

  assert stats["records"] == 3
  assert stats["from_scratch"] == 2 and stats["warm_started"] == 1
  assert stats["roots"] == ["001"]
  assert set(stats["leaves"]) == {"002", "003"}
  assert stats["verdicts"] == {"validated": 1, "provisional": 1, "falsified": 1}


def test_frontier_is_where_an_orchestrator_would_spend_the_next_hour():
  records = [
      _r("001", 1, verdict="validated"),                  # already branched
      _r("002", 2, parent="001", verdict="provisional"),  # a leaf with a result
      _r("003", 3, parent="001", verdict="falsified"),    # dead
      _r("004", 4, verdict="running"),                    # no result yet
  ]

  assert frontier(build(records)) == ["002"]


def test_payload_flags_a_multi_change_result_but_not_a_plan():
  records = [
      _r("001", 1, verdict="provisional", change=["a", "b"]),
      _r("002", 2, verdict="planned", change=["a", "b"]),
  ]
  payload = {p["id"]: p for p in to_payload(build(records))}

  assert payload["001"]["multi"] is True
  assert payload["002"]["multi"] is False


def test_a_cycle_does_not_hang_the_payload():
  records = [_r("001", 1, parent="002"), _r("002", 2, parent="001")]
  payload = to_payload(build(records))
  assert any("graph error" in str(p["recipe"]) for p in payload)


# Rendering.


def test_the_png_renders():
  png = plot_records([_r("001", 1), _r("002", 2, parent="001",
                                       weights=Weights("001"))])
  assert png[:4] == b"\x89PNG"


def test_the_png_handles_an_empty_records():
  assert plot_records([])[:4] == b"\x89PNG"


def test_the_viewer_is_one_self_contained_file():
  records = [
      _r("001", 1, verdict="validated", hypothesis="flat first"),
      _r("002", 2, parent="001", weights=Weights("001", "model_9.pt"),
         verdict="provisional"),
  ]

  page = render_records_html(records, title="tree")

  assert page.startswith("<!doctype html>")
  _assert_nothing_is_fetched(page)
  assert "flat first" in page
  assert "model_9.pt" in page
  assert "the fold of" in page  # the recipe note
  assert "__RUNS__" not in page and "__MEDIA__" not in page  # all substituted


def test_media_is_referenced_relatively_so_the_page_works_from_a_file_path():
  """A result that has not been watched has not been verified -- so it has to play."""
  record = _r("001", 1, verdict="provisional")
  record.assets = {
      "videos": [["001/videos/tour.mp4", "3-seed tour"]],
      "plots": [["001/plots/curves.png", "training curves"]],
  }

  page = render_records_html([record], media_base="media/")

  assert '"media/"' in page  # the base the page resolves against
  assert "001/videos/tour.mp4" in page
  assert "001/plots/curves.png" in page
  assert "<video" in page and "lightbox" in page


def test_the_interactive_engine_inlines_its_libraries():
  """Vendored, not fetched -- the page has to work offline and under a CSP."""
  from rlmcp.records.views import vendored

  if not vendored():
    pytest.skip("graph libraries are not vendored in this checkout")

  page = render_records_html([_r("001", 1)], engine="cytoscape")

  assert "cytoscape.use(cytoscapeDagre)" in page
  _assert_nothing_is_fetched(page)
  assert len(page) > 500_000  # the libraries really are in there


def test_the_simple_engine_needs_no_vendored_libraries():
  page = render_records_html([_r("001", 1)], engine="simple")
  assert "cytoscape" not in page
  assert len(page) < 200_000


def test_auto_picks_the_interactive_engine_when_available():
  from rlmcp.records.views import vendored

  page = render_records_html([_r("001", 1)], engine="auto")
  assert ("cytoscape" in page) == vendored()


def test_a_run_with_no_media_renders_no_player():
  page = render_records_html([_r("001", 1)])
  assert "<video" not in page.split("const RUNS=")[0]


@pytest.mark.parametrize("engine", ENGINES)
def test_the_viewer_escapes_record_text(engine):
  """The whole page. The old form checked page.split("const RUNS=")[0] -- the
  half that never contained the payload, which is where the bug lived."""
  page = render_records_html([_r("001", 1, hypothesis="<script>alert(1)</script>")],
                             engine=engine)
  assert "<script>alert(1)</script>" not in page


@pytest.mark.parametrize("engine", ENGINES)
def test_a_hostile_hypothesis_cannot_close_the_script_element(engine):
  """json.dumps leaves "</" alone and the HTML parser does not: an unescaped
  "</script>" inside the payload ends the script element right there and hands
  the rest of the text to the parser as markup. Every "<" in the embedded JSON
  is serialised as \\u003c, so the raw byte sequence cannot occur at all."""
  evil = "</script><img src=x onerror=alert(1)>"
  page = render_records_html([_r("001", 1, hypothesis=evil)], engine=engine)

  assert evil not in page
  assert "\\u003c/script>\\u003cimg src=x onerror=alert(1)>" in page
  # Script opens and closes balance: the payload contributes to neither count.
  # (The vendored libraries contain no "<script" of their own.)
  assert page.count("</script>") == page.count("<script")

  # The escape is content-preserving: the embedded array is still valid JSON
  # and the hypothesis comes back byte-identical.
  start = page.index("const RUNS=") + len("const RUNS=")
  runs = json.loads(page[start:page.index(", EDGES=", start)])
  assert runs[0]["hypothesis"] == evil


@pytest.mark.parametrize("engine", ENGINES)
def test_hostile_ids_slugs_verdicts_and_asset_keys_stay_escaped(engine):
  """Ids, slugs, verdicts and asset paths are attacker-reachable through
  ``lab import`` and a hand-edited meta.json, and the card/panel used to
  interpolate them into innerHTML raw.

  No browser runs here, so the pin is textual and twofold: (1) the hostile
  bytes appear in the page only inside the JSON payload, "<" escaped as
  \\u003c, never raw; (2) the card/panel HTML is built by in-page JS from that
  payload, so escaping must happen at render time -- assert the script source
  wraps every record-sourced interpolation in esc(...), and media paths in
  esc(mediaUrl(...)). Grep-style, deliberately: it is the honest check
  available without executing the page.
  """
  record = _r("001", 1)
  record.id = "<img src=x onerror=alert(1)>"
  record.slug = "s</b><svg onload=alert(2)>"
  record.verdict = "<script>alert(3)</script>"
  record.assets = {
      "videos": [["<video onplay=alert(4)>/clip.mp4", "cap"]],
      "plots": [["<svg onload=alert(5)>/p.png", "curves"]],
  }
  page = render_records_html([record], engine=engine)

  for raw in ("<img src=x onerror=alert(1)>", "</b><svg onload=alert(2)>",
              "<script>alert(3)", "<video onplay=alert(4)>",
              "<svg onload=alert(5)>"):
    assert raw not in page, f"raw hostile bytes reached the page: {raw}"
  assert "\\u003cimg src=x onerror=alert(1)>" in page  # present, but as data

  # Detail panel: id and verdict join the already-escaped free-text fields.
  assert "${esc(r.id)}" in page and "${esc(r.verdict)}" in page
  assert "${esc(r.slug)}" in page
  assert "<h2>${r.id}" not in page and ">${r.verdict}<" not in page
  # Verdict chips are record-sourced too.
  assert '${esc(v)}<span class="n">' in page
  # Media paths: URL-encode each segment (slashes survive, quotes and any
  # "javascript:" colon do not), then attribute-escape the result.
  assert 'src="${esc(mediaUrl(vids[0][0]))}"' in page
  assert 'data-src="${esc(mediaUrl(v[0]))}"' in page
  assert 'src="${esc(mediaUrl(pl[0]))}"' in page
  assert "${MEDIA}${" not in page  # no raw path concatenation is left
  if engine == "simple":  # the node card; cytoscape draws labels on canvas
    assert "<b>${esc(r.id)} ${esc(r.slug)}</b>" in page


@pytest.mark.parametrize("engine", ENGINES)
def test_free_text_containing_a_placeholder_is_data_not_a_template_hole(engine):
  """Chained .replace() calls re-scanned earlier output, so a hypothesis that
  contained __STATS__ got the stats JSON spliced into the payload. One-pass
  substitution: the text survives verbatim, the real slot is still filled."""
  evil = "watch __STATS__ and __EDGES__ and __COLORS__ closely"
  page = render_records_html([_r("001", 1, hypothesis=evil)], engine=engine)

  assert json.dumps(evil) in page           # verbatim, still one JSON string
  assert 'STATS={"records": 1' in page      # the template slot got the stats
  assert 'COLORS={"best":' in page          # and the palette its palette


@pytest.mark.parametrize("engine", ENGINES)
def test_an_empty_lab_still_renders_a_working_page(engine):
  """render_records_html([]) calls select(undefined) on load; without a guard
  the simple engine threw before drawing anything. Not executable here, so the
  pin is structural: select() bails out before its first use of the record."""
  page = render_records_html([], engine=engine)

  assert "const RUNS=[]" in page
  body = page[page.index("function select("):]
  assert "const r=byId[id]; if(!r) return;" in body
  assert body.index("if(!r) return") < body.index("r.")


def test_comparison_truncates_to_a_matched_iteration():
  series = {
      "001": {"reward": [[0, 1.0], [100, 5.0], [200, 9.0]]},
      "002": {"reward": [[0, 1.0], [100, 4.0]]},
  }
  assert plot_run_comparison(series, ["reward"], at_iteration=100)[:4] == b"\x89PNG"


def test_comparison_says_so_when_the_runs_share_no_metric():
  series = {"001": {"a": [[0, 1.0]]}, "002": {"b": [[0, 1.0]]}}
  assert plot_run_comparison(series, ["c"])[:4] == b"\x89PNG"


# The three views: the tree is the structure, the story is the order, the
# parameters are the numbers. All three ship in the one interactive page.


def _cytoscape_or_skip(records, **kw):
  if not vendored():
    pytest.skip("graph libraries are not vendored in this checkout")
  return render_records_html(records, engine="cytoscape", **kw)


def test_the_headline_clip_is_the_last_video_attached():
  """Attachments arrive in time order, so the last one is the most recent look
  at the policy -- which is the one a reader of the tree wants."""
  record = _r("001", 1)
  record.assets = {"videos": [["001/videos/early.mp4", "at 500"],
                              ["001/videos/late.mp4", "at 4000"]]}

  clip = to_payload(build([record]))[0]["clip"]

  assert clip["src"] == "001/videos/late.mp4"
  assert clip["caption"] == "at 4000"
  assert clip["poster"] == ""  # nothing derived one


def test_a_run_nobody_filmed_has_no_clip_rather_than_an_empty_one():
  """Most runs carry no video. The story tile, the node and the strip all key
  off clip being falsy, so a stub with empty strings would draw holes."""
  record = _r("001", 1)
  record.assets = {"plots": [["001/plots/curves.png", "curves"]]}

  assert to_payload(build([record]))[0]["clip"] is None


def test_a_derived_poster_reaches_the_payload_and_a_written_one_wins():
  """A still cached by the renderer is a cache; one written into the record at
  attach time is a fact about the record, and facts beat caches."""
  derived, written = _r("001", 1), _r("002", 2)
  derived.assets = {"videos": [["001/videos/tour.mp4", "tour"]]}
  written.assets = {"videos": [["002/videos/tour.mp4", "tour",
                                "002/posters/written.png"]]}
  posters = {"001/videos/tour.mp4": "001/posters/derived.png",
             "002/videos/tour.mp4": "002/posters/ignored.png"}

  payload = to_payload(build([derived, written]), posters)

  assert payload[0]["clip"]["poster"] == "001/posters/derived.png"
  assert payload[1]["clip"]["poster"] == "002/posters/written.png"


def test_the_summary_is_the_outcome_while_there_is_one_and_the_hypothesis_before():
  """The story tile shows one sentence per run, and which claim it is matters:
  what a run found and what it set out to find are not the same statement."""
  closed = _r("001", 1, outcome="The failure rate dropped by a third. It also ran faster.",
              hypothesis="A smaller penalty keeps the behaviour.")
  open_ = _r("002", 2, verdict="running", hypothesis="Halving the step size settles it.")

  payload = to_payload(build([closed, open_]))

  assert payload[0]["summary"] == "The failure rate dropped by a third."
  assert payload[0]["summary_source"] == "outcome"
  assert payload[1]["summary"] == "Halving the step size settles it."
  assert payload[1]["summary_source"] == "hypothesis"


def test_the_interactive_page_carries_all_three_views():
  page = _cytoscape_or_skip([_r("001", 1)])

  assert '[["tree","tree"],["story","story"],["params","parameters"]]' in page
  assert 'id="storyflow"' in page and "function drawStory(" in page
  assert 'id="params"' in page and "function drawChart(" in page
  assert "function drawBar(" in page and "function drawPanel(" in page
  assert "function drawStrip(" in page
  # The seam between graph and record, and the thumbnail toggle above it.
  assert 'id="grip"' in page and "cursor:col-resize" in page
  assert "body.resizing" in page
  assert 'id="thumbs"' in page and "function applyThumbs(" in page


def test_the_parameter_payload_travels_with_the_page():
  page = _cytoscape_or_skip([_r("001", 1)])

  assert "const PARAMS=" in page and ", PATHS=" in page
  assert "__PARAMS__" not in page and "__PATHS__" not in page


def test_thumbnails_are_bigger_than_dots_and_only_when_there_is_a_still():
  """A node with a still is worth four times the area of one without; a node
  with nothing to show stays a dot rather than a frame around emptiness."""
  page = _cytoscape_or_skip([_r("001", 1)])

  assert 'ele=>ele.data("poster")?76:15' in page
  assert 'ele=>ele.data("poster")?57:15' in page
  assert '"background-image":ele=>ele.data("poster")||"none"' in page


@pytest.mark.parametrize("engine", ENGINES)
def test_a_hostile_poster_key_is_escaped_like_every_other_media_path(engine):
  record = _r("001", 1)
  record.assets = {"videos": [["001/videos/c.mp4", "cap",
                               '"><svg onload=alert(6)>/p.png']]}

  page = render_records_html([record], engine=engine)

  assert '"><svg onload=alert(6)>' not in page
  if engine == "cytoscape":
    assert 'src="${esc(mediaUrl(clip.poster))}"' in page


def test_the_page_still_draws_when_a_session_directory_is_gone():
  """A records store whose logs were cleaned up loses the traces, not the tree.
  build_history reads every session; a failure there must not take the page."""
  record = _r("001", 1, session="/nowhere/at/all", config={"reward.a.weight": 1.0})

  page = render_records_html([record])

  assert "const PARAMS=" in page
  assert "run_001" in page


def test_a_written_headline_is_named_as_the_summarys_source():
  """`one_line` prefers a written headline, so the payload must say so --
  otherwise a viewer labels the sentence as the run's outcome."""
  record = RunRecord(id="001", slug="summarised",
                     outcome="Reward rose to 4.2.",
                     headline="The plateau is the story.")

  node = to_payload(build([record]))[0]

  assert node["summary"] == "The plateau is the story."
  assert node["summary_source"] == "headline"
