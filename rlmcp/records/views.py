"""Seeing the tree.

Two renderers over the same layout, because the two readers want different
things. An agent wants a picture it can look at in one tool call, so
:func:`plot_lineage` returns PNG bytes. A person wants to click a node and read
what the run claimed, so :func:`render_lineage_html` returns one self-contained
file -- no server, no build step, openable from a path.

Both draw the config edge and the warm-start edge separately, and the HTML view
carries the thing that makes the tree worth keeping: the recipe at a node,
folded from the root, with a note saying it was computed rather than stored.
"""

from __future__ import annotations

import html
import io
import json
import re
from typing import Any, Dict, List, Optional, Sequence

from pathlib import Path

from rlmcp.records.lineage import build, summarize, to_payload
from rlmcp.records.params import build_history, leaf_paths
from rlmcp.records.record import RunRecord

_HERE = Path(__file__).parent
VENDOR = _HERE / "vendor"
TEMPLATES = _HERE / "templates"

VERDICT_COLORS = {
    "best": "#1F77B4",
    "validated": "#2C7A57",
    "provisional": "#9A6314",
    "running": "#1F77B4",
    "planned": "#8A98A6",
    "control": "#6B7FA8",
    "falsified": "#B93E3C",
    "superseded": "#8A98A6",
    "interrupted": "#8A6A3C",
}


def plot_lineage(
    records: Sequence[RunRecord],
    title: Optional[str] = None,
    max_label: int = 30,
) -> bytes:
  """Draw the tree as a PNG.

  Rows are chronological, lanes show branching, node colour is the verdict.
  Config edges are solid; warm-start edges are dashed and drawn from the run the
  weights actually came from.
  """
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.lines import Line2D

  graph = build(records)
  if not graph.order:
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.text(0.5, 0.5, "No records yet", ha="center", va="center")
    ax.axis("off")
    return _png(fig)

  rows, lanes = graph.rows, graph.lanes
  fig, ax = plt.subplots(figsize=(max(6.0, 3.2 + lanes * 1.5), max(3.0, rows * 0.42)))

  def xy(node_id: str):
    node = graph.nodes[node_id]
    return node.lane * 1.0, -node.row * 1.0

  for source, target in graph.config_edges:
    x0, y0 = xy(source)
    x1, y1 = xy(target)
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-", color="#9AA7B5", lw=1.4,
                        connectionstyle="arc3,rad=0.08" if x0 != x1 else "arc3,rad=0"),
        zorder=1,
    )
  for source, target in graph.warm_edges:
    x0, y0 = xy(source)
    x1, y1 = xy(target)
    ax.annotate(
        "", xy=(x1 + 0.06, y1), xytext=(x0 + 0.06, y0),
        arrowprops=dict(arrowstyle="-", color="#C99A3A", lw=1.4, linestyle="--",
                        connectionstyle="arc3,rad=0.18"),
        zorder=1,
    )

  for node_id in graph.order:
    node = graph.nodes[node_id]
    x, y = xy(node_id)
    color = VERDICT_COLORS.get(node.record.verdict, "#8A98A6")
    ax.scatter([x], [y], s=190, c=color, zorder=3,
               edgecolors="white", linewidths=1.5,
               marker="o" if node.record.weights is None else "s")
    label = f"{node.record.id} {node.record.slug}"
    if len(label) > max_label:
      label = label[: max_label - 1] + "…"
    ax.text(x + 0.16, y, label, va="center", fontsize=8.5, zorder=4)
    ax.text(x + 0.16, y - 0.34, node.record.verdict, va="center", fontsize=7,
            color=color, zorder=4)

  ax.set_xlim(-0.5, lanes - 0.5 + 3.2)
  ax.set_ylim(-rows + 0.4, 0.8)
  ax.axis("off")
  ax.legend(
      handles=[
          Line2D([], [], color="#9AA7B5", lw=1.4, label="config lineage (parent)"),
          Line2D([], [], color="#C99A3A", lw=1.4, ls="--", label="warm start (weights)"),
          Line2D([], [], marker="o", color="none", markerfacecolor="#5B6A79",
                 markersize=8, label="from scratch"),
          Line2D([], [], marker="s", color="none", markerfacecolor="#5B6A79",
                 markersize=8, label="warm started"),
      ],
      loc="lower right", fontsize=7.5, framealpha=0.9,
  )
  ax.set_title(title or "run lineage", fontsize=11, fontweight="bold", loc="left")
  fig.tight_layout()
  return _png(fig)


def _png(fig: Any, dpi: int = 120) -> bytes:
  import matplotlib.pyplot as plt

  buf = io.BytesIO()
  fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
  plt.close(fig)
  return buf.getvalue()


def vendored() -> bool:
  """Whether the graph libraries are present to inline."""
  return all((VENDOR / name).exists() for name in
             ("cytoscape.min.js", "dagre.min.js", "cytoscape-dagre.js"))


def _script_json(value: Any) -> str:
  """``json.dumps`` for a payload embedded inside a ``<script>`` element.

  ``json.dumps`` leaves ``<`` alone and the HTML parser does not: a hypothesis
  containing ``</script>`` would end the script element mid-payload and hand
  the rest of the text to the parser as markup -- stored XSS from any free-text
  field an LLM or an imported records writes. ``<`` only ever occurs inside
  JSON string literals, so rewriting it as its escape is content-preserving,
  and it removes both ``</script>`` and ``<!--`` (the other sequence the HTML
  spec assigns meaning inside script data) in one move. The result is still
  valid JSON and still a valid JS literal.
  """
  return json.dumps(value).replace("<", "\\u003c")


def render_lineage_html(
    records: Sequence[RunRecord],
    title: str = "rlmcp run tree",
    media_base: str = "media/",
    engine: str = "auto",
    posters: Optional[Dict[str, str]] = None,
) -> str:
  """One self-contained page: the tree, a detail panel, search and filters.

  Args:
    engine: ``cytoscape`` for the interactive view -- real DAG layout, pan and
      zoom, lineage highlighting -- or ``simple`` for a dependency-free renderer
      that draws the same graph with none of that. ``auto`` picks cytoscape when
      the vendored libraries are present.
    posters: video key to poster key, from
      :func:`rlmcp.records.poster.ensure_posters`. Omitting it costs the
      thumbnails and nothing else -- every view has a path for a run nobody
      filmed, because most runs are.

  Either way the result is one file: the libraries are inlined rather than
  fetched, so the page opens from a path, needs no server, and survives a strict
  content-security policy.
  """
  graph = build(records)
  payload = to_payload(graph, posters)
  stats = summarize(graph)
  # Reading every session's event log to draw parameter traces is the most
  # expensive thing this function does, and a lineage whose logs are gone still
  # has to render: a failure here costs the parameter view, not the page.
  try:
    history = build_history(graph)
    paths = leaf_paths(graph)
  except Exception as exc:  # noqa: BLE001 -- the tree matters more than the traces.
    history, paths = {"runs": [], "index": [], "span": 0, "error": str(exc)}, {}

  substitutions = {
      "__MEDIA__": _script_json(media_base),
      "__TITLE__": html.escape(title),
      "__RUNS__": _script_json(payload),
      "__PARAMS__": _script_json(history),
      "__PATHS__": _script_json(paths),
      "__EDGES__": _script_json({
          "config": graph.config_edges,
          "warm": graph.warm_edges,
          "prior": graph.prior_edges,
      }),
      "__STATS__": _script_json(stats),
      "__COLORS__": _script_json(VERDICT_COLORS),
  }

  if engine == "auto":
    engine = "cytoscape" if vendored() else "simple"
  if engine == "cytoscape":
    if not vendored():
      raise FileNotFoundError(
          f"The graph libraries are not vendored in {VENDOR}. Use engine='simple', "
          "or see vendor/README.md for what to drop in."
      )
    page = (TEMPLATES / "lineage_cytoscape.html").read_text()
    for token, name in (
        ("__CYTOSCAPE__", "cytoscape.min.js"),
        ("__DAGRE__", "dagre.min.js"),
        ("__CYTOSCAPE_DAGRE__", "cytoscape-dagre.js"),
    ):
      substitutions[token] = (VENDOR / name).read_text()
  else:
    page = _HTML_TEMPLATE

  # One pass, one callback. Chained .replace() calls re-scan the output of the
  # earlier ones, so a hypothesis that happened to contain "__STATS__" got the
  # stats JSON spliced into the middle of the payload. Substituted values must
  # never become substitution sites. (The callback also keeps re.sub from
  # interpreting backslashes in the JSON as group references.)
  pattern = re.compile("|".join(
      re.escape(token) for token in sorted(substitutions, key=len, reverse=True)))
  return pattern.sub(lambda match: substitutions[match.group(0)], page)


_HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>__TITLE__</title>
<style>
:root {
  --bg:#EFF1F3; --surface:#fff; --ink:#17212B; --muted:#5B6A79; --faint:#8A98A6;
  --line:#D4DBE1; --cfg:#9AA7B5; --warm:#C99A3A; --prior:#9A7FD1;
  --mono: ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;
  --sans: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
@media (prefers-color-scheme: dark){:root:not([data-theme=light]){
  --bg:#0E151C; --surface:#17212B; --ink:#E2EAF1; --muted:#94A4B4; --faint:#6E7F90;
  --line:#293542; --cfg:#55647A;
}}
:root[data-theme=dark]{--bg:#0E151C;--surface:#17212B;--ink:#E2EAF1;--muted:#94A4B4;
  --faint:#6E7F90;--line:#293542;--cfg:#55647A;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:15px}
header{padding:20px 24px 14px;border-bottom:1px solid var(--line)}
h1{margin:0 0 4px;font-size:19px;letter-spacing:-.01em}
h1 span{font-family:var(--mono);color:var(--faint);font-size:13px;font-weight:400}
.controls{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px;align-items:center}
input[type=search]{font:inherit;font-size:13px;padding:6px 10px;border:1px solid var(--line);
  border-radius:3px;background:var(--surface);color:var(--ink);min-width:220px}
input:focus-visible,button:focus-visible{outline:2px solid var(--cfg);outline-offset:2px}
.chip{font-family:var(--mono);font-size:11px;letter-spacing:.06em;text-transform:uppercase;
  padding:4px 9px;border-radius:999px;border:1px solid var(--line);background:var(--surface);
  color:var(--muted);cursor:pointer}
.chip.on{color:#fff;border-color:transparent}
.chip .n{opacity:.75;margin-left:5px}
main{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:0;align-items:start}
@media (max-width:900px){main{grid-template-columns:1fr}}
#canvas{position:relative;padding:18px 24px 60px;overflow:auto}
#edges{position:absolute;inset:0;pointer-events:none}
.node{position:absolute;display:block;text-align:left;background:var(--surface);
  border:1px solid var(--line);border-left:3px solid var(--faint);border-radius:3px;
  padding:5px 8px;width:172px;cursor:pointer;font-family:var(--sans);color:var(--ink)}
.node:hover{border-color:var(--cfg)}
.node.sel{box-shadow:0 0 0 2px var(--cfg)}
.node.dim{opacity:.25}
.node b{display:block;font-family:var(--mono);font-size:11px;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.node .ch{display:block;font-size:10.5px;color:var(--muted);margin-top:1px;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.node .tag{font-family:var(--mono);font-size:9px;color:var(--faint);
  display:block;margin-top:1px}
aside{position:sticky;top:0;max-height:100vh;overflow:auto;padding:18px 22px;
  border-left:1px solid var(--line);background:var(--surface)}
aside h2{margin:0 0 2px;font-size:16px;font-family:var(--mono)}
aside h3{margin:16px 0 5px;font-size:10.5px;font-family:var(--mono);letter-spacing:.1em;
  text-transform:uppercase;color:var(--faint)}
aside p{margin:0 0 8px;font-size:13.5px;line-height:1.5}
aside ul{margin:0;padding-left:17px;font-size:13.5px;line-height:1.5}
.kv{width:100%;border-collapse:collapse;font-size:12.5px}
.kv td{padding:3px 0;border-bottom:1px solid var(--line);vertical-align:top}
.kv td:last-child{text-align:right;font-family:var(--mono);padding-left:10px}
.badge{display:inline-block;font-family:var(--mono);font-size:10px;padding:2px 7px;
  border-radius:999px;color:#fff}
.note{font-size:12px;color:var(--muted);border-left:3px solid var(--warm);
  padding:8px 10px;background:var(--bg);border-radius:0 3px 3px 0;margin-top:10px}
.cap{font-size:11.5px;color:var(--muted);margin-top:4px}
.segs{display:flex;flex-wrap:wrap;gap:5px;margin-top:6px}
.plotgrid{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.plotgrid figure{margin:0;cursor:zoom-in}
.plotgrid img{width:100%;border:1px solid var(--line);border-radius:3px;display:block}
.plotgrid figcaption{font-size:10px;color:var(--faint);margin-top:2px}
.lightbox{position:fixed;inset:0;background:rgba(0,0,0,.85);display:flex;
  align-items:center;justify-content:center;z-index:50;cursor:zoom-out;padding:24px}
.lightbox img{max-width:100%;max-height:100%;border-radius:4px}
.recipe li{margin-bottom:4px}
.recipe .rid{font-family:var(--mono);color:var(--faint);font-size:11px}
footer{padding:10px 24px 28px;font-family:var(--mono);font-size:11px;color:var(--faint)}
</style></head><body>
<header>
  <h1>__TITLE__ <span id="stat"></span></h1>
  <div class="controls">
    <input type="search" id="q" placeholder="search id, slug, hypothesis, outcome" />
    <span id="chips"></span>
    <span id="edgetoggles"></span>
  </div>
</header>
<main>
  <div id="canvas"><svg id="edges"></svg></div>
  <aside id="panel"><p style="color:var(--muted)">Select a run.</p></aside>
</main>
<footer id="foot"></footer>
<script>
const RUNS=__RUNS__, EDGES=__EDGES__, STATS=__STATS__, COLORS=__COLORS__, MEDIA=__MEDIA__;
const byId=Object.fromEntries(RUNS.map(r=>[r.id,r]));
const LANE=196, ROW=58, PADX=20, PADY=14;
let selected=null, filter=new Set(), query="";
const showEdge={config:true, warm:true, prior:false};

const stat=document.getElementById("stat");
stat.textContent=`${STATS.records} runs · ${STATS.from_scratch} from scratch · `
  +`${STATS.warm_started} warm · ${STATS.config_edges} config edges`;

const chips=document.getElementById("chips");
Object.entries(STATS.verdicts).forEach(([v,n])=>{
  const b=document.createElement("button");
  b.className="chip"; b.dataset.v=v;
  b.innerHTML=`${esc(v)}<span class="n">${n}</span>`;
  b.onclick=()=>{filter.has(v)?filter.delete(v):filter.add(v);paint();};
  chips.appendChild(b);
});
const et=document.getElementById("edgetoggles");
[["config","config edges"],["warm","warm starts"],["prior","priors"]].forEach(([k,label])=>{
  const b=document.createElement("button");
  b.className="chip"+(showEdge[k]?" on":"");
  b.style.background=showEdge[k]?({config:"var(--cfg)",warm:"var(--warm)",prior:"var(--prior)"})[k]:"";
  b.textContent=label;
  b.onclick=()=>{showEdge[k]=!showEdge[k];
    b.classList.toggle("on",showEdge[k]);
    b.style.background=showEdge[k]?({config:"var(--cfg)",warm:"var(--warm)",prior:"var(--prior)"})[k]:"";
    drawEdges();};
  et.appendChild(b);
});
document.getElementById("q").addEventListener("input",e=>{query=e.target.value.toLowerCase();paint();});

function visible(r){
  if(filter.size && !filter.has(r.verdict)) return false;
  if(!query) return true;
  return [r.id,r.slug,r.hypothesis,r.outcome,r.stage].join(" ").toLowerCase().includes(query);
}
function pos(r){return {x:PADX+r.lane*LANE, y:PADY+r.row*ROW};}

const canvas=document.getElementById("canvas");
RUNS.forEach(r=>{
  const p=pos(r), b=document.createElement("button");
  b.className="node"; b.id="n"+r.id;
  b.style.left=p.x+"px"; b.style.top=p.y+"px";
  b.style.borderLeftColor=COLORS[r.verdict]||"#8A98A6";
  b.innerHTML=`<b>${esc(r.id)} ${esc(r.slug)}</b>`
    +`<span class="ch">${esc(r.change[0]||r.hypothesis||"—")}</span>`
    +`<span class="tag">${esc(r.verdict)}${r.scratch?" · from scratch":" · warm"}`
    +`${r.multi?" · multi-change":""}</span>`;
  b.onclick=()=>select(r.id);
  canvas.appendChild(b);
});
canvas.style.height=(PADY+RUNS.length*ROW+40)+"px";
canvas.style.minWidth=(PADX+Math.max(1,STATS.lanes)*LANE+40)+"px";

function paint(){
  RUNS.forEach(r=>{
    const el=document.getElementById("n"+r.id);
    el.classList.toggle("dim",!visible(r));
    el.classList.toggle("sel",r.id===selected);
  });
  drawEdges();
}
function drawEdges(){
  const svg=document.getElementById("edges");
  svg.setAttribute("width",canvas.scrollWidth); svg.setAttribute("height",canvas.scrollHeight);
  const seg=(a,b,color,dash,off)=>{
    const p1=pos(byId[a]),p2=pos(byId[b]);
    const x1=p1.x+14+(off||0), y1=p1.y+44, x2=p2.x+14+(off||0), y2=p2.y;
    const mid=(y1+y2)/2;
    return `<path d="M${x1},${y1} C${x1},${mid} ${x2},${mid} ${x2},${y2}" fill="none" `
      +`stroke="${color}" stroke-width="1.6"${dash?` stroke-dasharray="${dash}"`:""}/>`;
  };
  let out="";
  const on=(a,b)=>byId[a]&&byId[b]&&visible(byId[a])&&visible(byId[b]);
  if(showEdge.config) EDGES.config.forEach(([a,b])=>{ if(on(a,b)) out+=seg(a,b,"var(--cfg)",null,0); });
  if(showEdge.warm)   EDGES.warm.forEach(([a,b])=>{ if(on(a,b)) out+=seg(a,b,"var(--warm)","6 4",9); });
  if(showEdge.prior)  EDGES.prior.forEach(([a,b])=>{ if(on(a,b)) out+=seg(a,b,"var(--prior)","2 5",-9); });
  svg.innerHTML=out;
}
function esc(s){return (s==null?"":String(s))
  .replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
function mediaUrl(p){return MEDIA+String(p||"").split("/").map(encodeURIComponent).join("/");}
function select(id){
  const r=byId[id]; if(!r) return;
  selected=id; const P=document.getElementById("panel");
  let h=`<h2>${esc(r.id)} <span class="badge" `
   +`style="background:${COLORS[r.verdict]||"#8A98A6"}">${esc(r.verdict)}</span></h2>`
   +`<p style="font-family:var(--mono);color:var(--muted);font-size:12.5px">${esc(r.slug)}`
   +` · ${esc(r.stage)} · ${esc(r.date)} · ${esc(r.proposed_by)}</p>`;
  h+=`<h3>origin</h3><p>${r.scratch?"random init":"warm start from <b>"+esc(r.warm)+"</b>"}`
   +`${r.parent?` · config parent <b>${esc(r.parent)}</b>`:" · root"}`
   +`${r.prior?` · prior ${esc(r.prior)}`:""}</p>`;
  const vids=(r.assets&&r.assets.videos)||[], plots=(r.assets&&r.assets.plots)||[];
  if(vids.length){
    h+=`<h3>watched</h3><video id="vp" controls loop muted playsinline `
      +`src="${esc(mediaUrl(vids[0][0]))}" style="width:100%;border-radius:3px;background:#000">`
      +`</video><p class="cap">${esc(vids[0][1]||"")}</p>`;
    if(vids.length>1) h+=`<div class="segs">`+vids.map((v,i)=>
      `<button class="chip" data-src="${esc(mediaUrl(v[0]))}">${esc(v[1]||("clip "+(i+1)))}</button>`
      ).join("")+`</div>`;
  }
  if(plots.length){
    h+=`<h3>plots</h3><div class="plotgrid">`+plots.map(pl=>
      `<figure><img loading="lazy" src="${esc(mediaUrl(pl[0]))}" alt="${esc(pl[1]||"")}" />`
      +`<figcaption>${esc(pl[1]||"")}</figcaption></figure>`).join("")+`</div>`;
  }
  if(r.hypothesis) h+=`<h3>hypothesis</h3><p>${esc(r.hypothesis)}</p>`;
  if(r.prediction) h+=`<h3>prediction</h3><p>${esc(r.prediction)}</p>`;
  if(r.falsifier||r.falsifier_conditions.length){
    h+=`<h3>falsifier</h3><p>${esc(r.falsifier)}</p>`;
    if(r.falsifier_conditions.length)
      h+=`<ul>${r.falsifier_conditions.map(c=>`<li><code>${esc(c)}</code></li>`).join("")}</ul>`;
  }
  if(r.change.length) h+=`<h3>change vs ${esc(r.parent||"root")}</h3>`
    +`<ul>${r.change.map(c=>`<li>${esc(c)}</li>`).join("")}</ul>`;
  if(r.multi) h+=`<div class="note">Multiple changes in one run — the outcome is `
    +`directional, not attributable to any single change.</div>`;
  if(r.outcome) h+=`<h3>outcome</h3><p>${esc(r.outcome)}</p>`;
  if(r.metrics.length) h+=`<h3>evidence</h3><table class="kv">`
    +r.metrics.map(m=>`<tr><td>${esc(m[0])}</td><td>${esc(m[1])}</td></tr>`).join("")+`</table>`;
  if(!r.scratch) h+=`<div class="note">Warm-started: this shows the change `
    +`<b>preserves</b> a behaviour, not that it creates one. Capped at `
    +`<code>provisional</code> until a from-scratch run promotes it.</div>`;
  h+=`<h3>recipe at this node — the fold of ${r.recipe.length} diff(s)</h3><ul class="recipe">`
    +r.recipe.map(step=>`<li><span class="rid">${esc(step.id)}</span> — `
      +`${esc((step.change||[]).join("; ")||"(no change recorded)")}</li>`).join("")+`</ul>`;
  h+=`<div class="note" style="border-left-color:var(--cfg)">The recipe is not stored `
    +`anywhere — it is computed by walking the config edges from the root to this node, `
    +`so provenance and the current recipe cannot drift apart.</div>`;
  if(r.session) h+=`<h3>session</h3><p style="font-family:var(--mono);font-size:11px;`
    +`word-break:break-all">${esc(r.session)}</p>`;
  P.innerHTML=h;
  P.querySelectorAll(".segs .chip").forEach(b=>b.onclick=()=>{
    const v=document.getElementById("vp"); v.src=b.dataset.src; v.play();
  });
  const v=document.getElementById("vp");
  if(v) v.addEventListener("error",()=>{
    v.outerHTML=`<p class="cap">Video not reachable from this copy of the page — `
      +`open it next to the lab directory, or fetch the file from the record.</p>`;
  },{once:true});
  P.querySelectorAll(".plotgrid img").forEach(img=>{
    img.addEventListener("error",()=>{img.closest("figure").style.display="none"},{once:true});
    img.onclick=()=>{
      const box=document.createElement("div"); box.className="lightbox";
      box.innerHTML=`<img src="${esc(img.src)}" />`;
      box.onclick=()=>box.remove(); document.body.appendChild(box);
    };
  });
  paint();
}
document.getElementById("foot").textContent=
  `roots: ${STATS.roots.join(", ")} · leaves: ${STATS.leaves.join(", ")} · `
  +`solid = config lineage, dashed amber = warm start, dotted violet = prior `
  +`(priors hidden by default — toggle above)`;
select((RUNS.find(r=>r.verdict==="best")||RUNS[RUNS.length-1]||{}).id);
paint();
</script></body></html>
"""


def plot_run_comparison(
    series_by_run: Dict[str, Dict[str, List[List[float]]]],
    metrics: Sequence[str],
    at_iteration: Optional[int] = None,
    title: Optional[str] = None,
) -> bytes:
  """Compare runs on the same axes, one panel per metric, one line per run.

  ``at_iteration`` truncates every run to the same x, because "better at the
  end" often just means "trained longer".
  """
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  metrics = [m for m in metrics if any(m in s for s in series_by_run.values())]
  if not metrics:
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.text(0.5, 0.5, "No shared metrics between these runs",
            ha="center", va="center")
    ax.axis("off")
    return _png(fig)

  cols = min(2, len(metrics))
  rows = (len(metrics) + cols - 1) // cols
  fig, axes = plt.subplots(rows, cols, figsize=(6.4 * cols, 3.2 * rows), squeeze=False)
  flat = axes.flatten()
  palette = ["#1F77B4", "#B93E3C", "#2C7A57", "#9A6314", "#6B7FA8", "#8C6BB1"]

  for ax, metric in zip(flat, metrics):
    for i, (run_id, series) in enumerate(sorted(series_by_run.items())):
      points = series.get(metric) or []
      if at_iteration is not None:
        points = [p for p in points if p[0] <= at_iteration]
      if not points:
        continue
      ax.plot([p[0] for p in points], [p[1] for p in points],
              color=palette[i % len(palette)], lw=1.3, label=run_id)
    ax.set_title(metric, fontsize=10, fontweight="bold")
    ax.set_xlabel("iteration", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)

  for ax in flat[len(metrics):]:
    ax.axis("off")
  suffix = f" — matched at iteration {at_iteration}" if at_iteration else ""
  fig.suptitle((title or "run comparison") + suffix, fontsize=12, fontweight="bold")
  fig.tight_layout()
  return _png(fig)
