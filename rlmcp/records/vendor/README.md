# Vendored viewer libraries

Inlined into the generated page so the run tree stays one self-contained file:
it opens from a `file://` path, works with no server and no network, and keeps
working inside a strict content-security policy. That property is why these are
committed rather than fetched.

| file | version | licence |
| --- | --- | --- |
| `cytoscape.min.js` | 3.30.2 | MIT — © 2016-2024 The Cytoscape Consortium |
| `dagre.min.js` | 0.8.5 | MIT |
| `cytoscape-dagre.js` | 2.5.0 | MIT |

Together ~660 KB, which is the cost of the interactive view. If they are absent,
`render_lineage_html` falls back to a dependency-free renderer that draws the
same graph without pan, zoom or automatic layout.

To update: fetch the same paths from unpkg, drop them in, and check the viewer
still lays out — nothing else pins the versions.
