"""Keep debug overlays distinguishable from the things they annotate.

A marker drawn in the same colour as an object in the scene is worse than no
marker: it reads as another object. This has cost real time in this project
twice -- a landing-point marker the same yellow as one of the juggling balls,
and neighbouring environments composited into a frame and read as duplicate
objects -- and in both cases the investigation went looking for a physics bug
that was never there.

The rule is cheap to check and the check is task-independent, so it lives here
rather than in any one task's config. Pure colour arithmetic; the backend-side
extraction of what is actually in a scene lives in the adapter.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

RGBA = Sequence[float]

# Distance below which a marker and a scene object are too easily confused.
# Calibrated on the pair that caused the problem: marker (1.00, 0.85, 0.10)
# against ball (0.95, 0.80, 0.15) sits at 0.09, and the two are indistinguishable
# on screen. Colours a comfortable step apart -- red against orange, say --
# score around 0.3.
DEFAULT_MIN_DISTANCE = 0.25

# Hues that read clearly against the usual robot greys and terrain blues, and
# against each other. Reserve these for overlays and keep objects out of them.
MARKER_PALETTE: tuple[tuple[float, float, float, float], ...] = (
  (0.85, 0.20, 0.95, 0.75),  # magenta
  (0.10, 0.90, 0.85, 0.75),  # cyan
  (1.00, 0.45, 0.00, 0.75),  # orange
  (0.55, 1.00, 0.20, 0.75),  # lime
  (1.00, 1.00, 1.00, 0.75),  # white
)


def rgba_distance(a: RGBA, b: RGBA) -> float:
  """Perceptual-ish distance between two colours, ignoring alpha.

  Weighted Euclidean in RGB using the usual luminance coefficients, which is
  crude but sufficient for "would a person mix these up in a video frame".
  Alpha is deliberately excluded: a translucent marker over a solid object of
  the same hue is exactly the confusion this is here to prevent.
  """
  weights = (0.30, 0.59, 0.11)
  return sum(w * (float(x) - float(y)) ** 2 for w, x, y in zip(weights, a, b, strict=False)) ** 0.5


def find_collisions(
  markers: dict[str, RGBA],
  scene: dict[str, RGBA],
  min_distance: float = DEFAULT_MIN_DISTANCE,
) -> list[dict]:
  """Marker/scene colour pairs a viewer would confuse.

  Returns one entry per offending pair, nearest first, each naming both sides
  and a replacement drawn from :data:`MARKER_PALETTE` that clears every colour
  in the scene.
  """
  out: list[dict] = []
  for marker_name, marker_rgba in markers.items():
    for scene_name, scene_rgba in scene.items():
      distance = rgba_distance(marker_rgba, scene_rgba)
      if distance < min_distance:
        out.append(
          {
            "marker": marker_name,
            "marker_color": tuple(round(float(c), 3) for c in marker_rgba),
            "collides_with": scene_name,
            "scene_color": tuple(round(float(c), 3) for c in scene_rgba),
            "distance": round(distance, 3),
            "suggestion": suggest_distinct(scene.values()),
          }
        )
  out.sort(key=lambda e: e["distance"])
  return out


def suggest_distinct(
  scene: Iterable[RGBA], min_distance: float = DEFAULT_MIN_DISTANCE
) -> tuple[float, float, float, float]:
  """A marker colour from the reserved palette that clears everything in scene.

  Falls back to whichever palette entry is furthest from the nearest scene
  colour, so this always returns something usable even in a rainbow scene.
  """
  colors = list(scene)
  if not colors:
    return MARKER_PALETTE[0]
  best = MARKER_PALETTE[0]
  best_margin = -1.0
  for candidate in MARKER_PALETTE:
    margin = min(rgba_distance(candidate, c) for c in colors)
    if margin >= min_distance:
      return candidate
    if margin > best_margin:
      best, best_margin = candidate, margin
  return best


def format_report(collisions: Sequence[dict]) -> str:
  """One human-readable line per collision, for a log or a warning."""
  lines = []
  for c in collisions:
    lines.append(
      f"debug marker {c['marker']} {c['marker_color']} is the same colour as "
      f"{c['collides_with']} {c['scene_color']} (distance {c['distance']}); "
      f"in a rendered frame the marker will read as another "
      f"{c['collides_with']}. Try {c['suggestion']}."
    )
  return "\n".join(lines)
