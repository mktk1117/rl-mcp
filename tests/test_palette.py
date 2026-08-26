"""A debug overlay that reads as an object in the scene has cost this project
real investigation time twice. These pin the check that now runs at startup."""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import pytest

from rlmcp.adapters.mjlab.viz_check import (
  check_marker_colors,
  collect_marker_colors,
  collect_scene_colors,
)
from rlmcp.core.palette import (
  DEFAULT_MIN_DISTANCE,
  MARKER_PALETTE,
  find_collisions,
  format_report,
  rgba_distance,
  suggest_distinct,
)

# The pair that actually caused the problem: a landing-point marker and the
# third juggling ball.
MARKER_YELLOW = (1.00, 0.85, 0.10, 0.70)
BALL_YELLOW = (0.95, 0.80, 0.15, 1.00)
BALL_RED = (0.90, 0.25, 0.15, 1.00)
BALL_BLUE = (0.15, 0.45, 0.90, 1.00)


def test_the_real_collision_is_caught():
  assert rgba_distance(MARKER_YELLOW, BALL_YELLOW) < DEFAULT_MIN_DISTANCE
  found = find_collisions({"m": MARKER_YELLOW}, {"ball2": BALL_YELLOW})
  assert [c["collides_with"] for c in found] == ["ball2"]


def test_distinct_colors_do_not_warn():
  assert find_collisions({"m": MARKER_PALETTE[0]}, {"ball0": BALL_RED}) == []


def test_alpha_is_ignored():
  """A translucent marker over a solid object of the same hue is exactly the
  confusion this exists to prevent, so alpha must not rescue it."""
  solid = (1.0, 0.85, 0.1, 1.0)
  ghost = (1.0, 0.85, 0.1, 0.05)
  assert rgba_distance(solid, ghost) == pytest.approx(0.0)


def test_collisions_are_reported_nearest_first():
  near = (0.95, 0.80, 0.15, 1.0)
  nearer = (1.00, 0.85, 0.10, 1.0)
  found = find_collisions({"m": MARKER_YELLOW}, {"far": near, "close": nearer})
  assert [c["collides_with"] for c in found] == ["close", "far"]


def test_suggestion_clears_every_scene_color():
  scene = [BALL_RED, BALL_BLUE, BALL_YELLOW]
  pick = suggest_distinct(scene)
  assert min(rgba_distance(pick, c) for c in scene) >= DEFAULT_MIN_DISTANCE


def test_suggestion_survives_a_rainbow_scene():
  """Never raise, even when no palette entry is comfortably clear."""
  scene = list(MARKER_PALETTE)
  assert suggest_distinct(scene) in MARKER_PALETTE


def test_report_names_both_sides():
  text = format_report(find_collisions({"juggle.viz": MARKER_YELLOW}, {"ball2": BALL_YELLOW}))
  assert "juggle.viz" in text and "ball2" in text


# --- the mjlab-side extraction -------------------------------------------


@dataclasses.dataclass
class _Cfg:
  debug_vis: bool = True
  viz_marker_color: Tuple[float, float, float, float] = MARKER_YELLOW
  viz_face_colors: Tuple[Tuple[float, float, float, float], ...] = (BALL_RED,)
  unrelated: float = 1.0
  optional_color: Optional[Tuple[float, ...]] = None


class _Term:
  def __init__(self, cfg):
    self.cfg = cfg


class _Manager:
  def __init__(self, terms):
    self._terms = terms


class _Geom:
  def __init__(self, name, rgba):
    self.name = name
    self.rgba = rgba


class _Model:
  def __init__(self, geoms):
    self._geoms = geoms
    self.ngeom = len(geoms)

  def geom(self, i):
    return self._geoms[i]


class _Sim:
  def __init__(self, model):
    self.mj_model = model


class _Env:
  def __init__(self, terms, geoms):
    self.command_manager = _Manager(terms)
    self.sim = _Sim(_Model(geoms))


def test_marker_colors_found_by_convention():
  env = _Env({"juggle": _Term(_Cfg())}, [])
  found = collect_marker_colors(env)
  assert found["juggle.viz_marker_color"] == MARKER_YELLOW
  assert found["juggle.viz_face_colors[0]"] == BALL_RED
  assert "juggle.unrelated" not in found
  assert "juggle.optional_color" not in found


def test_terms_with_debug_vis_off_are_skipped():
  env = _Env({"juggle": _Term(_Cfg(debug_vis=False))}, [])
  assert collect_marker_colors(env) == {}


def test_invisible_and_unnamed_geoms_are_skipped():
  env = _Env(
    {},
    [_Geom("ball", BALL_RED), _Geom("ghost", (1, 1, 1, 0.0)), _Geom("", BALL_BLUE)],
  )
  assert list(collect_scene_colors(env)) == ["ball"]


def test_end_to_end_flags_the_marker_against_the_ball():
  env = _Env({"juggle": _Term(_Cfg())}, [_Geom("ball2", BALL_YELLOW)])
  collisions = check_marker_colors(env)
  assert any(c["collides_with"] == "ball2" for c in collisions)


def test_no_scene_means_no_report_rather_than_a_crash():
  assert check_marker_colors(_Env({"juggle": _Term(_Cfg())}, [])) == []
