"""`rlmcp-train --curriculum-json / --config-json`: a recipe's files as flags.

No simulator here: the loaders and the argument parser are what a recipe's
launch.sh depends on, and they are testable without one.
"""

from __future__ import annotations

import json

import pytest

from rlmcp.core.curriculum import CurriculumStage, StageSchedule
from rlmcp.train import _parse_args, load_config_json, load_curriculum_json


def test_a_recipe_ladder_starts_a_new_run_on_its_first_rung(tmp_path):
  """A file saved from a live schedule remembers the rung it was on; a new
  run must not start there."""
  live = StageSchedule([CurriculumStage(name="0_a"), CurriculumStage(name="1_b")])
  live.index = 1
  live.entered_at_iteration = 900
  live.history = [{"to": "1_b"}]
  path = tmp_path / "curriculum.json"
  path.write_text(json.dumps(live.to_dict()))

  loaded = load_curriculum_json(str(path))

  assert [s.name for s in loaded.stages] == ["0_a", "1_b"]
  assert (loaded.index, loaded.entered_at_iteration, loaded.history) == (0, 0, [])


def test_a_bare_list_of_stages_is_accepted_too(tmp_path):
  """`params/curriculum.json` in a run directory is written as a list."""
  path = tmp_path / "curriculum.json"
  path.write_text(json.dumps([CurriculumStage(name="only").to_dict()]))
  assert [s.name for s in load_curriculum_json(str(path)).stages] == ["only"]


def test_no_path_means_no_schedule_and_no_config():
  assert load_curriculum_json("") is None
  assert load_config_json("") == {}


def test_a_config_that_is_not_an_object_is_refused_by_name(tmp_path):
  path = tmp_path / "config.json"
  path.write_text("[1, 2]")
  with pytest.raises(ValueError, match=r"config\.json"):
    load_config_json(str(path))


def test_the_flags_parse_beside_the_existing_ones():
  args = _parse_args(["Task", "--curriculum-json", "c.json", "--config-json",
                      "k.json", "--task-package", "shand.tasks", "--seed", "7"])
  assert (args.curriculum_json, args.config_json) == ("c.json", "k.json")
  assert args.task_package == ["shand.tasks"]
