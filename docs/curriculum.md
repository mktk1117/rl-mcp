# Curriculum: the ladder, written down

If you catch yourself writing "when metric X passes Y, change weight Z" into a
notes file, that is a curriculum stage. Encode it and three things follow: the
run drives itself, the ladder is saved with the run (`params/curriculum.json`),
and a human can still override it live.

## A stage says two things

What changes when the run enters it, and what has to be true before it leaves.
It says both in the only two vocabularies the core has: parameter edits and
commands. That is why the same structure drives a locomotion ladder and a
manipulation one.

```python
from rlmcp import Action, Condition, CurriculumStage

CurriculumStage(
    name="1_rough",
    parameters={"reward.foot_clearance.weight": -1.0},
    apply=[Action("set_terrain", {"terrains": ["flat", "random_rough"],
                                  "max_level": 5})],
    promote_when=[
        Condition("rlmcp/episode_length_frac", ">=", 0.5),   # it survives
        Condition("rlmcp/terrain_level_frac",  ">=", 0.6),   # it is progressing
    ],
    min_iterations=250,
    hold_iterations=20,
)
```

| field | meaning |
| --- | --- |
| `parameters` | keys from `rlmcp params`, applied on entry |
| `apply` | commands from `rlmcp commands`, including your extension's verbs |
| `promote_when` | conditions that all have to hold |
| `min_iterations` | floor. The stage cannot end before this |
| `hold_iterations` | conditions must hold this many iterations in a row |
| `notes` | free text, shown in `rlmcp curriculum` |

Promotion is earned, not scheduled. Every condition must hold for
`hold_iterations` consecutive iterations *and* the floor must have passed.

`set_terrain` above is not a core concept. It comes from the terrain extension.
See [extensions.md](extensions.md).

## A non-terrain example

`rlmcp-train --curriculum {terrain,none}` is a convenience entry point, and its
two choices are all *that CLI* offers. The library is not so limited. Here is a
rung from a manipulation ladder that starts with the object already in the
gripper and works up to picking it off a table:

```python
CurriculumStage(
    name="1_hold_and_place",
    apply=[Action("set_goal_difficulty",
                  {"spawn_mode": "in_gripper", "tolerance": 0.15})],
    parameters={"reward.reach_goal.weight": 300.0,
                "reward.grasp_stability.weight": 60.0,
                "reward.drop_penalty.weight": -120.0},
    promote_when=[
        Condition("Metrics/goals/goals_per_min", ">=", 25.0),
        Condition("Metrics/goals/episode_dropped", "<=", 0.5),
    ],
    min_iterations=500,
    hold_iterations=20,
    notes="Keep hold of it and place it, before learning to pick it up.",
)
```

Wire the stages into a schedule and hand it to `wrap`:

```python
from rlmcp.core.curriculum import StageSchedule

schedule = StageSchedule([stage0, stage1, stage2])
env = rlmcp.wrap(env, session_dir=..., curriculum=schedule)
```

Both `parameters` and `apply` are written to the event log when the stage is
entered. That is what lets the records page draw what changed and when, and what
lets `rlmcp play` put a checkpoint back on its own rung.

## Pick promotion metrics the policy cannot buy

Two rules, both usually learned the expensive way.

**Task-external, not the reward being optimised.** A reward term rises when the
policy games it. Successes per minute, success rate and holding fraction do not.

**Normalised: rates and fractions, never per-episode counts.** An episode that
ends early on a failure is shorter for reasons unrelated to skill, so a raw count
promotes whatever survives longest. Divide counts by
`episode_length_buf * step_dt` and publish the rate.

Extension `metrics()` are merged into telemetry, so anything your task publishes
is usable directly in a condition.

## Overriding it live

```bash
rlmcp curriculum                     # which rung, and what it is waiting for
rlmcp curriculum advance --why "the easy tolerance is solved"
rlmcp curriculum goto 3_from_table --why "the warm start already places reliably"
rlmcp curriculum auto-off            # stop it promoting itself
rlmcp curriculum auto-on
```

A manual command beats the automatic schedule within the same iteration. From an
agent: `curriculum_status`, `curriculum_advance`, `curriculum_goto`,
`curriculum_auto`.
