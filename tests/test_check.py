"""Verifying a task, without a task and without a simulator.

`rlmcp check` builds an environment, so most of it looks untestable from here.
It is not: the part worth testing is the *judgement* -- which gate fails, which
ones stop, how the terms are ranked and what counts as a broken termination mix
-- and every one of those is a function over a rollout's numbers. The rollout
itself is fed by a fake environment that behaves in the ways real ones break.

The one thing a simulator would add is confidence that the pieces are wired
together, and `tests/test_cli_dispatch.py` walks that.
"""

from __future__ import annotations

import math

import pytest

from rlmcp.check import (
    CheckConfig,
    CheckError,
    dominance,
    gates_from,
    rank_terms,
    roll,
    run_check,
    summarise,
)


# ── a fake environment, broken in specific ways ──────────────────────────

class FakeAdapter:
  def __init__(self, rewards=None, terminations=None, raises=None):
    self._rewards = rewards or {}
    self._terminations = terminations or {}
    self._raises = raises

  def reward_terms(self):
    if self._raises:
      raise self._raises
    return dict(self._rewards)

  def termination_terms(self):
    return dict(self._terminations)


class FakeTensor:
  """Just enough of a torch tensor for the rollout to read it."""

  def __init__(self, value, count=1):
    self._value = value
    self._count = count

  def float(self):
    return self

  def mean(self):
    return self

  def sum(self):
    return self

  def item(self):
    return self._value


class FakeEnv:
  def __init__(self, reward=1.0, done_every=0, steps_before_raising=0):
    self.reward = reward
    self.done_every = done_every
    self.steps_before_raising = steps_before_raising
    self.steps = 0

  def get_observations(self):
    return "obs"

  def step(self, _action):
    self.steps += 1
    if self.steps_before_raising and self.steps > self.steps_before_raising:
      raise RuntimeError("the environment fell over")
    done = 1 if self.done_every and self.steps % self.done_every == 0 else 0
    return "obs", FakeTensor(self.reward), FakeTensor(done), {}


def _policy(_obs):
  return "action"


# ── ranking, which is the finding this command exists for ────────────────

def test_terms_are_ranked_by_what_they_pay():
  rows = rank_terms({"alive": 300.0, "track": 30.0, "action_rate": -3.0}, steps=100)

  assert [r["term"] for r in rows] == ["alive", "track", "action_rate"]
  assert rows[0]["mean"] == pytest.approx(3.0)
  assert rows[0]["share"] == pytest.approx(300 / 333, rel=1e-3)


def test_a_penalty_that_swamps_everything_ranks_first():
  """Sorting by signed value would put the biggest term at the bottom, which
  is the one case where the ranking has to work."""
  rows = rank_terms({"alive": 10.0, "joint_acc_l2": -900.0}, steps=10)

  assert rows[0]["term"] == "joint_acc_l2"


def test_dominance_names_the_2600x_case():
  rows = rank_terms({"alive": 2600.0, "progress": 1.0}, steps=1)
  found = dominance(rows)

  assert found["term"] == "alive" and found["over"] == "progress"
  assert found["ratio"] == pytest.approx(2600.0)


def test_dominance_of_a_single_term_is_not_a_ratio():
  assert dominance(rank_terms({"only": 1.0}, steps=1)) == {}


def test_a_term_paying_nothing_at_all_is_infinite_dominance():
  rows = rank_terms({"alive": 2.0, "progress": 0.0}, steps=1)

  assert math.isinf(dominance(rows)["ratio"])


# ── the rollout ──────────────────────────────────────────────────────────

def test_a_rollout_totals_each_term_and_counts_the_ends():
  env = FakeEnv(reward=0.5, done_every=5)
  adapter = FakeAdapter(rewards={"alive": 1.0}, terminations={"fell": 0.25})

  rolled = roll(env, adapter, _policy, steps=10)

  assert rolled["completed"] == 10
  assert rolled["reward_totals"] == {"alive": 10.0}
  assert rolled["termination_totals"]["fell"] == pytest.approx(2.5)
  assert rolled["dones"] == 2
  assert rolled["first_done_step"] == 5


def test_a_nan_term_is_recorded_rather_than_summed():
  """One NaN poisons the sum, and PPO trains on it without complaining."""
  env = FakeEnv()
  adapter = FakeAdapter(rewards={"alive": 1.0, "broken": float("nan")})

  rolled = roll(env, adapter, _policy, steps=3)

  assert rolled["reward_totals"] == {"alive": 3.0}
  assert len(rolled["nonfinite"]) == 3
  assert "reward.broken" in rolled["nonfinite"][0]


def test_an_adapter_that_cannot_break_rewards_down_says_why_once():
  env = FakeEnv()
  adapter = FakeAdapter(raises=NotImplementedError("no terms here"))

  rolled = roll(env, adapter, _policy, steps=4)

  assert rolled["terms_seen"] is False
  assert "NotImplementedError" in rolled["terms_refused"]


# ── the gates ────────────────────────────────────────────────────────────

def _rolled(**over):
  base = {"completed": 100, "nonfinite": [], "dones": 0, "first_done_step": 0,
          "terms_seen": True, "terminations_seen": True, "reward_totals": {"a": 1.0},
          "termination_totals": {}, "terms_refused": "", "terminations_refused": ""}
  base.update(over)
  return base


def test_an_env_that_ends_on_step_one_fails_the_termination_gate():
  """The failure this command exists to catch before a GPU is spent on it."""
  gates = gates_from(_rolled(dones=4, first_done_step=1), steps=100, num_envs=4)
  by_name = {g["gate"]: g for g in gates}

  assert by_name["terminations"]["ok"] is False
  assert "step 1" in by_name["terminations"]["detail"]
  assert by_name["terminations"]["fix"]


def test_episodes_ending_far_too_often_fails_too():
  gates = gates_from(_rolled(dones=200, first_done_step=30), steps=100, num_envs=4)

  assert {g["gate"]: g["ok"] for g in gates}["terminations"] is False


def test_nothing_ever_ending_is_reported_but_is_not_a_failure():
  """A task can legitimately never terminate; it is worth seeing, not failing."""
  gates = gates_from(_rolled(), steps=100, num_envs=4)
  by_name = {g["gate"]: g for g in gates}

  assert by_name["terminations"]["ok"] is True
  assert "nothing terminated" in by_name["terminations"]["detail"]


def test_a_short_rollout_fails_the_steps_gate():
  gates = gates_from(_rolled(completed=12), steps=100, num_envs=4)

  assert {g["gate"]: g["ok"] for g in gates}["steps"] is False


def test_a_gate_that_did_not_run_is_neither_pass_nor_fail():
  """Four ticks and one cross, where one of the ticks never ran, is how a
  broken environment comes back looking mostly fine."""
  gates = gates_from(_rolled(terms_seen=False, terminations_seen=False), steps=100,
                     num_envs=4)
  by_name = {g["gate"]: g for g in gates}

  assert by_name["rewards_finite"]["ok"] is None
  assert by_name["rewards_finite"]["detail"].startswith("not run")

  verdict = summarise(gates)
  assert verdict["passed"] is True
  assert "rewards_finite" in verdict["skipped"]


def test_the_first_problem_is_named():
  gates = gates_from(_rolled(completed=2, nonfinite=["reward.x at step 1"]),
                     steps=100, num_envs=1)

  assert summarise(gates)["first_problem"] == "steps"


# ── refusals ─────────────────────────────────────────────────────────────

def test_a_check_with_no_task_says_why_there_is_no_default():
  with pytest.raises(CheckError) as caught:
    run_check(CheckConfig(task=""))

  assert "--task" in str(caught.value)


def test_a_checkpoint_policy_is_refused_because_a_check_has_no_weights():
  with pytest.raises(CheckError) as caught:
    run_check(CheckConfig(task="Some-Task", policy="checkpoint"))

  assert "zero or random" in str(caught.value)


def test_the_build_failing_stops_the_gates_after_it(monkeypatch):
  """A "reward is NaN" line under "the env would not construct" sends somebody
  to fix a reward that was never the problem."""
  import rlmcp.play as play

  def explode(*_args, **_kwargs):
    raise RuntimeError("SceneEntityCfg: no body matches 'torso_link'")

  monkeypatch.setattr(play, "build_env", explode)
  answer = run_check(CheckConfig(task="Some-Task", steps=4))

  by_name = {g["gate"]: g for g in answer["gates"]}
  assert by_name["imports"]["ok"] is True
  assert by_name["constructs"]["ok"] is False
  assert "torso_link" in by_name["constructs"]["detail"]
  for later in ("steps", "rewards_finite", "terminations"):
    assert by_name[later]["ok"] is None
  assert answer["passed"] is False
  assert answer["first_problem"] == "constructs"


def test_an_unimportable_package_is_an_import_problem_not_a_build_one(monkeypatch):
  import rlmcp.play as play

  def explode(*_args, **_kwargs):
    raise play.PlayError("Could not import task package 'nope': no module")

  monkeypatch.setattr(play, "build_env", explode)
  answer = run_check(CheckConfig(task="Some-Task"))

  by_name = {g["gate"]: g for g in answer["gates"]}
  assert by_name["imports"]["ok"] is False
  assert "--task-package" in by_name["imports"]["fix"]
  # Every gate is reported, and nothing claims the environment was built and
  # then failed: the four after the import say they did not run.
  assert [g["gate"] for g in answer["gates"]] == [
      "imports", "constructs", "steps", "rewards_finite", "terminations"]
  assert by_name["constructs"]["ok"] is None
