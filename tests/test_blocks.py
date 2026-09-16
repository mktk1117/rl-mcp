"""The building blocks in :mod:`rlmcp.blocks`: variables, pipes, groups.

No simulator. What these pin is the behaviour an env.py relies on: a
declared variable is allocated with the right shape and labels, a delay
returns the value from ``steps`` ago and forgets the old episode on reset,
an observation group concatenates in declaration order and knows where each
term sits, and the parameters of every stage are found by the walk rlmcp
serves them from.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from rlmcp import blocks
from rlmcp.blocks import Clip, Delay, Noise, Obs, Offset, Pipe, Scale, Vars, var


class State(Vars):
  base_lin_vel: Tensor = var(3)
  joint_pos: Tensor = var("joint")
  foot_pos: Tensor = var("foot", 3)
  foot_contact: Tensor = var("foot", dtype=torch.bool)
  commands: Tensor = var(("lin_vel_x", "lin_vel_y", "ang_vel_z"))
  episode_length: Tensor = var(dtype=torch.long)


def state(n: int = 4) -> State:
  return State(n, "cpu", joint=["hip", "knee"], foot=("L", "R"))


# Variables.


def test_variables_are_allocated_from_their_declaration():
  s = state()
  assert s.base_lin_vel.shape == (4, 3)
  assert s.joint_pos.shape == (4, 2)
  assert s.foot_pos.shape == (4, 2, 3)
  assert s.foot_contact.dtype is torch.bool
  assert s.episode_length.dtype is torch.long
  assert s.names() == ["base_lin_vel", "joint_pos", "foot_pos", "foot_contact",
                       "commands", "episode_length"]


def test_labels_come_from_the_dimension_or_the_declaration():
  s = state()
  assert s.labels("joint_pos") == ["hip", "knee"]
  assert s.labels("foot_pos") == ["L.0", "L.1", "L.2", "R.0", "R.1", "R.2"]
  assert s.labels("commands") == ["lin_vel_x", "lin_vel_y", "ang_vel_z"]
  assert s.labels("base_lin_vel") is None


def test_a_dimension_the_container_was_not_given_is_named():
  with pytest.raises(KeyError) as excinfo:
    State(2, "cpu", joint=2)
  assert "foot" in str(excinfo.value) and "joint" in str(excinfo.value)


def test_reset_zeroes_only_the_given_envs():
  s = state()
  s.joint_pos[:] = 1.0
  s.episode_length[:] = 7
  s.reset(torch.tensor([1, 3]))
  assert s.joint_pos[[1, 3]].sum() == 0 and s.joint_pos[[0, 2]].sum() == 4
  assert s.episode_length.tolist() == [7, 0, 7, 0]
  s.reset()
  assert s.episode_length.sum() == 0


def test_a_subclass_inherits_the_declarations():
  class More(State):
    extra: Tensor = var(2)

  s = More(2, "cpu", joint=1, foot=1)
  assert "joint_pos" in s and s.extra.shape == (2, 2)


# Stages.


def test_delay_returns_the_value_from_steps_ago_and_starts_full():
  d = Delay(2)
  x = torch.zeros(3, 1)
  assert torch.equal(d(x + 1), x + 1)  # First call: history is all this value.
  assert torch.equal(d(x + 2), x + 1)
  assert torch.equal(d(x + 3), x + 1)
  assert torch.equal(d(x + 4), x + 2)


def test_delay_forgets_the_previous_episode_for_reset_envs():
  d = Delay(1)
  x = torch.zeros(3, 1)
  d(x + 1)
  d.reset(torch.tensor([0]))
  out = d(x + 5)
  assert out[:, 0].tolist() == [5.0, 1.0, 1.0]


def test_delay_steps_is_live_within_its_history():
  d = Delay(1, max_steps=3)
  x = torch.zeros(1, 1)
  for k in range(1, 5):
    d(x + k)
  d.steps = 3
  assert float(d(x + 5)) == 2.0
  d.steps = 0
  assert float(d(x + 6)) == 6.0
  assert d.bounds() == {"steps": (0, 3)}
  with pytest.raises(ValueError):
    Delay(5, max_steps=2)


def test_noise_scale_clip_offset():
  x = torch.zeros(2, 3)
  assert torch.equal(Noise(0.0)(x), x)
  noisy = Noise(0.5)(x)
  assert noisy.abs().max() <= 0.5 and not torch.equal(noisy, x)
  assert torch.equal(Scale(2.0)(x + 1), x + 2)
  assert torch.equal(Clip(-1.0, 1.0)(x + 5), x + 1)
  bias = torch.tensor([[1.0], [2.0]])
  assert torch.equal(Offset(bias)(x), x + bias)


def test_a_pipe_names_repeated_stages():
  p = Pipe(Noise(0.1), Scale(2.0), Noise(0.2))
  assert list(p.named_stages()) == ["noise", "scale", "noise_2"]
  assert float(p(torch.zeros(1, 1)).abs()) <= 0.6


# Observation groups.


def test_obs_concatenates_in_order_and_knows_its_slices():
  s = state()
  s.joint_pos[:] = 2.0
  s.foot_contact[:, 0] = True
  obs = Obs(
      joint_pos="joint_pos",
      contact="foot_contact",
      doubled=(lambda st: st.joint_pos, Scale(2.0)),
      length="episode_length",
  )
  out = obs(s)
  assert out.shape == (4, 2 + 2 + 2 + 1) and obs.dim == 7
  assert obs.slices() == {"joint_pos": slice(0, 2), "contact": slice(2, 4),
                          "doubled": slice(4, 6), "length": slice(6, 7)}
  assert out[0].tolist() == [2.0, 2.0, 1.0, 0.0, 4.0, 4.0, 0.0]
  assert out.dtype is torch.float32


def test_obs_reset_reaches_every_delay():
  s = state()
  d = Delay(1)
  obs = Obs(a=("joint_pos", d), b="commands")
  obs(s)
  obs.reset(torch.tensor([2]))
  assert d._pending.tolist() == [False, False, True, False]


def test_flattened_sources_keep_their_width():
  s = state()
  obs = Obs(feet="foot_pos")
  assert obs(s).shape == (4, 6)


# What rlmcp serves.


def test_blocks_are_found_on_the_env_and_their_leaves_walked():
  class Env:
    def __init__(self):
      self.state = state()
      self.actor_obs = Obs(joint_pos=("joint_pos", Delay(1, max_steps=4), Noise(0.01)),
                           commands="commands")
      self.action_pipe = Pipe(Clip(-1.0, 1.0), Scale(0.25))
      self.not_a_block = 3

  env = Env()
  assert list(blocks.blocks(env)) == ["actor_obs", "action_pipe"]
  leaves = [(path, f, static) for path, _, f, static in blocks.block_leaves(env.actor_obs)]
  assert leaves == [(("joint_pos", "delay"), "steps", False),
                    (("joint_pos", "delay"), "max_steps", True),
                    (("joint_pos", "noise"), "half_width", False)]
  leaves = [(path, f) for path, _, f, _ in blocks.block_leaves(env.action_pipe)]
  assert leaves == [(("clip",), "low"), (("clip",), "high"), (("scale",), "factor")]
