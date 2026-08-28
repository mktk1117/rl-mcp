"""Traces and batch metrics for a legged-gym-shaped environment.

The parameter surface decides what an agent can change; this decides what it
can see. Two things are worth pinning: that a channel the environment does not
keep is *absent* rather than filled in from something nearby, and that the
command buffer is only published as ``CHANNEL_COMMAND`` when it really is a
plane velocity -- because the tracking diagnostics subtract that channel from
the measured base velocity, and would happily do it to a goal pose.
"""

from __future__ import annotations

import pytest
import torch

from rlmcp.adapters.base import (
    CHANNEL_ACTION,
    CHANNEL_BASE_LIN_VEL,
    CHANNEL_COMMAND,
    CHANNEL_FOOT_CONTACT,
    CHANNEL_JOINT_POS,
    CHANNEL_JOINT_TORQUE,
    CHANNEL_JOINT_VEL,
    CHANNEL_REWARD,
    TRACE_CHANNELS,
)
from rlmcp.adapters.legged_gym_style.metrics import episode_log, summary_metrics
from rlmcp.adapters.legged_gym_style.sampling import StateSampler

VELOCITY_NAMES = ["lin_vel_x", "lin_vel_y", "ang_vel"]


@pytest.fixture
def sampler(flat_env) -> StateSampler:
  return StateSampler(flat_env, command_names=VELOCITY_NAMES)


# Channels.


def test_the_channels_come_off_the_flat_buffers(sampler):
  sample = sampler.sample(env_id=0)
  assert {CHANNEL_JOINT_POS, CHANNEL_JOINT_VEL, CHANNEL_ACTION,
          CHANNEL_BASE_LIN_VEL, CHANNEL_REWARD} <= set(sample)


def test_every_channel_emitted_is_one_the_diagnostics_know(sampler):
  """A backend that invents a spelling records traces that diagnose as empty."""
  for name in sampler.sample():
    assert name in TRACE_CHANNELS or name == "command_raw", name


def test_widths_are_per_env_not_per_batch(sampler, flat_env):
  sample = sampler.sample()
  joints = len(flat_env.env_cfg["joint_names"])
  assert sample[CHANNEL_JOINT_POS].shape == (joints,)
  assert sample[CHANNEL_BASE_LIN_VEL].shape == (3,)
  assert sample[CHANNEL_REWARD].shape == (1,), "a scalar buffer still has width 1"


def test_a_channel_the_env_does_not_keep_is_absent_not_faked(sampler):
  """Go2Env keeps neither contacts nor torques, so gait and effort are skipped
  rather than computed from whatever else is to hand."""
  sample = sampler.sample()
  assert CHANNEL_FOOT_CONTACT not in sample
  assert CHANNEL_JOINT_TORQUE not in sample


def test_the_sample_is_of_the_env_that_was_asked_for(sampler, flat_env):
  flat_env.dof_pos[2, 0] = 1.25
  assert sampler.sample(env_id=2)[CHANNEL_JOINT_POS][0] == pytest.approx(1.25)
  assert sampler.sample(env_id=0)[CHANNEL_JOINT_POS][0] == pytest.approx(0.0)


# What may be called a command.


def test_a_plane_velocity_command_is_published_as_the_command_channel(sampler):
  assert CHANNEL_COMMAND in sampler.sample()


def test_a_command_that_is_not_a_velocity_gets_its_own_name(flat_env):
  """CHANNEL_COMMAND means [vx, vy, wz] by definition. A goal pose under that
  name would be subtracted from the base velocity by the tracking section."""
  sampler = StateSampler(flat_env, command_names=["goal_x", "goal_y", "goal_yaw"])
  sample = sampler.sample()
  assert CHANNEL_COMMAND not in sample
  assert "command_raw" in sample


def test_unnamed_commands_are_not_assumed_to_be_velocities(flat_env):
  sampler = StateSampler(flat_env)
  assert CHANNEL_COMMAND not in sampler.sample()


# Labels.


def test_labels_come_from_the_environment(sampler, flat_env):
  labels = sampler.labels()
  assert labels[CHANNEL_JOINT_POS] == flat_env.env_cfg["joint_names"]
  assert labels[CHANNEL_COMMAND] == ["cmd_lin_vel_x", "cmd_lin_vel_y", "cmd_ang_vel"]


def test_a_width_that_does_not_match_the_names_is_left_unlabelled(flat_env):
  """Numbering is honest; a wrong label misdirects whoever reads the diagnosis."""
  flat_env.env_cfg["joint_names"] = ["only_one"]
  labels = StateSampler(flat_env).labels()
  assert CHANNEL_ACTION not in labels


def test_an_env_without_joint_names_is_numbered_rather_than_guessed(flat_env):
  del flat_env.env_cfg["joint_names"]
  assert StateSampler(flat_env).labels() == {}


# Batch metrics.


def test_metrics_are_prefixed_so_they_reach_the_headline(flat_env, sampler):
  metrics = summary_metrics(flat_env, sampler)
  assert metrics
  assert all(key.startswith("rlmcp/") for key in metrics)


def test_tracking_error_is_measured_against_the_command(flat_env, sampler):
  flat_env.commands[:, 0] = 1.0
  flat_env.base_lin_vel[:, 0] = 0.25
  metrics = summary_metrics(flat_env, sampler)
  assert metrics["rlmcp/lin_vel_error_mean"] == pytest.approx(0.75)
  assert metrics["rlmcp/commanded_speed_mean"] == pytest.approx(1.0)


def test_tracking_is_omitted_when_the_command_is_not_a_velocity(flat_env):
  sampler = StateSampler(flat_env, command_names=["goal_x", "goal_y", "goal_yaw"])
  assert "rlmcp/lin_vel_error_mean" not in summary_metrics(flat_env, sampler)


def test_one_missing_buffer_costs_one_metric_not_all_of_them(flat_env, sampler):
  del flat_env.last_actions
  metrics = summary_metrics(flat_env, sampler)
  assert "rlmcp/action_rate_rms" not in metrics
  assert "rlmcp/tilt_deg_mean" in metrics


def test_tilt_is_reported_in_degrees_from_upright(flat_env, sampler):
  assert summary_metrics(flat_env, sampler)["rlmcp/tilt_deg_mean"] == pytest.approx(0.0)
  flat_env.projected_gravity = torch.tensor([[-1.0, 0.0, 0.0]] * flat_env.num_envs)
  assert summary_metrics(flat_env, sampler)["rlmcp/tilt_deg_mean"] == pytest.approx(90.0)


# Episode logs.


def test_per_term_episode_means_become_telemetry(flat_env):
  flat_env.extras["episode"] = {
      "rew_tracking_lin_vel": torch.tensor(0.5),
      "rew_action_rate": -0.02,
  }
  assert episode_log(flat_env.extras) == pytest.approx(
      {"rew_tracking_lin_vel": 0.5, "rew_action_rate": -0.02}
  )


def test_nothing_but_the_episode_block_is_treated_as_telemetry(flat_env):
  """time_outs is a per-env tensor the runner consumes, not a scalar to plot."""
  flat_env.extras["time_outs"] = torch.zeros(flat_env.num_envs)
  assert episode_log(flat_env.extras) == {}


def test_a_run_before_its_first_reset_reports_nothing_rather_than_failing():
  assert episode_log({}) == {}
  assert episode_log(None) == {}
