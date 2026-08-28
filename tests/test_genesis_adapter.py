"""The Genesis adapter, against a fake env shaped like Genesis's.

No Genesis, no GPU, no display. What these pin is the part that is genuinely
Genesis's -- how a frame is asked for from a camera the training script had to
create before it built the scene, and how episodes are restarted through a
reset that takes a mask rather than a list -- plus the claim the whole split
rests on: the shared legged-gym-shaped machinery works against Genesis's shapes
without knowing which simulator it is talking to.

The fake camera mirrors the real API as read from `genesis/vis/camera.py`:
``render(rgb=True)`` returns ``(rgb, depth, seg, normal)``, ``set_pose`` takes
``pos``/``lookat``, and cameras carry a ``debug`` flag.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import FakeCamera, FakeScene

from rlmcp.adapters.base import NotSupported
from rlmcp.adapters.genesis import GenesisSimAdapter, wrap
from rlmcp.core.parameters.spec import ParameterCategory


@pytest.fixture
def sim(genesis_env) -> GenesisSimAdapter:
  return GenesisSimAdapter(genesis_env)


# The shared machinery, reached through the Genesis adapter.


def test_parameters_are_discovered_without_describing_the_task(sim):
  keys = {spec.key for spec in sim.discover_parameters()}
  assert "reward.tracking_lin_vel.weight" in keys
  assert "command.lin_vel_x" in keys
  assert "action.scale" in keys


def test_an_edit_lands_where_the_environment_reads_it(sim, genesis_env):
  sim.set_parameter("reward.action_rate.weight", -0.02)
  assert genesis_env.reward_scales["action_rate"] == pytest.approx(
      -0.02 * genesis_env.dt)


def test_a_command_range_is_written_where_the_sampler_reads(sim, genesis_env):
  sim.set_parameter("command.lin_vel_x", [1.0, 2.0])
  assert genesis_env.sample_command(0) == (1.0, 2.0)


def test_the_basics_come_off_the_environment(sim, genesis_env):
  assert sim.num_envs() == genesis_env.num_envs
  assert sim.step_dt() == pytest.approx(genesis_env.dt)
  assert sim.joint_names() == genesis_env.env_cfg["joint_names"]
  assert sim.max_episode_length() == pytest.approx(1000)


def test_traces_and_metrics_come_through_the_adapter(sim):
  assert sim.sample_state()
  assert all(k.startswith("rlmcp/") for k in sim.summary_metrics())


def test_a_command_that_is_not_a_velocity_is_not_published_as_one(genesis_env):
  """The channel names come from command_cfg, so renaming them changes what
  the adapter is willing to call a velocity command."""
  genesis_env.command_cfg = {
      "goal_x_range": [0.0, 1.0],
      "goal_y_range": [0.0, 1.0],
      "goal_yaw_range": [0.0, 1.0],
  }
  assert "command" not in GenesisSimAdapter(genesis_env).sample_state()


# Resetting: a mask, not a list.


def test_resetting_converts_ids_into_the_mask_genesis_expects(sim, genesis_env):
  result = sim.reset_envs([1, 3])
  assert result["num_reset"] == 2
  mask = genesis_env.reset_calls[-1]
  assert mask.dtype == torch.bool
  assert mask.tolist() == [False, True, False, True]


def test_resetting_everything_passes_none_rather_than_a_full_mask(sim, genesis_env):
  """None is Genesis's own "all of them", and it takes a cheaper path."""
  result = sim.reset_envs(None)
  assert genesis_env.reset_calls[-1] is None
  assert result["num_reset"] == genesis_env.num_envs


def test_an_environment_that_does_not_exist_is_refused_by_number(sim, genesis_env):
  with pytest.raises(ValueError, match="No environment 9"):
    sim.reset_envs([9])
  assert not genesis_env.reset_calls, "a refused reset resets nothing"


def test_resetting_an_empty_selection_is_not_resetting_everything(sim, genesis_env):
  assert sim.reset_envs([])["num_reset"] == 0
  assert not genesis_env.reset_calls


# Frames.


def test_a_scene_with_no_camera_says_what_to_do_about_it(genesis_env):
  genesis_env.scene = FakeScene(cameras=[])
  sim = GenesisSimAdapter(genesis_env)
  assert sim.renderer_ready() is False
  with pytest.raises(NotSupported) as excinfo:
    sim.render()
  message = str(excinfo.value)
  assert "add_camera" in message and "before" in message


def test_an_observing_camera_is_preferred_over_a_sensor(genesis_env):
  """A debug camera records without joining the robot's sensor set; a camera
  that changes what the policy sees is not an observation of the run."""
  sensor, observer = FakeCamera(debug=False), FakeCamera(debug=True)
  genesis_env.scene = FakeScene(cameras=[sensor, observer],
                                rendered_envs_idx=[0, 1, 2, 3])
  GenesisSimAdapter(genesis_env).render(env_id=1)
  assert observer.poses and not sensor.poses


def test_a_frame_comes_back_as_rgb(sim):
  frame = sim.render()
  assert frame.ndim == 3 and frame.shape[-1] == 3
  assert frame.dtype == np.uint8


def test_a_frame_of_one_env_is_actually_a_frame_of_that_env(sim, genesis_env):
  camera = genesis_env.scene.visualizer.cameras[0]
  sim.render(env_id=2)
  aimed = camera.poses[0][1]
  assert aimed[0] == pytest.approx(20.0), "the camera should be on env 2's robot"


def test_the_camera_is_put_back_after_a_screenshot(sim, genesis_env):
  """A screenshot should not quietly re-aim a camera the script set up."""
  camera = genesis_env.scene.visualizer.cameras[0]
  before = (camera.pos, camera.lookat)
  sim.render(env_id=3)
  assert (camera.pos, camera.lookat) == before


def test_the_framing_the_script_chose_is_kept_when_the_subject_changes(sim, genesis_env):
  camera = genesis_env.scene.visualizer.cameras[0]
  offset = np.subtract(camera.pos, camera.lookat)
  sim.render(env_id=1)
  pos, lookat = camera.poses[0]
  assert np.allclose(np.subtract(pos, lookat), offset)


def test_an_env_the_scene_does_not_draw_is_refused_not_answered(genesis_env):
  """Answering with env 0 would be a picture that looks like an answer."""
  genesis_env.scene = FakeScene(cameras=[FakeCamera(debug=True)],
                                rendered_envs_idx=[0])
  with pytest.raises(NotSupported, match="not rendered"):
    GenesisSimAdapter(genesis_env).render(env_id=2)


def test_a_batched_renderer_returns_one_frame_not_a_stack(genesis_env):
  genesis_env.scene = FakeScene(cameras=[FakeCamera(debug=True, batched=True)],
                                rendered_envs_idx=[0, 1, 2, 3])
  assert GenesisSimAdapter(genesis_env).render().ndim == 3


# Wrapping.


def test_wrapping_gives_a_run_the_agent_can_reach(genesis_env, tmp_path):
  env = wrap(genesis_env, session_dir=tmp_path / "rlmcp", task_id="go2-walk")
  status = env.rlmcp.cmd_status()
  assert status["iteration"] == 0
  assert {s.key for s in env.rlmcp.sim.discover_parameters()}
  assert env.unwrapped is genesis_env


def test_a_run_with_no_camera_says_so_at_wrap_time(genesis_env, tmp_path, capsys):
  genesis_env.scene = FakeScene(cameras=[])
  wrap(genesis_env, session_dir=tmp_path / "rlmcp")
  assert "add_camera" in capsys.readouterr().out


def test_the_dt_assumption_is_stated_rather_than_left_implicit(
    genesis_env, tmp_path, capsys):
  """rlmcp cannot check this one: the multiplication already happened."""
  wrap(genesis_env, session_dir=tmp_path / "rlmcp")
  assert "dt" in capsys.readouterr().out


def test_categories_survive_the_trip_through_the_adapter(sim):
  by_key = {s.key: s for s in sim.discover_parameters()}
  assert by_key["command.lin_vel_x"].category is ParameterCategory.CURRICULUM


# Regressions from the first run against real Genesis. Both of these passed on
# a fake and failed on the robot, so each fake below reproduces the *mechanism*
# that broke rather than the shape of the fix.


def test_resetting_works_on_buffers_made_under_inference_mode(genesis_env):
  """rsl_rl collects rollouts inside torch.inference_mode().

  That makes the environment's state buffers inference tensors, and torch
  refuses an in-place write to one from outside inference mode -- which is
  exactly where rlmcp services commands. On a real run this failed with
  "Inplace update to inference tensor outside InferenceMode is not allowed";
  nothing about an ordinary fake tensor reproduces it, so the fake has to be
  built the way the runner builds them.
  """
  with torch.inference_mode():
    genesis_env.dof_pos = torch.zeros_like(genesis_env.dof_pos)
    genesis_env.base_pos = torch.zeros_like(genesis_env.base_pos)

  def reset(mask=None):
    # What Genesis's _reset_idx does: write the buffers in place.
    genesis_env.dof_pos.zero_()
    genesis_env.base_pos.zero_()
    genesis_env.reset_calls.append(mask)

  genesis_env._reset_idx = reset
  result = GenesisSimAdapter(genesis_env).reset_envs([1])
  assert result["num_reset"] == 1


def test_action_rate_is_omitted_when_the_env_has_already_synced_it(genesis_env):
  """Go2Env ends step() with last_actions.copy_(actions).

  So at the iteration boundary the two buffers are identical and the rate is
  exactly zero -- which is not "the policy is smooth" but "the question cannot
  be asked here". It read as a perfectly smooth policy on a robot that was
  visibly buzzing.
  """
  genesis_env.actions = torch.full_like(genesis_env.actions, 0.4)
  genesis_env.last_actions = genesis_env.actions.clone()
  assert "rlmcp/action_rate_rms" not in GenesisSimAdapter(genesis_env).summary_metrics()


def test_action_rate_is_reported_when_the_env_does_keep_the_previous_action(
    genesis_env):
  """A fork that keeps them distinct still gets the metric."""
  genesis_env.actions = torch.full_like(genesis_env.actions, 0.4)
  genesis_env.last_actions = torch.full_like(genesis_env.actions, 0.1)
  metrics = GenesisSimAdapter(genesis_env).summary_metrics()
  assert metrics["rlmcp/action_rate_rms"] == pytest.approx(0.3)


def test_the_drawn_env_list_comes_from_what_the_renderer_reads(genesis_env):
  """VisOptions holds what was asked for; the context holds what is drawn.

  On a real run those disagreed -- the options said [0] while the renderer was
  drawing four -- and `shot --env-id 2` was refused for an environment that
  was on screen. A false refusal is the same class of mistake as answering
  with the wrong env, in the other direction.
  """
  genesis_env.scene = FakeScene(cameras=[FakeCamera(debug=True)],
                                rendered_envs_idx=[0, 1, 2, 3],
                                options_say=[0])
  assert GenesisSimAdapter(genesis_env).render(env_id=2).ndim == 3


def test_no_restriction_recorded_means_no_env_is_refused(genesis_env):
  """rendered_envs_idx defaults to None, which means all of them."""
  genesis_env.scene = FakeScene(cameras=[FakeCamera(debug=True)],
                                rendered_envs_idx=None)
  assert GenesisSimAdapter(genesis_env).render(env_id=3).ndim == 3
