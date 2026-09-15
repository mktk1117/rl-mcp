"""The plain-MuJoCo player: every term it knows, on a model it builds itself.

No mjlab, no torch, no GPU. A small floating body with two hinge joints and
the sensors a locomotion policy reads, a hand-written spec in the shape
`rlmcp bundle export` writes, and a numpy policy. The tests pin the term
functions, the compute → clip → scale → history pipeline, what the player
refuses, and what `rlmcp bundle play` answers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from rlmcp import bundle_play  # noqa: E402
from rlmcp.bundle_play import (  # noqa: E402
  BundleError,
  Player,
  decode_value,
  quat_rotate_inverse,
  resolve_names,
  rollout,
  unsupported_terms,
)

MJCF = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body name="base" pos="0 0 0.5">
      <freejoint name="root"/>
      <site name="imu"/>
      <geom type="box" size="0.1 0.1 0.05" mass="1"/>
      <body name="thigh" pos="0 0 -0.05">
        <joint name="hip" type="hinge" axis="0 1 0" range="-1 1"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.02" mass="0.2"/>
        <body name="shank" pos="0 0 -0.2">
          <joint name="knee" type="hinge" axis="0 1 0" range="-1 1"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.02" mass="0.2"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="hip_act" joint="hip" kp="20"/>
    <position name="knee_act" joint="knee" kp="20"/>
  </actuator>
  <sensor>
    <gyro name="robot/imu_ang_vel" site="imu"/>
    <velocimeter name="robot/imu_lin_vel" site="imu"/>
  </sensor>
</mujoco>
"""


def _spec(model: mujoco.MjModel, **overrides) -> dict:
  sensors = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, s):
             {"adr": int(model.sensor_adr[s]), "dim": int(model.sensor_dim[s])}
             for s in range(model.nsensor)}
  spec = {
      "spec_version": 1, "task": "Test-Hopper", "checkpoint": "model_1.pt",
      "timing": {"physics_dt": 0.005, "decimation": 4},
      "default_entity": "robot",
      "entities": {"robot": {
          "joint_names": ["hip", "knee"],
          "joint_qpos_adr": [7, 8], "joint_qvel_adr": [6, 7],
          "root_qpos_adr": [0, 1, 2, 3, 4, 5, 6], "root_qvel_adr": [0, 1, 2, 3, 4, 5],
          "default_joint_pos": [0.1, -0.2], "default_joint_vel": [0.0, 0.0],
          "initial_root_pose": [0, 0, 0.5, 1, 0, 0, 0], "root_body_id": 1,
      }},
      "observation": {"groups": [{"name": "actor", "terms": [
          {"name": "ang_vel", "func": "mjlab.envs.mdp.observations.builtin_sensor",
           "params": {"sensor_name": "robot/imu_ang_vel"}, "dim": [3], "scale": [0.25, 0.25, 0.25]},
          {"name": "gravity", "func": "mjlab.envs.mdp.observations.projected_gravity",
           "params": {}, "dim": [3]},
          {"name": "joint_pos", "func": "mjlab.envs.mdp.observations.joint_pos_rel",
           "params": {}, "dim": [2], "clip": [-0.5, 0.5], "scale": 2.0},
          {"name": "joint_vel", "func": "mjlab.envs.mdp.observations.joint_vel_rel",
           "params": {}, "dim": [2]},
          {"name": "cmd", "func": "mjlab.envs.mdp.observations.generated_commands",
           "params": {"command_name": "base_velocity"}, "dim": [3]},
          {"name": "actions", "func": "mjlab.envs.mdp.observations.last_action",
           "params": {}, "dim": [2]},
      ]}]},
      "actions": [{"name": "joint_pos", "kind": "joint_position", "dim": 2, "entity": "robot",
                   "joint_local_ids": [0, 1], "joint_names": ["hip", "knee"],
                   "scale": [0.5, 0.5], "offset": [0.1, -0.2], "ctrl_ids": [0, 1]}],
      "commands": {"base_velocity": {"dim": 3, "default": [0.0, 0.0, 0.0]}},
      "sensors": sensors,
      "clip_actions": 1.0,
  }
  spec.update(overrides)
  return spec


@pytest.fixture
def model():
  return mujoco.MjModel.from_xml_string(MJCF)


def _zero_policy(obs):
  return np.zeros((1, 2), dtype=np.float32)


# ── terms ────────────────────────────────────────────────────────────────
def test_the_terms_read_the_state_the_way_the_training_stack_does(model):
  player = Player(model, _spec(model), _zero_policy)
  # Tilt the base 90° about y (w, x, y, z), give it a body-frame spin and a
  # world-frame velocity, bend the joints.
  q = np.array([0, 0, 0.5, np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0, 0.3, -0.1])
  v = np.zeros(model.nv)
  v[0:3] = [1.0, 0.0, 0.0]        # world x, forward
  v[3:6] = [0.0, 0.0, 2.0]        # body-frame yaw rate
  v[6:8] = [0.5, -0.5]
  player.reset(q, v)

  gravity = player.term("actor", player.groups[0]["terms"][1])
  # Rotated 90° about y, world "down" is along the body's +x axis.
  assert (np.allclose(gravity, [-1.0, 0.0, 0.0], atol=1e-6)
          or np.allclose(gravity, [1.0, 0.0, 0.0], atol=1e-6))
  assert np.allclose(quat_rotate_inverse(np.array([1, 0, 0, 0]), np.array([0, 0, -1])), [0, 0, -1])

  joint_pos = player.term("actor", player.groups[0]["terms"][2])
  # (0.3-0.1, -0.1+0.2) = (0.2, 0.1), clipped to ±0.5, then x2: clip before scale.
  assert np.allclose(joint_pos, [0.4, 0.2])
  joint_vel = player.term("actor", player.groups[0]["terms"][3])
  assert np.allclose(joint_vel, [0.5, -0.5])

  base_ang = bundle_play.term_base_ang_vel(player, {"params": {}})
  assert np.allclose(base_ang, [0.0, 0.0, 2.0]), "a free joint's angular velocity is body-frame"
  base_lin = bundle_play.term_base_lin_vel(player, {"params": {}})
  assert np.isclose(np.linalg.norm(base_lin), 1.0)
  gyro = player.term("actor", player.groups[0]["terms"][0])
  assert gyro.shape == (3,) and np.allclose(gyro, 0.25 * player.data.sensordata[0:3])


def test_clip_comes_before_scale_and_a_subset_of_joints_is_by_name(model):
  spec = _spec(model)
  term = spec["observation"]["groups"][0]["terms"][2]
  term["params"] = {"asset_cfg": {"name": "robot", "joint_names": ["knee"]}}
  spec["observation"]["groups"][0]["terms"][2]["dim"] = [1]
  spec["observation"]["groups"][0]["terms"][2]["clip"] = [-0.05, 0.05]
  player = Player(model, spec, _zero_policy)
  q = player.data.qpos.copy()
  q[8] = 0.3                          # knee at 0.3, default -0.2 → 0.5 → clip 0.05 -> x2
  player.reset(q)
  assert np.allclose(player.term("actor", spec["observation"]["groups"][0]["terms"][2]), [0.1])
  assert resolve_names(["kn.*"], ["hip", "knee"]) == [1]
  assert resolve_names(None, ["hip", "knee"]) == [0, 1]
  with pytest.raises(BundleError):
    resolve_names(["ankle"], ["hip", "knee"])


def test_history_backfills_from_the_first_observation_and_runs_oldest_to_newest(model):
  spec = _spec(model)
  term = spec["observation"]["groups"][0]["terms"][3]
  term["history_length"] = 3
  term["dim"] = [6]
  player = Player(model, spec, _zero_policy)
  q = player.data.qpos.copy()
  v = np.zeros(model.nv)
  v[6:8] = [1.0, 1.0]
  player.reset(q, v)
  first = player.term("actor", term)
  assert np.allclose(first, [1, 1, 1, 1, 1, 1]), "the buffer starts full of the first value"
  player.data.qvel[6:8] = [2.0, 2.0]
  second = player.term("actor", term)
  assert np.allclose(second, [1, 1, 1, 1, 2, 2]), "oldest first, newest last"
  player.reset(q, v)
  assert np.allclose(player.term("actor", term), [1, 1, 1, 1, 1, 1]), "a reset empties the history"


def test_observe_concatenates_every_group_in_order_and_step_applies_actions(model):
  spec = _spec(model)
  calls = []

  def policy(obs):
    calls.append(obs.copy())
    return np.array([[4.0, -4.0]], dtype=np.float32)   # clipped to ±1

  player = Player(model, spec, policy)
  player.set_command("base_velocity", [0.5, 0.0, 0.0])
  obs = player.step()
  assert obs.shape == (15,) and obs.dtype == np.float32
  assert np.allclose(obs[10:13], [0.5, 0.0, 0.0]), "the command is what was set"
  assert np.allclose(obs[13:15], [0.0, 0.0]), "no action yet on the first step"
  assert np.allclose(player.last_action, [1.0, -1.0]), "clip_actions applied"
  # target = raw * scale + offset, onto the actuators the spec names
  assert np.allclose(player.data.ctrl, [1.0 * 0.5 + 0.1, -1.0 * 0.5 - 0.2])
  assert player.steps == 1 and np.isclose(player.control_dt, 0.02)
  second = player.step()
  assert np.allclose(second[13:15], [1.0, -1.0]), "last_action is the previous raw action"
  with pytest.raises(BundleError):
    player.set_command("base_velocity", [1.0])
  with pytest.raises(BundleError):
    player.command("no_such_command")


# ── refusals ─────────────────────────────────────────────────────────────
def test_a_term_the_player_does_not_know_is_named_and_refused(model):
  spec = _spec(model)
  spec["observation"]["groups"][0]["terms"].append(
      {"name": "scan", "func": "mjlab.envs.mdp.observations.height_scan",
       "params": {}, "dim": [187]})
  spec["actions"].append({"name": "wrist", "kind": "unsupported:SiteAction", "dim": 3})
  problems = unsupported_terms(spec)
  assert problems == ["actor.scan: height_scan is not a term this player knows",
                      "action wrist: kind 'unsupported:SiteAction' is not supported"]
  with pytest.raises(BundleError, match="height_scan"):
    Player(model, spec, _zero_policy)


def test_a_delayed_term_is_refused(model):
  spec = _spec(model)
  spec["observation"]["groups"][0]["terms"][0]["delay_max_lag"] = 2
  assert unsupported_terms(spec) == ["actor.ang_vel: observation delay is not supported"]


def test_decode_value_undoes_the_capture_encoding():
  encoded = {"__map__": {"asset_cfg": {"__obj__": {"module": "mjlab.managers.scene_entity_config",
                                                    "name": "SceneEntityCfg",
                                                    "fields": {"name": "robot",
                                                               "joint_names": {"__tuple__": [".*"]},
                                                               "joint_ids": {"__slice_all__": True},
                                                               }}},
                         "scale": {"__tuple__": [1, 2]}, "n": 3}}
  decoded = decode_value(encoded)
  assert decoded["asset_cfg"]["name"] == "robot" and decoded["asset_cfg"]["joint_names"] == (".*",)
  assert decoded["asset_cfg"]["__class__"].endswith("SceneEntityCfg")
  assert decoded["scale"] == (1, 2) and decoded["n"] == 3
  assert decode_value({"__repr__": "<thing>", "__unrenderable__": True}) == "<thing>"


# ── a bundle on disk ─────────────────────────────────────────────────────
def _write_bundle(tmp_path: Path, model: mujoco.MjModel) -> Path:
  out = tmp_path / "bundle"
  out.mkdir()
  mujoco.mj_saveModel(model, str(out / bundle_play.MODEL_FILE))
  (out / bundle_play.SPEC_FILE).write_text(json.dumps(_spec(model)))
  return out


def test_rollout_plays_a_bundle_directory_and_reports_the_root_height(tmp_path, model):
  out = _write_bundle(tmp_path, model)
  report = rollout(out, seconds=0.5, policy=_zero_policy, commands={"base_velocity": [0.2, 0, 0]})
  assert report["steps"] == 25 and report["seconds"] == 0.5
  assert report["obs_dim"] == 15 and report["action_dim"] == 2
  assert report["root_height"]["start"] == 0.5
  assert report["root_height"]["min"] <= report["root_height"]["start"], "a zero policy falls"
  assert report["realtime_factor"] is None or report["realtime_factor"] > 0


def test_a_bundle_without_its_files_or_from_the_future_is_refused(tmp_path, model):
  with pytest.raises(BundleError, match=r"spec\.json"):
    rollout(tmp_path / "missing", policy=_zero_policy)
  out = _write_bundle(tmp_path, model)
  (out / bundle_play.MODEL_FILE).unlink()
  with pytest.raises(BundleError, match=r"model\.mjb"):
    rollout(out, policy=_zero_policy)
  out2 = tmp_path / "future"
  out2.mkdir()
  (out2 / bundle_play.SPEC_FILE).write_text(json.dumps({"spec_version": 99}))
  with pytest.raises(BundleError, match="spec version 99"):
    bundle_play.read_spec(out2)


def test_an_onnx_policy_needs_onnxruntime_and_says_so_otherwise(tmp_path, model, monkeypatch):
  out = _write_bundle(tmp_path, model)
  import builtins
  real_import = builtins.__import__

  def no_ort(name, *args, **kwargs):
    if name == "onnxruntime":
      raise ImportError("no onnxruntime")
    return real_import(name, *args, **kwargs)
  monkeypatch.setattr(builtins, "__import__", no_ort)
  with pytest.raises(BundleError, match="onnxruntime"):
    bundle_play.load_player(out)
