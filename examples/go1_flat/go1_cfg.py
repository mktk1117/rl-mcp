"""Genesis's stock quadruped config, retargeted to a Go1.

Only what the robot forces changes: the asset, the default pose, the spawn and
the base-height target. Rewards, gains, termination limits, observation scales
and the PPO settings are Genesis's own, so a difference between a Go1 run here
and Genesis's Go2 is a difference between two robots rather than two recipes.

The asset is mjlab's ``go1.xml`` -- the same file mjlab drives
``Mjlab-Velocity-Flat-Unitree-Go1`` from. One robot description feeding two
backends beats two that drift apart.
"""

from __future__ import annotations

import os
from pathlib import Path

MJLAB_GO1_XML = "MJLAB_GO1_XML"
"""Point this at mjlab's go1.xml if it is not importable from this interpreter."""

# From mjlab's go1_constants.INIT_STATE: thighs 0.9, calves -1.8, hips splayed
# 0.1 outward. This is the pose mjlab's own Go1 task spawns in.
GO1_DEFAULT_ANGLES = {
    "FL_hip_joint": -0.1, "FR_hip_joint": 0.1,
    "RL_hip_joint": -0.1, "RR_hip_joint": 0.1,
    "FL_thigh_joint": 0.9, "FR_thigh_joint": 0.9,
    "RL_thigh_joint": 0.9, "RR_thigh_joint": 0.9,
    "FL_calf_joint": -1.8, "FR_calf_joint": -1.8,
    "RL_calf_joint": -1.8, "RR_calf_joint": -1.8,
}

GO1_STANDING_HEIGHT = 0.278
"""mjlab spawns its Go1 here, so it is the height this pose actually stands at."""


def find_go1_xml() -> Path:
  """mjlab's Go1 description, however this machine has mjlab.

  Asked of the installed package first so the example works without anyone
  editing a path, with an environment variable for a checkout that is not
  importable from this interpreter -- which is the normal case, since Genesis
  and mjlab rarely share a virtualenv.
  """
  override = os.environ.get(MJLAB_GO1_XML)
  if override:
    return Path(override).expanduser()
  try:
    import mjlab
    root = Path(mjlab.__file__).parent
  except ImportError:
    raise SystemExit(
        "This example uses mjlab's Go1 description, and mjlab is not importable "
        f"here. Set ${MJLAB_GO1_XML} to the path of go1.xml -- it lives at "
        "src/mjlab/asset_zoo/robots/unitree_go1/xmls/go1.xml in an mjlab "
        "checkout. Genesis ships a Go2 but no Go1, which is why this is needed."
    ) from None
  found = root / "asset_zoo/robots/unitree_go1/xmls/go1.xml"
  if not found.exists():
    raise SystemExit(f"mjlab is installed but has no Go1 at {found}.")
  return found


def get_cfgs(action_scale: float | None = None,
             action_rate_weight: float | None = None):
  """The Go1 config. Overrides default to Genesis's stock numbers.

  The campaign in this repo's run records found `action_scale=0.20` roughly
  halves the high-frequency joint content at ~2% of tracking speed; see
  README.md. Stock is kept as the default so the example reproduces Genesis's
  own behaviour unless asked otherwise.
  """
  from go2_train import get_cfgs as stock

  env_cfg, obs_cfg, reward_cfg, command_cfg = stock()

  env_cfg["robot_file"] = str(find_go1_xml())
  env_cfg["default_joint_angles"] = dict(GO1_DEFAULT_ANGLES)
  # Spawn a little above standing so first contact is a settle, exactly as the
  # stock Go2 config spawns 0.42 for a ~0.32 stance.
  env_cfg["base_init_pos"] = [0.0, 0.0, GO1_STANDING_HEIGHT + 0.10]
  # The Go1 is the smaller dog; a target copied from the Go2 would ask it to
  # stand on tiptoe for the whole run.
  reward_cfg["base_height_target"] = GO1_STANDING_HEIGHT

  if action_scale is not None:
    env_cfg["action_scale"] = action_scale
  if action_rate_weight is not None:
    reward_cfg["reward_scales"]["action_rate"] = action_rate_weight

  return env_cfg, obs_cfg, reward_cfg, command_cfg
