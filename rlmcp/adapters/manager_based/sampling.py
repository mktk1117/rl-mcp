"""Per-step state sampling for one environment.

Recording a trace means pulling a dozen signals off the GPU every step. Doing
that field by field would cost a dozen synchronisations; instead every channel
is concatenated into one tensor, copied once, and split back on the host. That
is the difference between a trace you can take during training and one you
cannot.

Channel names and semantics come from the trace vocabulary in
:mod:`rlmcp.adapters.base` (the ``CHANNEL_*`` constants); labels are derived
from the live environment, never assumed from the task.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from rlmcp.adapters.base import (
    CHANNEL_ACTION,
    CHANNEL_BASE_ANG_VEL,
    CHANNEL_BASE_LIN_VEL,
    CHANNEL_BASE_POS,
    CHANNEL_COMMAND,
    CHANNEL_FOOT_CONTACT,
    CHANNEL_JOINT_ACC,
    CHANNEL_JOINT_POS,
    CHANNEL_JOINT_TORQUE,
    CHANNEL_JOINT_VEL,
    CHANNEL_PROJECTED_GRAVITY,
    CHANNEL_REWARD,
    NotSupported,
)

# Honest component names for a plane-velocity command, in channel order.
_VELOCITY_COMMAND_LABELS = ["cmd_vx", "cmd_vy", "cmd_wz"]


def first_command(env: Any) -> tuple[str, Any, torch.Tensor] | None:
  """The first command term that yields a tensor: ``(name, term, command)``.

  Terms are checked in manager order, and a term whose command is ``None`` is
  skipped rather than ending the search -- a bare first term must not hide the
  ones behind it.
  """
  manager = getattr(env, "command_manager", None)
  if manager is None:
    return None
  for name in getattr(manager, "active_terms", []) or []:
    command = manager.get_command(name)
    if command is None:
      continue
    get_term = getattr(manager, "get_term", None)
    term = get_term(name) if callable(get_term) else None
    return name, term, command
  return None


def is_velocity_command(term: Any, command: torch.Tensor) -> bool:
  """Whether a command term is a plane-velocity twist ``[vx, vy(, wz)]``.

  ``CHANNEL_COMMAND`` means exactly that (see :mod:`rlmcp.adapters.base`), so
  the term itself has to vouch: named like a velocity command, or carrying the
  ``vel_command_b`` buffer mjlab's (and IsaacLab's) velocity commands keep, and
  2-3 wide. A motion target or goal pose fails this and is recorded under its
  own name instead of being misread as a velocity.
  """
  if int(command.shape[-1]) not in (2, 3):
    return False
  named = term is not None and "velocity" in type(term).__name__.lower()
  return named or hasattr(term, "vel_command_b")


class StateSampler:
  """Reads one environment's signals, one GPU sync per step."""

  def __init__(self, env: Any, robot_name: str):
    self.env = env
    self.robot_name = robot_name
    self._contact_candidates = self._rank_contact_sensors(env)
    self._contact_sensor_name: str | None = None
    self._contact_resolved = False

  @property
  def robot(self) -> Any:
    return self.env.scene[self.robot_name]

  # Contact sensor resolution.

  @staticmethod
  def _rank_contact_sensors(env: Any) -> list[str]:
    """Order sensors by how likely they are to be the foot-contact sensor.

    Names alone are ambiguous -- a G1 scene has both ``feet_ground_contact``
    (what we want) and ``foot_height_scan`` (a raycast with no contact flags) --
    so the final choice is confirmed against the sensor's actual data.
    """
    sensors = getattr(env.scene, "sensors", {}) or {}
    scored = []
    for name in sensors:
      lowered = name.lower()
      score = 0
      if "contact" in lowered:
        score += 2
      if "feet" in lowered or "foot" in lowered:
        score += 2
      if "self" in lowered or "collision" in lowered:
        score -= 3
      if "scan" in lowered or "height" in lowered:
        score -= 3
      if score > 0:
        scored.append((score, name))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [name for _, name in scored]

  def contact_found(self) -> torch.Tensor | None:
    """Foot contact flags, resolving (once) which sensor actually provides them."""
    if self._contact_resolved:
      candidates = [self._contact_sensor_name] if self._contact_sensor_name else []
    else:
      candidates = self._contact_candidates

    for name in candidates:
      try:
        found = getattr(self.env.scene[name].data, "found", None)
      except Exception:
        continue
      if found is None:
        continue
      self._contact_sensor_name = name
      self._contact_resolved = True
      return found
    self._contact_resolved = True
    self._contact_sensor_name = None
    return None

  # Command resolution, shared between data and labels so they cannot diverge.

  def _resolve_command(self) -> tuple[str, torch.Tensor, list[str]] | None:
    """``(channel, tensor, labels)`` for the first command term, if any.

    A plane-velocity command records as ``CHANNEL_COMMAND`` with its true
    component names; anything else records as ``command_<term>`` with numbered
    components -- honest about what is known, never wrong about what is not.
    """
    resolved = first_command(self.env)
    if resolved is None:
      return None
    name, term, command = resolved
    width = int(command.shape[-1])
    if is_velocity_command(term, command):
      return CHANNEL_COMMAND, command, _VELOCITY_COMMAND_LABELS[:width]
    return f"command_{name}", command, [f"cmd_{i}" for i in range(width)]

  # Sampling.

  def sample(self, env_id: int = 0) -> dict[str, np.ndarray]:
    env_id = int(max(0, min(env_id, int(self.env.num_envs) - 1)))
    data = self.robot.data
    pieces: list[tuple[str, torch.Tensor]] = []

    def add(name: str, tensor: torch.Tensor | None) -> None:
      if tensor is None:
        return
      pieces.append((name, tensor[env_id].reshape(-1).float()))

    add(CHANNEL_JOINT_POS, getattr(data, "joint_pos", None))
    add(CHANNEL_JOINT_VEL, getattr(data, "joint_vel", None))
    add(CHANNEL_JOINT_ACC, getattr(data, "joint_acc", None))
    try:
      add(CHANNEL_JOINT_TORQUE, data.actuator_force)
    except Exception:
      pass
    add(CHANNEL_BASE_LIN_VEL, getattr(data, "root_link_lin_vel_b", None))
    add(CHANNEL_BASE_ANG_VEL, getattr(data, "root_link_ang_vel_b", None))
    add(CHANNEL_PROJECTED_GRAVITY, getattr(data, "projected_gravity_b", None))
    add(CHANNEL_BASE_POS, getattr(data, "root_link_pos_w", None))

    action_manager = getattr(self.env, "action_manager", None)
    if action_manager is not None:
      add(CHANNEL_ACTION, action_manager.action)

    command = self._resolve_command()
    if command is not None:
      channel, tensor, _ = command
      add(channel, tensor)

    found = self.contact_found()
    if found is not None:
      add(CHANNEL_FOOT_CONTACT, (found > 0).float())

    reward_buf = getattr(self.env, "reward_buf", None)
    if reward_buf is None:
      reward_buf = getattr(getattr(self.env, "reward_manager", None), "_reward_buf", None)
    if reward_buf is not None:
      add(CHANNEL_REWARD, reward_buf.reshape(-1, 1))

    if not pieces:
      raise NotSupported(
          "sample_state found nothing to record: this environment exposes "
          "none of the trace channels (no robot joint/base data, actions, "
          "commands, contacts or rewards)."
      )

    flat = torch.cat([tensor for _, tensor in pieces]).detach().cpu().numpy()
    out: dict[str, np.ndarray] = {}
    offset = 0
    for name, tensor in pieces:
      width = tensor.shape[0]
      out[name] = flat[offset : offset + width]
      offset += width
    return out

  # Labels.

  def _action_labels(self) -> list[str] | None:
    """Per-component action names, walked from the manager's own terms.

    The policy's action vector is the manager's terms concatenated in order,
    so the labels are too. mjlab's ``BaseAction`` keeps the names of the
    actuator targets it controls in ``_target_names`` (mjlab-private;
    IsaacLab's joint actions keep the same list as ``_joint_names``); a term
    that offers neither gets its slice numbered rather than guessed at.
    """
    manager = getattr(self.env, "action_manager", None)
    if manager is None:
      return None
    try:
      labels: list[str] = []
      for term_name in getattr(manager, "active_terms", []) or []:
        term = manager.get_term(term_name)
        width = int(term.action_dim)
        names = getattr(term, "_target_names", None) or getattr(
            term, "_joint_names", None
        )
        if names is not None and len(names) == width:
          labels.extend(f"act:{n}" for n in names)
        else:
          labels.extend(f"{term_name}_{i}" for i in range(width))
      if labels:
        return labels
    except Exception:
      pass
    # Terms could not be walked; number the live action vector instead.
    action = getattr(manager, "action", None)
    if action is None:
      return None
    return [f"act_{i}" for i in range(int(action.shape[-1]))]

  def labels(self) -> dict[str, list[str]]:
    """Component names for the channels :meth:`sample` produces."""
    joints = list(self.robot.joint_names)
    labels = {
        CHANNEL_JOINT_POS: joints,
        CHANNEL_JOINT_VEL: joints,
        CHANNEL_JOINT_ACC: joints,
        CHANNEL_JOINT_TORQUE: joints,
        CHANNEL_BASE_LIN_VEL: ["vx", "vy", "vz"],
        CHANNEL_BASE_ANG_VEL: ["wx", "wy", "wz"],
        CHANNEL_PROJECTED_GRAVITY: ["gx", "gy", "gz"],
        CHANNEL_BASE_POS: ["x", "y", "z"],
        CHANNEL_REWARD: ["reward"],
    }
    action_labels = self._action_labels()
    if action_labels:
      labels[CHANNEL_ACTION] = action_labels
    command = self._resolve_command()
    if command is not None:
      channel, _, command_labels = command
      labels[channel] = command_labels
    found = self.contact_found()
    if found is not None:
      width = int(found[0].reshape(-1).shape[0])
      labels[CHANNEL_FOOT_CONTACT] = self._contact_labels(width)
    return labels

  def _contact_labels(self, width: int) -> list[str]:
    """Per-column contact names, derived from the sensor rather than assumed.

    mjlab's ContactSensor names its columns (``primary_names``, one per
    matched geom/body; IsaacLab's keeps ``body_names``), and those are used
    when they match the channel's width. A robot is not assumed to be a biped:
    when the sensor offers no usable names, columns are numbered ``foot_0..n``
    -- the same honest-labels rule as the action and command channels.
    """
    sensor = None
    if self._contact_sensor_name:
      try:
        sensor = self.env.scene[self._contact_sensor_name]
      except Exception:
        sensor = None
    names = getattr(sensor, "primary_names", None) or getattr(
        sensor, "body_names", None
    )
    try:
      if names is not None and len(names) == width:
        return [str(n) for n in names]
    except TypeError:
      pass  # Not a sized collection; fall through to numbering.
    return [f"foot_{i}" for i in range(width)]
