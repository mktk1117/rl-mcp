"""Quaternion and frame helpers shared by the backends and single-file envs.

Every quaternion here is ``(w, x, y, z)``, which is what MuJoCo and Genesis
both use. Every function is pure and batched over the leading dimension.
"""

from __future__ import annotations

import torch
from torch import Tensor


def quat_rotate(q: Tensor, v: Tensor) -> Tensor:
  """Rotate ``v`` (*, 3) by ``q`` (*, 4): body frame to world frame."""
  q_w = q[..., 0:1]
  q_vec = q[..., 1:4]
  t = 2.0 * torch.cross(q_vec, v, dim=-1)
  return v + q_w * t + torch.cross(q_vec, t, dim=-1)


def quat_rotate_inverse(q: Tensor, v: Tensor) -> Tensor:
  """Rotate ``v`` (*, 3) by the inverse of ``q`` (*, 4): world to body frame."""
  q_w = q[..., 0:1]
  q_vec = q[..., 1:4]
  t = 2.0 * torch.cross(q_vec, v, dim=-1)
  return v - q_w * t + torch.cross(q_vec, t, dim=-1)


def quat_conjugate(q: Tensor) -> Tensor:
  return torch.cat([q[..., 0:1], -q[..., 1:4]], dim=-1)


def quat_from_euler_xyz(roll: Tensor, pitch: Tensor, yaw: Tensor) -> Tensor:
  """Euler angles (*,) each, XYZ convention, to quaternions (*, 4)."""
  cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
  cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
  cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
  return torch.stack(
      [
          cr * cp * cy + sr * sp * sy,
          sr * cp * cy - cr * sp * sy,
          cr * sp * cy + sr * cp * sy,
          cr * cp * sy - sr * sp * cy,
      ],
      dim=-1,
  )


def projected_gravity(base_quat: Tensor) -> Tensor:
  """The world's unit gravity vector seen from the base frame, (N, 3).

  ``(0, 0, -1)`` when the base is upright; the x and y components measure
  tilt, which is what a flat-orientation reward and a fall check read.
  """
  gravity = torch.tensor([0.0, 0.0, -1.0], device=base_quat.device, dtype=base_quat.dtype)
  return quat_rotate_inverse(base_quat, gravity.expand(base_quat.shape[0], -1))


def wrap_to_pi(angles: Tensor) -> Tensor:
  return (angles + torch.pi) % (2 * torch.pi) - torch.pi


__all__ = [
    "projected_gravity",
    "quat_conjugate",
    "quat_from_euler_xyz",
    "quat_rotate",
    "quat_rotate_inverse",
    "wrap_to_pi",
]
