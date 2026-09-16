"""Genesis: the same robot, from the same MJCF, on a different physics engine.

Genesis reads MJCF as readily as URDF, so the :class:`RobotSpec` needs no
second asset. What it does not read is the MuJoCo-only part of the contract
-- actuators and touch sensors -- so those are set up through Genesis's own
API: PD gains and force limits on the dofs of the actuated joints, and
contact from the net contact force on each contact site's link. The plain
MuJoCo model is still compiled, on the CPU, purely to answer questions about
the file (which body a site is on, joint limits) with one code path.

Two Genesis facts shape the rest:

* ``gs.init`` is per process and cannot be called twice, so it is guarded.
* A camera must exist before ``scene.build``, and this backend builds the
  scene, so it adds one -- observing (``debug=True``), aimed at environment
  0. Every environment is built at the same place, so the render of any
  ``env_id`` is a view of that one pile; the scene draws environment 0 only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from rlmcp.backends import frames
from rlmcp.backends.base import (
    RobotSpec,
    SimBackend,
    SimOptions,
    compile_model,
    gain_for,
)

_initialised: dict[str, Any] = {}
_RIGID_OPTIONS = ("constraint_solver", "max_collision_pairs", "enable_self_collision",
                  "enable_joint_limit")


def _init_genesis(device: torch.device) -> Any:
  import genesis as gs

  if not _initialised:
    backend = gs.gpu if device.type == "cuda" else gs.cpu
    try:
      gs.init(backend=backend, precision="32", logging_level="warning")
    except Exception as exc:  # Already initialised by the training script.
      if "already" not in str(exc).lower():
        raise
    _initialised["device"] = gs.device
  return gs


class GenesisBackend(SimBackend):
  name = "genesis"

  def __init__(
      self,
      robot: RobotSpec,
      num_envs: int,
      dt: float,
      decimation: int,
      device: str | torch.device = "cuda",
      options: SimOptions | None = None,
  ):
    super().__init__(robot, num_envs, dt, decimation, device, options)
    try:
      gs = _init_genesis(self.device)
    except ImportError as exc:  # pragma: no cover - depends on the install.
      raise ImportError(
          "The genesis backend needs the `genesis-world` package: pip install genesis-world"
      ) from exc
    self._gs = gs

    self._model, self._layout = compile_model(robot, self.options)
    extra = dict(self.options.extra)
    self._scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=self.control_dt, substeps=self.decimation),
        rigid_options=gs.options.RigidOptions(
            iterations=int(self.options.iterations),
            ls_iterations=int(self.options.ls_iterations),
            **{k: v for k, v in extra.items() if k in _RIGID_OPTIONS},
        ),
        vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
        show_viewer=False,
    )
    if self._layout.ground_added:
      self._scene.add_entity(gs.morphs.Plane())
    self._robot = self._scene.add_entity(gs.morphs.MJCF(file=robot.xml))
    self._camera = self._scene.add_camera(
        res=(640, 480), pos=(1.6, -1.2, 0.9), lookat=(0.0, 0.0, 0.3), GUI=False, debug=True,
    )
    self._scene.build(n_envs=self.num_envs, env_spacing=(0.0, 0.0))

    # Dofs of the actuated joints, in spec order.
    names = self._layout.joint_names
    self._motor_dofs = [int(self._robot.get_joint(n).dofs_idx_local[0]) for n in names]
    kp = [gain_for(robot.stiffness, n) for n in names]
    kd = [gain_for(robot.damping, n) for n in names]
    self._robot.set_dofs_kp(kp, self._motor_dofs)
    self._robot.set_dofs_kv(kd, self._motor_dofs)
    if robot.effort_limit is not None:
      limit = [gain_for(robot.effort_limit, n) for n in names]
      self._robot.set_dofs_force_range([-v for v in limit], limit, self._motor_dofs)

    self._base_dofs = next(
        [int(i) for i in j.dofs_idx_local] for j in self._robot.joints if j.n_dofs == 6
    )
    self._contact_links = [
        int(self._robot.get_link(b).idx_local) for b in self._layout.contact_bodies]
    # Contact sites are body-fixed points; Genesis reports links, so each site
    # is its link's pose plus the offset the MJCF gave it.
    self._site_offsets = torch.as_tensor(
        self._model.site_pos[self._layout.contact_site_ids], dtype=torch.float32,
        device=self._gs.device)
    foot_bodies = {self._model.body(self._model.geom_bodyid[g]).name
                   for g in self._layout.foot_geom_ids}
    self._foot_links = [int(self._robot.get_link(b).idx_local) for b in sorted(foot_bodies)] \
        or list(range(self._robot.n_links))
    self._limits = torch.as_tensor(self._layout.limits, dtype=torch.float32, device=self.device)
    self._default_friction = float(self._model.geom_friction[:, 0].mean()) or 1.0
    self._snapshot()

  # Helpers.

  def _t(self, value: Any) -> Tensor:
    tensor = torch.as_tensor(value)
    if tensor.device != self.device:
      tensor = tensor.to(self.device)
    return tensor.to(torch.float32)

  def _g(self, value: Tensor) -> Tensor:
    """A tensor where Genesis wants it."""
    return value.detach().to(self._gs.device, torch.float32)

  def _snapshot(self) -> None:
    robot = self._robot
    self._s_root_pos = self._t(robot.get_pos())
    self._s_root_quat = self._t(robot.get_quat())
    self._s_root_lin_vel = self._t(robot.get_vel())
    self._s_root_ang_vel = self._t(robot.get_ang())
    self._s_dof_pos = self._t(robot.get_dofs_position(self._motor_dofs))
    self._s_dof_vel = self._t(robot.get_dofs_velocity(self._motor_dofs))
    self._s_dof_torque = self._t(robot.get_dofs_control_force(self._motor_dofs))
    forces = self._t(robot.get_links_net_contact_force())
    self._s_contact = torch.linalg.norm(forces[:, self._contact_links, :], dim=-1)
    link_pos = torch.as_tensor(robot.get_links_pos(self._contact_links))
    link_quat = torch.as_tensor(robot.get_links_quat(self._contact_links))
    offset = self._site_offsets.unsqueeze(0).expand(link_pos.shape[0], -1, -1)
    self._s_site_pos = self._t(link_pos + frames.quat_rotate(link_quat, offset))

  # What the robot is.

  @property
  def joint_names(self) -> list[str]:
    return list(self._layout.joint_names)

  @property
  def dof_limits(self) -> Tensor:
    return self._limits

  @property
  def scene(self) -> Any:
    return self._scene

  @property
  def entity(self) -> Any:
    return self._robot

  # State.

  @property
  def root_pos(self) -> Tensor:
    return self._s_root_pos

  @property
  def root_quat(self) -> Tensor:
    return self._s_root_quat

  @property
  def root_lin_vel(self) -> Tensor:
    return self._s_root_lin_vel

  @property
  def root_ang_vel(self) -> Tensor:
    return self._s_root_ang_vel

  @property
  def dof_pos(self) -> Tensor:
    return self._s_dof_pos

  @property
  def dof_vel(self) -> Tensor:
    return self._s_dof_vel

  @property
  def dof_torque(self) -> Tensor:
    return self._s_dof_torque

  @property
  def contact_forces(self) -> Tensor:
    return self._s_contact

  @property
  def contact_site_pos(self) -> Tensor:
    return self._s_site_pos

  # Control.

  def push(self, env_ids: Tensor, lin_vel: Tensor, ang_vel: Tensor) -> None:
    ids = self._ids(env_ids)
    if ids.size == 0:
      return
    gs_ids = torch.as_tensor(ids, device=self._gs.device, dtype=torch.int32)
    current = torch.as_tensor(self._robot.get_dofs_velocity(self._base_dofs, envs_idx=gs_ids))
    delta = torch.cat([self._g(lin_vel), self._g(ang_vel)], dim=-1)
    self._robot.set_dofs_velocity(current + delta, self._base_dofs, envs_idx=gs_ids)
    self._snapshot()

  def set_dof_targets(self, targets: Tensor) -> None:
    self._robot.control_dofs_position(self._g(targets), self._motor_dofs)

  def step(self) -> None:
    self._scene.step()
    self._snapshot()

  def reset(
      self,
      env_ids: Tensor,
      root_pos: Tensor,
      root_quat: Tensor,
      dof_pos: Tensor,
      dof_vel: Tensor | None = None,
      root_lin_vel: Tensor | None = None,
      root_ang_vel: Tensor | None = None,
  ) -> None:
    ids = self._ids(env_ids)
    if ids.size == 0:
      return
    gs_ids = torch.as_tensor(ids, device=self._gs.device, dtype=torch.int32)
    robot = self._robot
    robot.set_pos(self._g(root_pos), envs_idx=gs_ids, relative=False, zero_velocity=True)
    robot.set_quat(self._g(root_quat), envs_idx=gs_ids, relative=False, zero_velocity=True)
    robot.set_dofs_position(self._g(dof_pos), self._motor_dofs, envs_idx=gs_ids, zero_velocity=True)
    robot.zero_all_dofs_velocity(envs_idx=gs_ids)
    if dof_vel is not None:
      robot.set_dofs_velocity(self._g(dof_vel), self._motor_dofs, envs_idx=gs_ids)
    if root_lin_vel is not None or root_ang_vel is not None:
      vel = torch.zeros(len(ids), 6, device=self._gs.device)
      if root_lin_vel is not None:
        vel[:, :3] = self._g(root_lin_vel)
      if root_ang_vel is not None:
        vel[:, 3:] = self._g(root_ang_vel)
      robot.set_dofs_velocity(vel, self._base_dofs, envs_idx=gs_ids)
    robot.control_dofs_position(self._g(dof_pos), self._motor_dofs, envs_idx=gs_ids)
    self._snapshot()

  # Optional.

  def set_friction(self, env_ids: Tensor, coefficient: Tensor) -> None:
    ids = torch.as_tensor(self._ids(env_ids), device=self._gs.device, dtype=torch.int32)
    ratio = (self._g(coefficient) / self._default_friction).reshape(-1, 1)
    ratio = ratio.expand(-1, len(self._foot_links)).contiguous()
    self._robot.set_friction_ratio(ratio, self._foot_links, envs_idx=ids)

  def render(self, env_id: int = 0, width: int = 640, height: int = 480) -> np.ndarray:
    frame = self._camera.render(rgb=True)[0]
    array = np.asarray(frame)
    if array.ndim == 4:
      array = array[0]
    array = array[..., :3]
    if array.dtype != np.uint8:
      array = np.clip(array * 255.0 if array.max() <= 1.0 else array, 0, 255).astype(np.uint8)
    return array

  def close(self) -> None:
    pass


__all__ = ["GenesisBackend"]
