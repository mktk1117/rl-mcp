"""mjbatch: thousands of C MuJoCo simulations on CPU threads.

`mjbatch <https://github.com/kevinzakka/mjbatch>`_ runs one ``MjData`` per
environment on a thread pool with the GIL released, and hands back live numpy
views of every field. That makes this the simplest backend of the three --
state is read straight out of ``qpos``/``qvel``/``sensordata`` -- and the one
that needs no GPU. It is also the only one whose physics is bit-for-bit
MuJoCo's, which makes it the reference the others are checked against.

Tensors are copied out of the numpy views on every read (``float32``, on
:attr:`device`). At a few thousand environments that is a few hundred
kilobytes per control step, which the simulation itself dwarfs. Writes go
the other way, straight into the views.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from rlmcp.backends import frames
from rlmcp.backends.base import RobotSpec, SimBackend, SimOptions, compile_model


class MjBatchBackend(SimBackend):
  name = "mjbatch"

  def __init__(
      self,
      robot: RobotSpec,
      num_envs: int,
      dt: float,
      decimation: int,
      device: str | torch.device = "cpu",
      options: SimOptions | None = None,
  ):
    super().__init__(robot, num_envs, dt, decimation, device, options)
    try:
      import mjbatch
    except ImportError as exc:  # pragma: no cover - depends on the install.
      raise ImportError(
          "The mjbatch backend needs the `mjbatch` package: pip install mjbatch"
      ) from exc

    self._model, self._layout = compile_model(robot, self.options)
    self._model.opt.timestep = self.dt
    self._batch = mjbatch.Batch(
        self._model, self.num_envs, num_threads=int(self.options.num_threads), forward=True,
    )
    self._qpos = self._batch.bind("qpos")
    self._qvel = self._batch.bind("qvel")
    self._ctrl = self._batch.bind("ctrl")
    self._sensordata = self._batch.bind("sensordata")
    self._actuator_force = self._batch.bind("actuator_force")
    self._limits = torch.as_tensor(self._layout.limits, dtype=torch.float32, device=self.device)
    self._renderer: Any = None
    self._render_data: Any = None
    # Every simulation starts at the model's neutral pose; derived fields current.
    self._batch.reset()

  # What the robot is.

  @property
  def joint_names(self) -> list[str]:
    return list(self._layout.joint_names)

  @property
  def dof_limits(self) -> Tensor:
    return self._limits

  @property
  def mj_model(self) -> Any:
    return self._model

  @property
  def batch(self) -> Any:
    """The underlying ``mjbatch.Batch``, for anything the contract lacks."""
    return self._batch

  # State.

  def _t(self, array: np.ndarray) -> Tensor:
    return torch.as_tensor(np.ascontiguousarray(array, dtype=np.float32), device=self.device)

  @property
  def root_pos(self) -> Tensor:
    q = self._layout.base_qpos
    return self._t(self._qpos[:, q:q + 3])

  @property
  def root_quat(self) -> Tensor:
    q = self._layout.base_qpos
    return self._t(self._qpos[:, q + 3:q + 7])

  @property
  def root_lin_vel(self) -> Tensor:
    v = self._layout.base_qvel
    return self._t(self._qvel[:, v:v + 3])

  @property
  def root_ang_vel(self) -> Tensor:
    v = self._layout.base_qvel
    return frames.quat_rotate(self.root_quat, self._t(self._qvel[:, v + 3:v + 6]))

  @property
  def dof_pos(self) -> Tensor:
    return self._t(self._qpos[:, self._layout.qpos_adr])

  @property
  def dof_vel(self) -> Tensor:
    return self._t(self._qvel[:, self._layout.qvel_adr])

  @property
  def dof_torque(self) -> Tensor:
    return self._t(self._actuator_force[:, self._layout.actuator_ids])

  @property
  def contact_forces(self) -> Tensor:
    return self._t(self._sensordata[:, self._layout.contact_sensor_adr])

  # Control.

  def set_dof_targets(self, targets: Tensor) -> None:
    self._ctrl[:, self._layout.actuator_ids] = targets.detach().cpu().numpy()

  def step(self) -> None:
    self._batch.step(nstep=self.decimation)

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
    self._batch.reset(ids)
    q, v = self._layout.base_qpos, self._layout.base_qvel
    self._qpos[ids, q:q + 3] = root_pos.detach().cpu().numpy()
    self._qpos[ids, q + 3:q + 7] = root_quat.detach().cpu().numpy()
    self._qpos[np.ix_(ids, self._layout.qpos_adr)] = dof_pos.detach().cpu().numpy()
    self._qvel[ids] = 0.0
    if dof_vel is not None:
      self._qvel[np.ix_(ids, self._layout.qvel_adr)] = dof_vel.detach().cpu().numpy()
    if root_lin_vel is not None:
      self._qvel[ids, v:v + 3] = root_lin_vel.detach().cpu().numpy()
    if root_ang_vel is not None:
      body = frames.quat_rotate_inverse(root_quat, root_ang_vel)
      self._qvel[ids, v + 3:v + 6] = body.detach().cpu().numpy()
    self._ctrl[np.ix_(ids, self._layout.actuator_ids)] = dof_pos.detach().cpu().numpy()
    self._batch.forward(ids)

  # Optional.

  def set_friction(self, env_ids: Tensor, coefficient: Tensor) -> None:
    ids = self._ids(env_ids)
    friction = self._batch.expand("geom_friction")
    friction[ids, :, 0] = coefficient.detach().cpu().numpy().reshape(-1, 1)
    self._batch.set_const(ids)

  def render(self, env_id: int = 0, width: int = 640, height: int = 480) -> np.ndarray:
    import mujoco

    if self._renderer is None or self._renderer.width != width or self._renderer.height != height:
      self._renderer = mujoco.Renderer(self._model, height, width)
      self._render_data = mujoco.MjData(self._model)
    data = self._render_data
    data.qpos[:] = self._qpos[int(env_id)]
    data.qvel[:] = self._qvel[int(env_id)]
    mujoco.mj_forward(self._model, data)
    camera = _tracking_camera(self._model, data, self._layout.base_body)
    self._renderer.update_scene(data, camera=camera)
    return np.asarray(self._renderer.render(), dtype=np.uint8)

  def close(self) -> None:
    if self._renderer is not None:
      self._renderer.close()
    self._renderer = None


def _tracking_camera(model: Any, data: Any, base_body: str) -> Any:
  """A free camera looking at the base from a little behind and above."""
  import mujoco

  camera = mujoco.MjvCamera()
  camera.type = mujoco.mjtCamera.mjCAMERA_FREE
  base = data.xpos[model.body(base_body).id]
  camera.lookat[:] = base
  camera.distance = 1.6
  camera.azimuth = 135.0
  camera.elevation = -20.0
  return camera


__all__ = ["MjBatchBackend"]
