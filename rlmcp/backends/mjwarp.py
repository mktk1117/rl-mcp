"""MuJoCo Warp: every environment on the GPU, one kernel launch per step.

`mujoco_warp <https://github.com/google-deepmind/mujoco_warp>`_ puts a
MuJoCo model and ``nworld`` copies of its data on the GPU and steps them all
at once. The shape of this backend is distilled from mjlab's ``Simulation``:
model and data go up once, ``step``/``forward``/``reset`` are captured as CUDA
graphs where the device allows it, and torch reads the arrays in place.

Streams are the part that bites. Warp launches on its own CUDA stream and
torch on its own, so a torch read right after a Warp step can see the state
from before it. Every read here happens on Warp's stream and is copied into
a torch-owned tensor there; the calling stream then waits for those copies.
Every write does the reverse. It costs a handful of small copies per control
step and removes a whole class of "the robot teleports sometimes" bugs.

Per-world model fields (for friction randomisation) are asked for at
``put_model`` time through ``batch_sizes``, so no array is ever replaced
after the graphs are captured.
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np
import torch
from torch import Tensor

from rlmcp.backends import frames
from rlmcp.backends.base import RobotSpec, SimBackend, SimOptions, compile_model


class MjWarpBackend(SimBackend):
  name = "mjwarp"

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
      import mujoco
      import mujoco_warp as mjw
      import warp as wp
    except ImportError as exc:  # pragma: no cover - depends on the install.
      raise ImportError(
          "The mjwarp backend needs `mujoco_warp` and `warp-lang`: "
          "pip install warp-lang 'mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp'"
      ) from exc
    self._wp = wp
    self._mjw = mjw

    self._model, self._layout = compile_model(robot, self.options)
    self._model.opt.timestep = self.dt
    self._mj_data = mujoco.MjData(self._model)
    mujoco.mj_forward(self._model, self._mj_data)

    wp.init()
    self._wp_device = wp.get_device(str(self.device))
    with wp.ScopedDevice(self._wp_device):
      self._wp_model = mjw.put_model(
          self._model, batch_sizes={"geom_friction": self.num_envs},
      )
      # A solver that hits its line-search budget prints a warning per step,
      # which at 50 Hz is a log that grows by gigabytes an hour. mjlab's
      # settings hit it routinely and mjlab silences it the same way.
      self._wp_model.opt.warn_overflow = 0
      extra = dict(self.options.extra)
      self._wp_data = mjw.put_data(
          self._model, self._mj_data, nworld=self.num_envs,
          nconmax=self.options.nconmax, njmax=self.options.njmax,
          **{k: v for k, v in extra.items() if k in ("nccdmax", "naconmax", "naccdmax", "nvmax")},
      )
      self._reset_mask = wp.zeros(self.num_envs, dtype=bool)

    self._cuda = not self._wp_device.is_cpu
    self._stream = (
        torch.cuda.ExternalStream(wp.get_stream(self._wp_device).cuda_stream)
        if self._cuda else None
    )
    # Torch views over the Warp arrays, made once. Reads copy out of these on
    # Warp's stream; writes copy into them there.
    self._qpos = wp.to_torch(self._wp_data.qpos)
    self._qvel = wp.to_torch(self._wp_data.qvel)
    self._ctrl = wp.to_torch(self._wp_data.ctrl)
    self._sensordata = wp.to_torch(self._wp_data.sensordata)
    self._actuator_force = wp.to_torch(self._wp_data.actuator_force)
    self._site_xpos = wp.to_torch(self._wp_data.site_xpos)
    self._geom_friction = wp.to_torch(self._wp_model.geom_friction)
    self._reset_mask_t = wp.to_torch(self._reset_mask)

    self._qpos_adr = torch.as_tensor(self._layout.qpos_adr, device=self.device, dtype=torch.long)
    self._qvel_adr = torch.as_tensor(self._layout.qvel_adr, device=self.device, dtype=torch.long)
    self._act_ids = torch.as_tensor(self._layout.actuator_ids, device=self.device, dtype=torch.long)
    self._contact_adr = torch.as_tensor(
        self._layout.contact_sensor_adr, device=self.device, dtype=torch.long)
    self._site_ids = torch.as_tensor(
        self._layout.contact_site_ids, device=self.device, dtype=torch.long)
    self._contact_geoms = torch.as_tensor(
        self._layout.contact_geom_ids, device=self.device, dtype=torch.long)
    self._limits = torch.as_tensor(self._layout.limits, dtype=torch.float32, device=self.device)
    self._default = torch.as_tensor(
        self._layout.default_dof_pos, dtype=torch.float32, device=self.device)

    self._graphs: dict[str, Any] = {}
    self._capture_graphs()
    self._renderer: Any = None
    self._render_data: Any = None
    self._snapshot()
    self.announce(self._layout)

  # Streams.

  def _on_warp_stream(self):
    if self._stream is None:
      return contextlib.nullcontext()
    # Torch work queued so far must land before Warp reads or writes.
    self._stream.wait_stream(torch.cuda.current_stream(self.device))
    return torch.cuda.stream(self._stream)

  def _back_to_torch(self) -> None:
    if self._stream is not None:
      torch.cuda.current_stream(self.device).wait_stream(self._stream)

  # Graphs.

  def _capture_graphs(self) -> None:
    wp, mjw = self._wp, self._mjw
    self._graphs = {}
    if not self._cuda or not wp.is_mempool_enabled(self._wp_device):
      return
    with wp.ScopedDevice(self._wp_device):
      with wp.ScopedCapture() as capture:
        mjw.step(self._wp_model, self._wp_data)
      self._graphs["step"] = capture.graph
      with wp.ScopedCapture() as capture:
        mjw.forward(self._wp_model, self._wp_data)
      self._graphs["forward"] = capture.graph
      with wp.ScopedCapture() as capture:
        mjw.reset_data(self._wp_model, self._wp_data, reset=self._reset_mask)
      self._graphs["reset"] = capture.graph

  def _launch(self, which: str) -> None:
    wp, mjw = self._wp, self._mjw
    with wp.ScopedDevice(self._wp_device):
      graph = self._graphs.get(which)
      if graph is not None:
        wp.capture_launch(graph)
      elif which == "step":
        mjw.step(self._wp_model, self._wp_data)
      elif which == "forward":
        mjw.forward(self._wp_model, self._wp_data)
      else:
        mjw.reset_data(self._wp_model, self._wp_data, reset=self._reset_mask)

  # State: copied out once per step/reset, read many times.

  def _snapshot(self) -> None:
    layout = self._layout
    q, v = layout.base_qpos, layout.base_qvel
    with self._on_warp_stream():
      qpos, qvel = self._qpos, self._qvel
      if layout.floating:
        self._s_root_pos = qpos[:, q:q + 3].clone()
        self._s_root_quat = qpos[:, q + 3:q + 7].clone()
        self._s_root_lin_vel = qvel[:, v:v + 3].clone()
        self._s_root_ang_vel_body = qvel[:, v + 3:v + 6].clone()
      self._s_dof_pos = qpos[:, self._qpos_adr].clone()
      self._s_dof_vel = qvel[:, self._qvel_adr].clone()
      self._s_dof_torque = self._actuator_force[:, self._act_ids].clone()
      self._s_contact = self._sensordata[:, self._contact_adr].clone()
      self._s_site_pos = self._site_xpos[:, self._site_ids, :].clone()
    self._back_to_torch()

  @property
  def joint_names(self) -> list[str]:
    return list(self._layout.joint_names)

  @property
  def dof_limits(self) -> Tensor:
    return self._limits

  @property
  def default_dof_pos(self) -> Tensor:
    return self._default

  @property
  def floating_base(self) -> bool:
    return self._layout.floating

  @property
  def contact_names(self) -> list[str]:
    return list(self._layout.contact_names)

  @property
  def mj_model(self) -> Any:
    return self._model

  @property
  def wp_model(self) -> Any:
    return self._wp_model

  @property
  def wp_data(self) -> Any:
    return self._wp_data

  @property
  def root_pos(self) -> Tensor:
    self._require_floating("root_pos")
    return self._s_root_pos

  @property
  def root_quat(self) -> Tensor:
    self._require_floating("root_quat")
    return self._s_root_quat

  @property
  def root_lin_vel(self) -> Tensor:
    self._require_floating("root_lin_vel")
    return self._s_root_lin_vel

  @property
  def root_ang_vel(self) -> Tensor:
    self._require_floating("root_ang_vel")
    return frames.quat_rotate(self._s_root_quat, self._s_root_ang_vel_body)

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
    self._require_floating("push")
    ids = torch.as_tensor(self._ids(env_ids), device=self.device, dtype=torch.long)
    if ids.numel() == 0:
      return
    v = self._layout.base_qvel
    dev = self.device
    body = frames.quat_rotate_inverse(self._s_root_quat[ids], ang_vel.to(dev, torch.float32))
    with self._on_warp_stream():
      self._qvel[ids, v:v + 3] += lin_vel.to(dev, torch.float32)
      self._qvel[ids, v + 3:v + 6] += body
    self._snapshot()

  def set_dof_targets(self, targets: Tensor) -> None:
    with self._on_warp_stream():
      self._ctrl[:, self._act_ids] = targets.to(self.device, torch.float32)

  def step(self) -> None:
    with self._on_warp_stream():
      pass  # Order any pending torch writes before the launches.
    for _ in range(self.decimation):
      self._launch("step")
    self._launch("forward")
    self._snapshot()

  def reset(
      self,
      env_ids: Tensor,
      dof_pos: Tensor,
      dof_vel: Tensor | None = None,
      root_pos: Tensor | None = None,
      root_quat: Tensor | None = None,
      root_lin_vel: Tensor | None = None,
      root_ang_vel: Tensor | None = None,
  ) -> None:
    ids = torch.as_tensor(self._ids(env_ids), device=self.device, dtype=torch.long)
    if ids.numel() == 0:
      return
    q, v = self._layout.base_qpos, self._layout.base_qvel
    dev = self.device
    with self._on_warp_stream():
      self._reset_mask_t.fill_(False)
      self._reset_mask_t[ids] = True
    self._launch("reset")  # Back to the model's spawn pose, velocities zero.
    with self._on_warp_stream():
      qpos, qvel = self._qpos, self._qvel
      qpos[ids[:, None], self._qpos_adr[None, :]] = dof_pos.to(dev, torch.float32)
      if dof_vel is not None:
        qvel[ids[:, None], self._qvel_adr[None, :]] = dof_vel.to(dev, torch.float32)
      if self._layout.floating:
        if root_pos is not None:
          qpos[ids, q:q + 3] = root_pos.to(dev, torch.float32)
        if root_quat is not None:
          qpos[ids, q + 3:q + 7] = root_quat.to(dev, torch.float32)
        if root_lin_vel is not None:
          qvel[ids, v:v + 3] = root_lin_vel.to(dev, torch.float32)
        if root_ang_vel is not None:
          body = frames.quat_rotate_inverse(qpos[ids, q + 3:q + 7], root_ang_vel.to(dev))
          qvel[ids, v + 3:v + 6] = body.to(torch.float32)
      self._ctrl[ids[:, None], self._act_ids[None, :]] = dof_pos.to(dev, torch.float32)
    self._launch("forward")
    self._snapshot()

  # Optional.

  def set_friction(self, env_ids: Tensor, coefficient: Tensor) -> None:
    ids = torch.as_tensor(self._ids(env_ids), device=self.device, dtype=torch.long)
    value = coefficient.to(self.device, torch.float32).reshape(-1, 1)
    with self._on_warp_stream():
      if self._contact_geoms.numel():
        self._geom_friction[ids[:, None], self._contact_geoms[None, :], 0] = value
      else:
        self._geom_friction[ids, :, 0] = value
    self._back_to_torch()

  def render(self, env_id: int = 0, width: int = 640, height: int = 480) -> np.ndarray:
    import mujoco

    from rlmcp.backends.mjbatch import _tracking_camera

    if self._renderer is None or self._renderer.width != width or self._renderer.height != height:
      self._renderer = mujoco.Renderer(self._model, height, width)
      self._render_data = mujoco.MjData(self._model)
    with self._on_warp_stream():
      qpos = self._qpos[int(env_id)].detach().cpu().numpy()
      qvel = self._qvel[int(env_id)].detach().cpu().numpy()
    self._back_to_torch()
    data = self._render_data
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(self._model, data)
    camera = _tracking_camera(self._model, data)
    self._renderer.update_scene(data, camera=camera)
    return np.asarray(self._renderer.render(), dtype=np.uint8)

  def close(self) -> None:
    if self._renderer is not None:
      self._renderer.close()
    self._renderer = None
    self._graphs = {}


__all__ = ["MjWarpBackend"]
