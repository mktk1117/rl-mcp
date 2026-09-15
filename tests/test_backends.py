"""The physics backends, against a two-joint hopper nobody has to install.

The contract in :mod:`rlmcp.backends.base` is what a single-file environment
relies on to run unchanged on MuJoCo Warp, mjbatch and Genesis, so it is
tested once, in one place, against every backend the machine has. The robot
is a small MJCF written by the test -- a floating box on one leg with a wide
box foot -- so nothing here needs an asset zoo. mjlab's Go1 is used for one
cross-backend check when ``MJLAB_GO1_XML`` points at it.

What is skipped is skipped by name: a backend whose simulator is not
importable is reported as such, not as a pass. Genesis needs a GPU and a
minute to compile, so it runs only when ``RLMCP_TEST_GENESIS=1``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
mujoco = pytest.importorskip("mujoco")

from rlmcp.backends import BACKENDS, RobotSpec, SimOptions, available, make_backend  # noqa: E402
from rlmcp.backends.base import compile_model, gain_for  # noqa: E402

HOPPER = """
<mujoco model="hopper">
  <compiler angle="radian"/>
  <option timestep="0.005"/>
  <worldbody>
    <light pos="0 0 3"/>
    <body name="base" pos="0 0 0.6">
      <freejoint name="root"/>
      <geom name="base_box" type="box" size="0.1 0.1 0.05" mass="2.0"/>
      <body name="thigh" pos="0 0 -0.05">
        <joint name="hip" type="hinge" axis="0 1 0" range="-1.0 1.0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.02" mass="0.3"/>
        <body name="shin" pos="0 0 -0.25">
          <joint name="knee" type="hinge" axis="0 1 0" range="-1.5 0.0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.02" mass="0.2"/>
          <site name="foot" pos="0 0 -0.25" type="box" size="0.16 0.16 0.03"/>
          <geom name="foot_geom" type="box" pos="0 0 -0.25" size="0.15 0.15 0.02" mass="0.2"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def hopper_xml(tmp_path_factory) -> str:
  path = tmp_path_factory.mktemp("hopper") / "hopper.xml"
  path.write_text(HOPPER)
  return str(path)


@pytest.fixture
def spec(hopper_xml) -> RobotSpec:
  return RobotSpec(
      xml=hopper_xml,
      stiffness={"knee": 40.0, "*": 20.0},
      damping=1.0,
      effort_limit=30.0,
      contact_sites=("foot", "base"),
      foot_geoms=("foot_geom",),
  )


# The compile step: what every MuJoCo backend gets from a RobotSpec.


def test_gains_match_by_pattern_in_order():
  table = {"*_calf_joint": 35.0, "*": 20.0}
  assert gain_for(table, "FR_calf_joint") == 35.0
  assert gain_for(table, "FR_hip_joint") == 20.0
  assert gain_for(20.0, "anything") == 20.0
  with pytest.raises(KeyError):
    gain_for({"*_calf_joint": 35.0}, "FR_hip_joint")


def test_compile_adds_pd_actuators_touch_sensors_and_a_floor(spec):
  model, layout = compile_model(spec)
  assert layout.joint_names == ["hip", "knee"]
  assert model.nu == 2
  # A PD controller in MuJoCo's spelling: gain kp, biases (0, -kp, -kd).
  knee = model.actuator(layout.actuator_ids[1])
  assert knee.gainprm[0] == pytest.approx(40.0)
  assert list(knee.biasprm[:3]) == pytest.approx([0.0, -40.0, -1.0])
  assert int(model.actuator_biastype[layout.actuator_ids[1]]) == int(mujoco.mjtBias.mjBIAS_AFFINE)
  assert list(knee.forcerange) == pytest.approx([-30.0, 30.0])
  # One touch sensor per contact site; the body got a site around its geoms.
  assert model.nsensor == 2
  assert layout.contact_bodies == ["shin", "base"]
  assert model.site("base_contact").size[0] == pytest.approx(0.1)
  # The foot geom got priority, friction and the hardened solref.
  foot = model.geom("foot_geom")
  assert int(foot.priority[0]) == 1
  assert list(foot.friction) == pytest.approx([1.0, 0.005, 0.0001])
  assert list(foot.solref) == pytest.approx([0.01, 1.0])
  assert list(layout.foot_geom_ids) == [foot.id]
  assert len(layout.contact_site_ids) == 2
  # The file had no floor, so one was added, and the layout says so.
  assert layout.ground_added
  assert model.geom("rlmcp_ground").type == mujoco.mjtGeom.mjGEOM_PLANE
  assert layout.base_qpos == 0 and layout.base_qvel == 0


def test_compile_leaves_an_existing_floor_alone(spec, tmp_path):
  with_floor = HOPPER.replace("<worldbody>", '<worldbody><geom type="plane" size="0 0 1"/>')
  path = tmp_path / "floored.xml"
  path.write_text(with_floor)
  model, layout = compile_model(RobotSpec(xml=str(path), contact_sites=("foot",)))
  assert not layout.ground_added
  assert sum(int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE)
             for g in range(model.ngeom)) == 1


def test_a_missing_joint_or_site_is_named(hopper_xml):
  with pytest.raises(KeyError) as excinfo:
    compile_model(RobotSpec(xml=hopper_xml, joints=("hip", "ankle")))
  assert "ankle" in str(excinfo.value)
  with pytest.raises(KeyError) as excinfo:
    compile_model(RobotSpec(xml=hopper_xml, contact_sites=("toe",)))
  assert "toe" in str(excinfo.value)


def test_the_registry_names_every_backend():
  assert set(BACKENDS) == {"mjwarp", "mjbatch", "genesis"}
  reasons = available()
  assert set(reasons) == set(BACKENDS)
  with pytest.raises(KeyError):
    make_backend("brax", RobotSpec(xml="x"), 1, 0.01, 1)


# The contract, on every backend that is here.


def _backends() -> list:
  out = []
  reasons = available()
  cuda = torch.cuda.is_available()
  for name in ("mjbatch", "mjwarp", "genesis"):
    device = "cpu" if name == "mjbatch" else ("cuda" if cuda else "cpu")
    marks = []
    if reasons[name]:
      marks.append(pytest.mark.skip(reason=f"{name}: {reasons[name]}"))
    elif name == "mjwarp" and not cuda:
      marks.append(pytest.mark.skip(reason="mjwarp: no CUDA device here"))
    elif name == "genesis" and os.environ.get("RLMCP_TEST_GENESIS") != "1":
      marks.append(pytest.mark.skip(reason="genesis: set RLMCP_TEST_GENESIS=1 (needs a GPU)"))
    out.append(pytest.param((name, device), id=name, marks=marks))
  return out


@pytest.fixture(params=_backends())
def backend(request, spec):
  name, device = request.param
  sim = make_backend(name, spec, num_envs=3, dt=0.005, decimation=4, device=device)
  yield sim
  sim.close()


def _standing(sim, knee: float = 0.0):
  """Upright on its foot, a little above the floor, knee at ``knee``."""
  n = sim.num_envs
  dev = sim.device
  ids = torch.arange(n, device=dev)
  pos = torch.tensor([[0.0, 0.0, 0.58]] * n, device=dev)
  quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n, device=dev)
  dof = torch.tensor([[0.0, knee]] * n, device=dev)
  return ids, pos, quat, dof


def test_the_robot_is_described_the_same_way_everywhere(backend):
  assert backend.joint_names == ["hip", "knee"]
  assert backend.n_dof == 2
  assert backend.dof_limits.shape == (2, 2)
  assert backend.dof_limits[1].tolist() == pytest.approx([-1.5, 0.0])
  assert backend.contact_names == ["foot", "base"]
  assert backend.control_dt == pytest.approx(0.02)


def test_reset_puts_the_robot_where_it_was_told(backend):
  ids, pos, quat, dof = _standing(backend, knee=-0.1)
  backend.reset(ids, pos, quat, dof)
  assert torch.allclose(backend.root_pos, pos, atol=1e-5)
  assert torch.allclose(backend.root_quat, quat, atol=1e-5)
  assert torch.allclose(backend.dof_pos, dof, atol=1e-5)
  assert torch.allclose(backend.dof_vel, torch.zeros_like(dof), atol=1e-5)
  assert torch.allclose(backend.root_lin_vel, torch.zeros_like(pos), atol=1e-5)


def test_stepping_lands_the_foot_and_reports_the_contact(backend):
  ids, pos, quat, dof = _standing(backend)
  backend.reset(ids, pos, quat, dof)
  for _ in range(60):
    backend.set_dof_targets(dof)
    backend.step()
  # It fell a little onto its foot, and the foot carries the weight.
  assert (backend.root_pos[:, 2] < 0.58).all()
  assert (backend.root_pos[:, 2] > 0.2).all(), "fell over or through the floor"
  forces = backend.contact_forces
  assert forces.shape == (3, 2)
  assert (forces[:, 0] > 5.0).all(), "foot should be loaded"
  assert (forces[:, 1] < 1.0).all(), "base should be off the ground"
  assert torch.isfinite(backend.dof_torque).all()
  assert backend.root_ang_vel.shape == (3, 3)


def test_a_partial_reset_touches_only_its_envs(backend):
  ids, pos, quat, dof = _standing(backend)
  backend.reset(ids, pos, quat, dof)
  for _ in range(20):
    backend.set_dof_targets(dof)
    backend.step()
  settled = backend.root_pos[:, 2].clone()
  backend.reset(ids[:1], pos[:1] + torch.tensor([0.0, 0.0, 0.3], device=backend.device),
                quat[:1], dof[:1])
  after = backend.root_pos[:, 2]
  assert after[0].item() == pytest.approx(0.88, abs=1e-4)
  assert torch.allclose(after[1:], settled[1:], atol=1e-6)


def test_sites_and_pushes_are_available(backend):
  ids, pos, quat, dof = _standing(backend)
  backend.reset(ids, pos, quat, dof)
  feet = backend.contact_site_pos
  assert feet.shape == (3, 2, 3)
  # The foot site hangs 0.55 m under a base at 0.58: just above the floor.
  assert torch.allclose(feet[:, 0, 2], torch.full((3,), 0.03, device=backend.device), atol=0.02)
  backend.push(ids[:1], torch.tensor([[1.0, 0.0, 0.0]], device=backend.device),
               torch.zeros(1, 3, device=backend.device))
  assert backend.root_lin_vel[0, 0].item() == pytest.approx(1.0, abs=1e-4)
  assert backend.root_lin_vel[1, 0].item() == pytest.approx(0.0, abs=1e-4)


def test_friction_and_frames_are_available(backend):
  ids, pos, quat, dof = _standing(backend)
  backend.reset(ids, pos, quat, dof)
  backend.set_friction(ids, torch.full((3,), 0.7, device=backend.device))
  if backend.name != "genesis" and os.environ.get("MUJOCO_GL", "") == "":
    pytest.skip("MuJoCo rendering needs MUJOCO_GL set for this machine")
  frame = backend.render(0, width=64, height=48)
  assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype.name == "uint8"
  if backend.name != "genesis":  # Genesis's camera keeps the size it was built with.
    assert frame.shape == (48, 64, 3)


# The two MuJoCo backends are the same physics, so they must agree.


@pytest.mark.skipif(
    bool(available()["mjbatch"] or available()["mjwarp"]) or not torch.cuda.is_available(),
    reason="needs mjbatch and mjwarp with a GPU")
def test_mjbatch_and_mjwarp_agree_on_the_go1_if_it_is_here(spec):
  xml = os.environ.get("MJLAB_GO1_XML")
  if not xml or not Path(xml).exists():
    pytest.skip("set MJLAB_GO1_XML to mjlab's go1.xml")
  robot = RobotSpec(xml=xml, stiffness={"*_calf_joint": 35.0, "*": 20.0}, damping=0.5,
                    effort_limit={"*_calf_joint": 35.55, "*": 23.7},
                    contact_sites=("FR", "FL", "RR", "RL", "trunk"))
  results = {}
  for name, device in (("mjbatch", "cpu"), ("mjwarp", "cuda")):
    sim = make_backend(name, robot, 2, dt=0.005, decimation=4, device=device,
                       options=SimOptions())
    dev = sim.device
    default = torch.tensor([0.1, 0.9, -1.8] * 4, device=dev).expand(2, -1)
    sim.reset(torch.arange(2, device=dev), torch.tensor([[0.0, 0.0, 0.32]] * 2, device=dev),
              torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, device=dev), default)
    for _ in range(50):
      sim.set_dof_targets(default)
      sim.step()
    results[name] = (sim.root_pos.cpu(), sim.dof_pos.cpu(), sim.contact_forces.cpu())
    sim.close()
  for a, b in zip(results["mjbatch"], results["mjwarp"], strict=True):
    assert torch.allclose(a, b, atol=1e-3), (a, b)
