"""rlmcp: agentic control, introspection and curriculum steering for RL training.

Wrap a task environment once and the training run becomes something an LLM
agent (or a human at a shell) can watch and steer while it runs::

    from mjlab.envs import ManagerBasedRlEnv
    import rlmcp

    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode="rgb_array")
    env = rlmcp.wrap(env, session_dir=log_dir / "rlmcp")
    vec_env = RslRlVecEnvWrapper(env)
    runner = MjlabOnPolicyRunner(vec_env, ...)
    env.attach_runner(runner)   # optional: PPO stats + checkpoints

Heavy dependencies (torch, mujoco, matplotlib, the MCP SDK) are imported lazily
so that ``import rlmcp.session`` works in a bare interpreter -- the MCP server
process deliberately has no simulator installed.
"""

from __future__ import annotations

from typing import Any

# Read from the installed metadata rather than a second literal: the two used
# to drift (0.2.0 here against 0.3.0 in pyproject.toml), and the version a bug
# report quotes should be the one that was installed.
try:
  from importlib.metadata import PackageNotFoundError, version as _dist_version

  __version__ = _dist_version("rl-mcp")
except PackageNotFoundError:  # A source tree that was never installed.
  __version__ = "0.0.0+unknown"

_LAZY = {
    # Session protocol (dependency-free).
    "Session": "rlmcp.session",
    "Request": "rlmcp.session",
    "Response": "rlmcp.session",
    # Core controller and parameter model.
    "RlMcp": "rlmcp.core.controller",
    "ParameterSpec": "rlmcp.core.parameters.spec",
    "ParameterCategory": "rlmcp.core.parameters.spec",
    "ParameterRegistry": "rlmcp.core.parameters.registry",
    "TelemetryBuffer": "rlmcp.core.telemetry.buffer",
    "TraceRecorder": "rlmcp.core.telemetry.trace",
    # Curriculum.
    "CurriculumStage": "rlmcp.core.curriculum",
    "StageSchedule": "rlmcp.core.curriculum",
    "Condition": "rlmcp.core.curriculum",
    "Action": "rlmcp.core.curriculum",
    # Adapters.
    "SimAdapter": "rlmcp.adapters.base",
    "RunnerAdapter": "rlmcp.adapters.base",
    "MjlabSimAdapter": "rlmcp.adapters.mjlab",
    "MjlabRunnerAdapter": "rlmcp.adapters.mjlab",
    "RlMcpEnvWrapper": "rlmcp.adapters.mjlab",
    "wrap": "rlmcp.adapters.mjlab",
    # Server.
    "create_mcp_server": "rlmcp.server.mcp_server",
}

__all__ = sorted(_LAZY) + ["__version__"]


def __getattr__(name: str) -> Any:
  module_path = _LAZY.get(name)
  if module_path is None:
    raise AttributeError(f"module 'rlmcp' has no attribute '{name}'")
  import importlib

  return getattr(importlib.import_module(module_path), name)


def __dir__() -> list[str]:
  return list(__all__)
