"""What every legged-gym-shaped environment has in common.

Genesis's locomotion examples, legged_gym itself and the forks of both describe
a task the same way: the tunables are plain dicts on the environment instance
(``env_cfg``, ``reward_cfg``, ``command_cfg``), ``reward_scales`` maps a term
name to a number, and ``reward_functions`` binds ``_reward_<name>`` for each of
those keys once, at construction. That shared shape -- not any one simulator --
is what the parameter access, trace sampling and summary metrics here are
written against, which is why they live in this package rather than inside a
backend.

It is the sibling of :mod:`rlmcp.adapters.manager_based`, and the two answer the
same questions about environments built two different ways. Both hand their
answers to the same :class:`~rlmcp.adapters.env_wrapper.RlMcpEnvWrapper`, which
is what keeps ``wrap()`` the same call and the commands the same commands.

A backend on top of this is then only what is genuinely its own: how a robot is
found, and how a frame is rendered.
"""

from rlmcp.adapters.legged_gym_style.access import ParameterAccess
from rlmcp.adapters.legged_gym_style.metrics import episode_log, summary_metrics
from rlmcp.adapters.legged_gym_style.sampling import StateSampler
from rlmcp.adapters.legged_gym_style.spec import (
    FlatEnvSpec,
    NotAFlatEnv,
    detect,
)

__all__ = [
    "FlatEnvSpec",
    "NotAFlatEnv",
    "ParameterAccess",
    "StateSampler",
    "detect",
    "episode_log",
    "summary_metrics",
]
