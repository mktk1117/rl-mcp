"""The parameter providers for manager-based environments.

One provider per family of parameters -- rewards, terminations, events,
commands, actions -- each answering *which objects hold tunable values and what
are they called* for an environment built out of manager term configs. The
machinery that walks, reads and writes them is family-neutral and lives in
:mod:`rlmcp.adapters.access`.

To add a family here: write a provider (see
:mod:`rlmcp.adapters.access.base`) and append it to :data:`PROVIDER_TYPES`.
Discovery, get, set, bounds, the MCP tools and the CLI all pick it up with no
further change.
"""

from __future__ import annotations

from typing import List

from rlmcp.adapters.access import paths
from rlmcp.adapters.access.base import (
    AccessProvider,
    ProviderRegistry,
    Synthetic,
    Term,
    constructor_cfg_reads,
    is_term_instance,
)
from rlmcp.adapters.access.registry import ParameterAccess as _ParameterAccess
from rlmcp.adapters.manager_based.access.actions import ActionAccess
from rlmcp.adapters.manager_based.access.commands import CommandAccess
from rlmcp.adapters.manager_based.access.events import EventAccess
from rlmcp.adapters.manager_based.access.rewards import RewardAccess
from rlmcp.adapters.manager_based.access.terminations import TerminationAccess

PROVIDER_TYPES: List[type] = [
    RewardAccess,
    TerminationAccess,
    EventAccess,
    CommandAccess,
    ActionAccess,
]

__all__ = [
    "AccessProvider",
    "ActionAccess",
    "CommandAccess",
    "EventAccess",
    "ParameterAccess",
    "PROVIDER_TYPES",
    "ProviderRegistry",
    "RewardAccess",
    "Synthetic",
    "TerminationAccess",
    "Term",
    "constructor_cfg_reads",
    "is_term_instance",
    "paths",
]


class ParameterAccess(_ParameterAccess):
  """:class:`~rlmcp.adapters.access.registry.ParameterAccess`, wired to the
  manager-based providers. Constructing it needs nothing but the environment."""

  PROVIDER_TYPES = PROVIDER_TYPES
