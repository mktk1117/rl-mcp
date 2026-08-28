"""Parameter access, written against no family in particular.

Three pieces, none of which knows what a simulator or a manager is:

* :mod:`.paths` -- the reflective core. Walk a config object for tunable
  leaves, resolve a dotted path to the object holding one, coerce a write.
* :mod:`.base` -- what a provider supplies, and the registry that holds them.
* :mod:`.registry` -- :class:`~rlmcp.adapters.access.registry.ParameterAccess`,
  which turns a list of providers into the discover/get/set surface the
  adapters expose.

Config objects are ordinary dataclasses and dicts in every family rlmcp
targets, so the walk is the same walk. What differs is only *which* objects
hold the tunables, and that is what a provider answers.
"""

from rlmcp.adapters.access import paths
from rlmcp.adapters.access.base import (
    AccessProvider,
    ProviderRegistry,
    Synthetic,
    Term,
    constructor_cfg_reads,
    is_term_instance,
)
from rlmcp.adapters.access.registry import ParameterAccess

__all__ = [
    "AccessProvider",
    "ParameterAccess",
    "ProviderRegistry",
    "Synthetic",
    "Term",
    "constructor_cfg_reads",
    "is_term_instance",
    "paths",
]
