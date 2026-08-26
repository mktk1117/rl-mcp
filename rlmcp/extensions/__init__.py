"""Where capabilities live.

The core defines what an extension *is* (:mod:`rlmcp.core.extensions`). This
package is where implementations are kept and how they are found.

Adding one is three steps:

1. Drop a module in this package (or in your own project).
2. Subclass :class:`~rlmcp.core.extensions.Extension` and decorate it with
   :func:`register`.
3. Import the module from this package's ``__init__`` (a module in your own
   project declares the entry point below instead) -- importing is what runs
   the decorator. ``rlmcp.wrap`` then discovers it, its commands appear in
   ``rlmcp commands``, MCP's ``run_command`` can call them, and a curriculum
   stage can name them in ``apply``.

```python
from rlmcp.core.extensions import Extension
from rlmcp.extensions import register

@register
class ObjectSet(Extension):
  \"\"\"Which objects appear in a manipulation scene.\"\"\"

  name = "objects"

  def available(self) -> bool:
    return hasattr(self.env, "object_pool")

  def commands(self):
    return {"set_objects": self.cmd_set_objects}

  def cmd_set_objects(self, names: list[str]) -> dict:
    \"\"\"Choose which objects the scene spawns.\"\"\"
    self.env.object_pool = list(names)
    return {"objects": self.env.object_pool}
```

An extension that lives outside rlmcp is registered the same way, and found
without anyone editing this package, by declaring an entry point::

    [project.entry-points."rlmcp.extensions"]
    my_capability = "my_project.rlmcp_ext"

Every extension decides for itself whether an environment supports it, so a
capability that does not apply is simply absent rather than broken.
"""

from __future__ import annotations

from rlmcp.core.extensions import Extension, ExtensionRegistry
from rlmcp.extensions.registry import (
    ENTRY_POINT_GROUP,
    catalog,
    discover,
    load_plugins,
    register,
    registered,
    unregister,
)

# Importing a built-in module is what runs its @register decorator.
from rlmcp.extensions import terrain as _terrain  # noqa: F401  (self-registering)

__all__ = [
    "ENTRY_POINT_GROUP",
    "Extension",
    "ExtensionRegistry",
    "catalog",
    "discover",
    "load_plugins",
    "register",
    "registered",
    "unregister",
]
