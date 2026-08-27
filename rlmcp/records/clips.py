"""The clips a run took of itself, in the order they were taken.

``assets["videos"]`` is a flat list of ``[key, caption]`` pairs with no notion
of time, which is the right shape for attachments and the wrong one for a
*series*. So the iteration is written into the caption when a clip is filed and
read back out here -- :func:`caption_for` writes it, :func:`iteration_of` reads
it -- and nothing downstream parses those strings. The filename
(``progress_env0_it000200.mp4``) is the fallback, so a clip attached by hand, or
one still sitting in a session directory, still sorts into place. A clip whose
iteration cannot be recovered either way is listed last rather than dropped: a
video nobody can order is still a video somebody should be able to watch.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from rlmcp.records.record import RunRecord

_CAPTION = re.compile(r"^iteration\s+(\d+)$", re.IGNORECASE)
_FILENAME = re.compile(r"_it(\d+)(?:\D|$)")

PROGRESS_STEM = "progress"
"""Filename prefix of a scheduled clip -- what tells it apart from an asked-for
one in a directory listing that has both."""


def caption_for(iteration: int) -> str:
  """The caption a progress clip is filed under."""
  return f"iteration {int(iteration)}"


def iteration_of(*candidates: Any) -> Optional[int]:
  """The iteration named by a caption or a filename, first answer wins."""
  for candidate in candidates:
    if candidate is None:
      continue
    text = str(candidate).strip()
    if not text:
      continue
    match = _CAPTION.match(text)
    if match:
      return int(match.group(1))
    match = _FILENAME.search(Path(text).name)
    if match:
      return int(match.group(1))
  return None


def is_progress_clip(key: str) -> bool:
  """Whether this asset key names a clip the run took on its own schedule."""
  return Path(str(key)).name.startswith(f"{PROGRESS_STEM}_")


def from_record(
    record: RunRecord,
    posters: Optional[Dict[str, str]] = None,
    kind: str = "videos",
) -> List[Dict[str, Any]]:
  """One entry per attached clip, ordered by the iteration it is of.

  Plain dicts -- ``key``, ``caption``, ``iteration``, ``poster``,
  ``scheduled`` -- so a viewer, a report or a GUI renders them without learning
  the record's layout. ``posters`` is what
  :func:`rlmcp.records.poster.record_posters` returns.
  """
  posters = posters or {}
  out: List[Dict[str, Any]] = []
  for entry in (record.assets or {}).get(kind) or []:
    if not entry:
      continue
    key = str(entry[0])
    caption = str(entry[1]) if len(entry) > 1 else ""
    out.append({
        "key": key,
        "caption": caption,
        "iteration": iteration_of(caption, key),
        "poster": posters.get(key, ""),
        "scheduled": is_progress_clip(key),
    })
  return order(out)


def order(clips: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
  """Sort clips by iteration, with the un-orderable ones kept at the end."""
  return sorted(
      clips,
      key=lambda c: (c.get("iteration") is None, c.get("iteration") or 0,
                     str(c.get("key") or c.get("name") or "")),
  )
