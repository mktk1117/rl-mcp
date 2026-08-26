"""A still frame to stand in for a clip.

A node on the tree, a tile in the story, a thumbnail under a parameter trace --
all of them want to show what a run *looked like* without loading ten videos at
once. One frame per clip does that, and a frame is cheap enough to inline into
a node's background where a video never could be.

The frame is taken from a third of the way in rather than from the start: the
first frames of a rollout clip are the instant after a reset, which is the one
moment every run looks identical.

Where the still comes from is a decision this module owns, and it is worth
saying which of the plausible answers was taken. A record carries
``assets: {"videos": [[key, caption]], "plots": [[key, caption]]}`` and nothing
that names a poster, so a poster has to be *derived*. It is derived from the
video, not borrowed from a plot: a plot is a chart of the run, and putting one
in the slot where the reader expects to see the policy moving would answer a
different question than the one being asked. The still is cached in the media
store beside the clip it came from, under a key that is a pure function of the
video's, so a second render is a stat rather than a decode -- and so a store
whose media directory is read-only simply renders without thumbnails instead of
failing.
"""

from __future__ import annotations

import hashlib
import re
import tempfile
from pathlib import Path
from typing import Dict, Optional, Sequence

from rlmcp.records.record import RunRecord

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _component(value: str) -> str:
  """One path segment, with anything that could climb out of it removed."""
  cleaned = _SAFE.sub("_", str(value or "")).strip("._-")
  return cleaned or "unnamed"


def poster_key(record_id: str, video_key: str) -> str:
  """The media key a still for ``video_key`` is cached under.

  A pure function of the two inputs, which is what makes the cache work without
  anything being written down: the renderer asks for the key it would have
  used, and a file already there is a poster already made. The digest is in the
  name because two clips in one record can share a basename across directories,
  and a thumbnail silently standing for the wrong run is worse than no
  thumbnail.
  """
  digest = hashlib.sha1(video_key.encode("utf-8")).hexdigest()[:8]
  stem = _component(Path(video_key).stem)
  return f"{_component(record_id)}/posters/{digest}-{stem}.png"


def extract_poster(video: str | Path, out: str | Path,
                   at_fraction: float = 0.34, max_width: int = 480) -> Optional[Path]:
  """Write one frame of ``video`` to ``out``. ``None`` if it cannot be read.

  Returning None rather than raising is deliberate: a poster is a nicety, and a
  codec this machine cannot open must not be able to fail a render.
  """
  try:
    import imageio.v2 as imageio
    import numpy as np
  except ImportError:
    return None

  try:
    reader = imageio.get_reader(str(video))
  except Exception:  # noqa: BLE001 -- any unreadable file, for any reason.
    return None

  try:
    try:
      count = reader.count_frames()
    except Exception:  # noqa: BLE001 -- some formats will not say.
      count = 0
    index = int(count * at_fraction) if count else 0
    try:
      frame = reader.get_data(index)
    except (IndexError, ValueError):
      frame = reader.get_data(0)

    array = np.asarray(frame)
    if array.ndim == 3 and array.shape[2] == 4:
      array = array[:, :, :3]
    width = array.shape[1]
    if width > max_width:
      # Nearest-neighbour by slicing: a thumbnail does not justify a dependency
      # on a resampler, and the aliasing is invisible at this size.
      step = max(1, width // max_width)
      array = array[::step, ::step]

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(out), array)
    return out
  except Exception:  # noqa: BLE001 -- see the docstring.
    return None
  finally:
    reader.close()


def _under(root: Path, key: str) -> Optional[Path]:
  """``root/key``, or None when the key is not a key at all.

  The check is on the key's *parts* rather than on the resolved path, and the
  difference matters: a media directory whose subdirectories are symlinks into
  wherever the artifacts actually live is a normal way to run this, and
  resolving first would call every one of those keys an escape attempt. An
  absolute key or one containing ``..`` is refused, which is the whole of what
  a record-supplied string can do to climb out.
  """
  path = Path(key)
  if not key or path.is_absolute() or any(part == ".." for part in path.parts):
    return None
  return root / path


def ensure_posters(records: Sequence[RunRecord], media_root: Path | str,
                   derive: bool = True) -> Dict[str, str]:
  """Stills for the clip that stands for each run, keyed by the clip's key.

  One per record rather than one per video: the tree, the story and the strip
  all show a run's *headline* clip, and decoding every attachment to fill a
  cache nothing reads would make rendering a large records store slow for no
  visible gain.

  ``derive=False`` looks for stills already cached and makes none, which is the
  honest thing to do against a store somebody else is writing to.

  Anything that fails -- no imageio, an unreadable codec, a read-only media
  directory -- is left out of the map. The viewer's no-poster path is the
  common case anyway: most records carry no video at all.
  """
  from rlmcp.records.lineage import headline_video

  root = Path(media_root)
  out: Dict[str, str] = {}
  for record in records:
    entry = headline_video(record)
    if not entry:
      continue
    video_key = entry[0]
    # A record that already names its own still -- a store that extracts one
    # at attach time, or an import that carried one across -- is believed.
    if len(entry) > 2 and entry[2]:
      out[video_key] = entry[2]
      continue
    key = poster_key(record.id, video_key)
    target = _under(root, key)
    if target is None:
      continue
    if target.exists():
      out[video_key] = key
      continue
    if not derive:
      continue
    source = _under(root, video_key)
    if source is None or not source.exists():
      continue
    # Written to a scratch file and moved into place, so a render interrupted
    # half a frame in cannot leave a truncated PNG that every later render
    # then treats as a cache hit.
    with tempfile.TemporaryDirectory() as tmp:
      still = extract_poster(source, Path(tmp) / "poster.png")
      if still is None:
        continue
      try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(still.read_bytes())
      except OSError:
        continue
    out[video_key] = key
  return out
