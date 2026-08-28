"""What the code looked like when a run launched.

The records fold *config* edits from the root down to a node and call it the
recipe. Code changes are invisible to that, which is why the most important run
in a campaign is often the one the graph cannot explain: three runs re-priced
rewards and failed, the fourth changed the action interface -- a code change --
and went from 1.1 to 16.8 goals per minute.

So a run stamps its package at launch. Two facts, because sometimes you have
committed and often you have not:

* ``head`` is the human anchor -- "9f2c1ab + 14 uncommitted lines" is what a
  reader cites and what somebody else can fetch;
* ``tree`` is the truth. It is a real git tree object written with plumbing, so
  HEAD does not move, no branch appears, and the content survives the commit
  being rebased, amended or dropped.

The tree is kept alive by a ref under ``refs/rlmcp/runs/``: without it the
object is unreachable and the next ``git gc`` throws the snapshot away. Nothing
else about the repository is touched.

Storage is close to free in the common case. A clean tree is *the same object*
git already stored for HEAD, and two runs that changed one file share every
other blob -- so a hundred runs of one task cost about one copy plus the
diffs.

Scope is the package directory: the pathspec is handed to ``git add``, so
whatever the repository ignores -- logs, checkpoints, ``.venv`` -- is ignored
here too. A snapshot that includes a 400 MB checkpoint is not a snapshot.

A directory that is not in a git repository gets ``{"kind": "none"}`` and a
reason, never an exception: a run must not fail to launch because provenance
could not be recorded. (DESIGN.md §8.1 also describes a content-addressed store
for that case; this module is the git half, and the payload shape is what a
second implementation would fill in.)
"""

from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path
from typing import Any

REF_PREFIX = "refs/rlmcp/runs"
"""Where snapshot trees are anchored so ``git gc`` keeps them."""

MAX_PATCH_BYTES = 400_000
"""A patch this big is not being read by anyone; it is being scrolled past."""


class SnapshotError(RuntimeError):
  """A git command that was expected to work did not."""


def _git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
  """Run one git command in ``repo`` and return its stdout."""
  result = subprocess.run(
      ["git", "-C", str(repo), *args],
      capture_output=True, text=True,
      env={**os.environ, **(env or {})},
      check=False,  # The returncode is read below, with the stderr in the message.
  )
  if result.returncode != 0:
    raise SnapshotError(
        f"git {' '.join(args)} failed in {repo}: {result.stderr.strip()}")
  return result.stdout.strip()


def repo_of(path: Path | str) -> Path | None:
  """The git repository ``path`` lives in, or None."""
  directory = Path(path).expanduser().resolve()
  if not directory.is_dir():
    directory = directory.parent
  try:
    return Path(_git(directory, "rev-parse", "--show-toplevel")).resolve()
  except (SnapshotError, FileNotFoundError):
    return None


def head_of(repo: Path) -> dict[str, Any] | None:
  """The commit a reader would cite, or None in a repo with no commits yet."""
  try:
    commit = _git(repo, "rev-parse", "HEAD")
  except SnapshotError:
    return None
  branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
  # Pushed means "somebody else can fetch this", which is what makes the anchor
  # worth citing -- so it is asked of the remotes, not of the branch's name.
  remotes = _git(repo, "branch", "-r", "--contains", commit).split("\n")
  remotes = [r.strip() for r in remotes if r.strip()]
  return {
      "commit": commit,
      "short": commit[:7],
      "branch": None if branch == "HEAD" else branch,
      "pushed": bool(remotes),
      "remote": remotes[0] if remotes else None,
  }


def capture(package: Path | str, record_id: str = "") -> dict[str, Any]:
  """Stamp ``package`` as it is right now. Never raises.

  Returns the ``code`` payload a record stores: ``head``, ``tree``, ``dirty``
  and ``clean``, plus the repository and the path within it that were snapped.
  """
  package = Path(package).expanduser().resolve()
  repo = repo_of(package)
  if repo is None:
    return {"kind": "none", "root": str(package),
            "reason": f"{package} is not inside a git repository"}
  try:
    inside = package.relative_to(repo)
    scope = str(inside) if str(inside) != "." else "."
    tree = _write_tree(repo, scope)
    head = head_of(repo)
    against_head = (diff(repo, f"{head['commit']}^{{tree}}", tree, scope=scope)
                    if head else {"changed": 0, "added": 0, "removed": 0})
    ref = ""
    if record_id:
      ref = f"{REF_PREFIX}/{record_id}"
      _git(repo, "update-ref", ref, tree)
    return {
        "kind": "git",
        "repo": str(repo),
        "root": scope,
        # How much was stamped, so a snapshot that quietly swallowed the whole
        # repository is visible in the record rather than only on disk.
        "files": len([p for p in _git(repo, "ls-tree", "-r", "--name-only",
                                      tree).split("\n") if p]),
        "head": head,
        "tree": tree,
        "clean": against_head["changed"] == 0,
        "dirty": {"files": against_head["changed"],
                  "added": against_head["added"],
                  "removed": against_head["removed"]},
        "ref": ref,
    }
  except (SnapshotError, ValueError, OSError) as exc:
    # Provenance is worth having, never worth a failed launch.
    return {"kind": "none", "root": str(package),
            "reason": f"{type(exc).__name__}: {exc}"}


def _write_tree(repo: Path, scope: str) -> str:
  """Write the current content of ``scope`` as a tree object.

  Through a throwaway index, which is the whole trick: the real index, HEAD and
  every branch are untouched, so stamping a run cannot disturb whatever the
  person at the keyboard is in the middle of.
  """
  index = repo / ".git" / f"rlmcp-index-{uuid.uuid4().hex[:8]}"
  env = {"GIT_INDEX_FILE": str(index)}
  try:
    _git(repo, "add", "-A", "--", scope, env=env)
    return _git(repo, "write-tree", env=env)
  finally:
    index.unlink(missing_ok=True)


def diff(repo: Path | str, before: str, after: str,
         scope: str = "") -> dict[str, Any]:
  """Per-file line counts between two trees.

  Whitespace-only changes are marked ``trivial`` rather than dropped: deciding
  whether an edit changes behaviour is undecidable in general, so the honest
  version records everything and lets a viewer grey the noise. Dropping it
  silently is how a real change eventually disappears.
  """
  repo = Path(repo)
  paths = ["--", scope] if scope and scope != "." else []
  numstat = _git(repo, "diff", "--numstat", before, after, *paths)
  meaningful = {
      line.split("\t")[-1]
      for line in _git(repo, "diff", "-w", "--numstat", before, after,
                       *paths).split("\n") if line.strip()
  }
  files: list[dict[str, Any]] = []
  added = removed = 0
  for line in numstat.split("\n"):
    if not line.strip():
      continue
    plus, minus, path = line.split("\t", 2)
    # "-" is git's marker for a binary file: real change, uncountable lines.
    gained = 0 if plus == "-" else int(plus)
    lost = 0 if minus == "-" else int(minus)
    added += gained
    removed += lost
    files.append({"path": path, "added": gained, "removed": lost,
                  "binary": plus == "-", "trivial": path not in meaningful})
  return {"files": files, "changed": len(files), "added": added,
          "removed": removed}


def patch(repo: Path | str, before: str, after: str, scope: str = "",
          max_bytes: int = MAX_PATCH_BYTES) -> str:
  """The unified diff between two trees, truncated rather than unbounded."""
  paths = ["--", scope] if scope and scope != "." else []
  text = _git(Path(repo), "diff", before, after, *paths)
  if len(text) > max_bytes:
    return text[:max_bytes] + f"\n… truncated at {max_bytes} bytes\n"
  return text


def restore(repo: Path | str, tree: str, destination: Path | str) -> Path:
  """Write a snapshot back out -- the package exactly as that run had it."""
  destination = Path(destination).expanduser().resolve()
  destination.mkdir(parents=True, exist_ok=True)
  archive = subprocess.run(
      ["git", "-C", str(repo), "archive", tree],
      capture_output=True, check=False)
  if archive.returncode != 0:
    raise SnapshotError(f"git archive {tree} failed: {archive.stderr.decode()}")
  subprocess.run(["tar", "-x", "-C", str(destination)],
                 input=archive.stdout, check=True)
  return destination
