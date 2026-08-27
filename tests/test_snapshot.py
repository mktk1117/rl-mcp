"""Stamping the code a run launched with, without disturbing the repository."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from rlmcp.records import snapshot


def _run(repo: Path, *args: str) -> str:
  out = subprocess.run(["git", "-C", str(repo), *args],
                       capture_output=True, text=True, check=True)
  return out.stdout.strip()


@pytest.fixture
def repo(tmp_path) -> Path:
  """A repository with a task package in it, one commit deep."""
  root = tmp_path / "tasks"
  (root / "shand").mkdir(parents=True)
  (root / "shand" / "task.py").write_text("weight = 1.0\n")
  (root / "logs").mkdir()
  (root / "logs" / "huge.pt").write_text("pretend checkpoint")
  (root / ".gitignore").write_text("logs/\n")
  _run(root.parent, "init", "-q", str(root))
  _run(root, "config", "user.email", "test@example.com")
  _run(root, "config", "user.name", "Test")
  _run(root, "add", "-A")
  _run(root, "commit", "-qm", "the task")
  return root


def test_a_clean_package_is_stamped_with_its_commit(repo):
  code = snapshot.capture(repo / "shand", record_id="011")

  assert code["kind"] == "git"
  assert code["clean"] is True
  assert code["dirty"] == {"files": 0, "added": 0, "removed": 0}
  assert code["head"]["commit"] == _run(repo, "rev-parse", "HEAD")
  assert code["head"]["pushed"] is False        # nothing to fetch it from
  assert code["root"] == "shand"


def test_uncommitted_work_is_recorded_rather_than_lost(repo):
  """The common case: you have edited, you have not committed, and the run is
  the thing that matters. `head` stays the anchor; `tree` is what ran."""
  (repo / "shand" / "task.py").write_text("weight = 1.0\nfilter = 'ema'\n")

  code = snapshot.capture(repo / "shand", record_id="012")

  assert code["clean"] is False
  assert code["dirty"] == {"files": 1, "added": 1, "removed": 0}
  assert code["head"]["commit"] == _run(repo, "rev-parse", "HEAD")
  assert code["tree"] != _run(repo, "rev-parse", "HEAD^{tree}")


def test_stamping_disturbs_nothing_the_person_at_the_keyboard_is_doing(repo):
  """No branch, no moved HEAD, no staged file -- the whole reason this uses a
  throwaway index instead of committing."""
  (repo / "shand" / "task.py").write_text("weight = 2.0\n")
  before = (_run(repo, "rev-parse", "HEAD"), _run(repo, "status", "--porcelain"),
            _run(repo, "branch", "--format=%(refname)"))

  snapshot.capture(repo / "shand", record_id="013")

  after = (_run(repo, "rev-parse", "HEAD"), _run(repo, "status", "--porcelain"),
           _run(repo, "branch", "--format=%(refname)"))
  assert before == after
  assert not list((repo / ".git").glob("rlmcp-index-*"))   # and no litter


def test_the_snapshot_survives_garbage_collection(repo):
  """A tree nothing points at is unreachable; the ref is what makes this safe
  rather than clever."""
  (repo / "shand" / "task.py").write_text("weight = 3.0\n")
  code = snapshot.capture(repo / "shand", record_id="014")

  _run(repo, "gc", "--prune=now", "--quiet")

  assert _run(repo, "rev-parse", code["ref"]) == code["tree"]
  assert _run(repo, "cat-file", "-t", code["tree"]) == "tree"


def test_what_the_repository_ignores_is_not_in_the_snapshot(repo):
  """A snapshot with a checkpoint in it is not a snapshot."""
  code = snapshot.capture(repo / "shand")
  listing = _run(repo, "ls-tree", "-r", "--name-only", code["tree"])

  assert listing == "shand/task.py"


def test_two_runs_are_comparable_whether_or_not_anything_was_committed(repo):
  first = snapshot.capture(repo / "shand", record_id="015")
  (repo / "shand" / "task.py").write_text("weight = 1.0\nfilter = 'ema'\n")
  (repo / "shand" / "extra.py").write_text("# new\n")
  second = snapshot.capture(repo / "shand", record_id="016")

  changed = snapshot.diff(repo, first["tree"], second["tree"])

  assert changed["changed"] == 2 and changed["added"] == 2
  assert sorted(f["path"] for f in changed["files"]) == [
      "shand/extra.py", "shand/task.py"]
  assert "filter = 'ema'" in snapshot.patch(repo, first["tree"], second["tree"])


def test_a_whitespace_only_change_is_marked_not_dropped(repo):
  """Deciding whether an edit changes behaviour is undecidable, so the honest
  version records everything and greys the noise."""
  first = snapshot.capture(repo / "shand")
  (repo / "shand" / "task.py").write_text("weight   =   1.0\n")
  second = snapshot.capture(repo / "shand")

  files = snapshot.diff(repo, first["tree"], second["tree"])["files"]

  assert [f["path"] for f in files] == ["shand/task.py"]
  assert files[0]["trivial"] is True


def test_a_directory_outside_git_says_so_instead_of_failing(tmp_path):
  """Provenance is worth having; it is never worth a failed launch."""
  loose = tmp_path / "loose"
  loose.mkdir()

  code = snapshot.capture(loose, record_id="017")

  assert code["kind"] == "none"
  assert "not inside a git repository" in code["reason"]


def test_a_snapshot_can_be_written_back_out(repo, tmp_path):
  (repo / "shand" / "task.py").write_text("weight = 4.0\n")
  code = snapshot.capture(repo / "shand", record_id="018")

  snapshot.restore(repo, code["tree"], tmp_path / "restored")

  assert (tmp_path / "restored" / "shand" / "task.py").read_text() == "weight = 4.0\n"
