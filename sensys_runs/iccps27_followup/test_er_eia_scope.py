"""Prove the additive arm cannot enter pre-existing dispatch/import paths."""
import subprocess
from pathlib import Path
from unittest.mock import patch

from sensys_runs.iccps27_followup import scope_check

REPO = Path(__file__).resolve().parents[2]
APPROVED = "4ce4341d867cba0ceb4836e4e288a3c19812a0ce"
PREFIX = "sensys_runs/iccps27_followup/"


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).splitlines()


def existing_paths():
    return set(git("ls-tree", "-r", "--name-only", APPROVED))


def new_paths():
    tracked = set(git("diff", "--name-only", APPROVED))
    untracked = set(git("ls-files", "--others", "--exclude-standard", "--", PREFIX))
    return tracked | untracked


def test_additive_scope_has_no_existing_source_change():
    changed = new_paths()
    assert changed
    assert all(path.startswith(PREFIX) for path in changed)
    assert not changed.intersection(existing_paths())
    assert not git("diff", "--name-only", APPROVED, "--", "lio", "tests", "docs")
    for path in changed:
        raw = (REPO/path).read_bytes()
        assert b"\r" not in raw
        assert "\u2014" not in raw.decode("utf-8")


def test_no_existing_clean_or_eia_entry_point_imports_the_new_arm():
    roots = (*scope_check.CLEAN_ROOTS,
             PREFIX+"a9_campaign.py", PREFIX+"eia_adapter.py",
             PREFIX+"eia_campaign.py", PREFIX+"eia_runner.py",
             PREFIX+"reset_probe.py", PREFIX+"analyze_eia.py",
             "lio/alg/train_lio_eia_er.py", "lio/alg/train_REFiNE_eia_er.py")
    with patch.object(scope_check, "CLEAN_ROOTS", roots):
        closure = set(scope_check.clean_training_imports())
    assert not closure.intersection(new_paths())
    assert "lio/alg/lio_eia_er.py" in closure
    assert "lio/alg/REFiNE_eia_er.py" in closure
    assert "sensys_runs/energy_reward_runtime.py" in closure
