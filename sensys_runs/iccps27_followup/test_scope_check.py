"""Regression checks for the narrowly approved C03 scope exception."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sensys_runs.iccps27_followup import scope_check as scope


def test_real_clean_closure_excludes_all_followup_and_c03():
    reachable = scope.clean_training_imports()
    assert "lio/alg/lio_agent_er.py" in reachable
    assert "lio/alg/REFiNE_er.py" in reachable
    assert scope.C03 not in reachable
    assert not any(p.startswith("sensys_runs/iccps27_followup/") for p in reachable)


def test_exception_never_hides_an_unapproved_path(monkeypatch):
    monkeypatch.setattr(scope, "clean_training_imports", lambda root: ["lio/alg/REFiNE_er.py"])
    assert scope.permitted_c03_exception(None, [scope.C03, "lio/alg/REFiNE_er.py"]) == ["lio/alg/REFiNE_er.py"]


def test_exception_fails_if_c03_is_reachable(monkeypatch):
    monkeypatch.setattr(scope, "clean_training_imports", lambda root: [scope.C03])
    with pytest.raises(AssertionError, match="reachable"):
        scope.permitted_c03_exception(None, [scope.C03])
