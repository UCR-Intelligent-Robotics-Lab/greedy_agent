"""Validate the fixed campaign without importing TensorFlow or starting jobs.

Tests compare manifest configs against the existing launcher semantics. Failure
must be fixed in command generation rather than by relaxing matrix invariants.
"""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sensys_runs import energy_profile_campaign as campaign
from sensys_runs import run_batch, run_single
from sensys_runs.energy_profiles import get_profile


@pytest.fixture
def isolated_campaign(monkeypatch, tmp_path):
    """Read real configs but redirect freshness checks into a temporary root."""
    factory = campaign.load_config
    configs = {m: factory(m) for m in ("lio_er", "refine_er")}
    from copy import deepcopy
    monkeypatch.setattr(campaign, "load_config", lambda m: deepcopy(configs[m]))
    monkeypatch.setattr(campaign, "source_identity", lambda: {"test_fixture": True})
    monkeypatch.setattr(campaign, "REPO_ROOT", tmp_path)
    return tmp_path


def test_matrix_and_effective_arguments(isolated_campaign, monkeypatch):
    """Exactly four arms use the existing ten seeds and effective CLI settings."""
    manifest = campaign.build_campaign("test_campaign", "local", 2)
    assert manifest["n_runs"] == 40
    assert Counter((r["method"], r["training_profile"]) for r in manifest["runs"]) == {
        arm: 10 for arm in campaign.ARMS}
    seeds = {run_batch.SEED_BASE + i for i in range(1, 11)}
    for arm in campaign.ARMS:
        assert {r["seed"] for r in manifest["runs"] if (r["method"], r["training_profile"]) == arm} == seeds
    assert not (isolated_campaign / "lio/results").exists()
    for row in manifest["runs"]:
        monkeypatch.setattr(sys, "argv", row["argv"][1:])
        parsed = run_single._parse_args()
        cfg = row["effective_config"]
        assert (parsed.n_agents, parsed.min_at_lever) == (4, 2)
        assert parsed.seed == row["seed"] == cfg.main.seed
        assert parsed.n_episodes == cfg.alg.n_episodes
        assert parsed.period == cfg.alg.period and parsed.n_eval == cfg.alg.n_eval
        assert parsed.cost_profile == row["training_profile"]
        assert cfg.cost_profile == get_profile(parsed.cost_profile).snapshot()
        assert Path(row["expected_checkpoint"]) == Path(row["output_directory"]) / cfg.main.model_name
        assert parsed.threads == 2 and parsed.platform == "local" and not parsed.use_gpu
        if parsed.method == "refine_er":
            assert parsed.fairness_mult == cfg.lio.Fairness_multiplier == 0.0
            assert parsed.energy_weight == cfg.lio.energy_weight
        else:
            assert parsed.fairness_mult is None and parsed.energy_weight is None
    assert manifest["time_budget"]["measured_serial_seconds"] is None


def test_smoke_inherits_duration_but_keeps_campaign_eval(isolated_campaign):
    """Smoke uses four fresh directories and never mixes with the formal panel."""
    short = campaign.build_campaign("panel", "nano", 2, smoke=True)
    full = campaign.build_campaign("panel", "nano", 2)
    assert len(short["runs"]) == 4
    assert not {r["output_directory"] for r in short["runs"]} & {r["output_directory"] for r in full["runs"]}
    for row in short["runs"]:
        cfg = row["effective_config"]
        existing = run_batch.build_matrix(row["method"], "smoke")[0]
        baseline = campaign.load_config(row["method"])
        assert cfg.alg.n_episodes == existing["n_episodes"]
        assert cfg.alg.period == existing["period"]
        assert cfg.alg.n_eval == baseline.alg.n_eval >= cfg.env.n_agents
        assert cfg.alg.n_test == baseline.alg.n_test


@pytest.mark.parametrize("invalid", ["../escape", "a/b", "a\\b", "", "/absolute"])
def test_invalid_campaign_names(isolated_campaign, invalid):
    """Campaign names cannot redirect results outside the intended root."""
    with pytest.raises(campaign.CampaignError):
        campaign.build_campaign(invalid, "local", 2)
    assert not list(isolated_campaign.iterdir())


def test_refusal_validates_last_run_before_any_output(isolated_campaign, monkeypatch, capsys):
    """One stale directory anywhere invalidates the whole matrix before printing."""
    plan = campaign.build_campaign("panel", "local", 2)
    last = Path(plan["runs"][-1]["output_directory"])
    last.mkdir(parents=True)
    sentinel = last / "keep.txt"
    sentinel.write_text("existing output")
    monkeypatch.setattr(sys, "argv", ["campaign", "--campaign", "panel", "--platform", "local", "--threads", "2", "--dry-run"])
    assert campaign.main() == 2
    output = capsys.readouterr()
    assert not output.out and "already exists" in output.err
    assert sentinel.read_text() == "existing output"
    assert not Path(plan["runs"][0]["output_directory"]).exists()


def test_cli_dry_run_and_exclusive_manifest(isolated_campaign, monkeypatch, capsys):
    """Dry run is repeatable and read-only; explicit artifacts never overwrite."""
    base = ["campaign", "--platform", "local", "--threads", "2"]
    monkeypatch.setattr(sys, "argv", base + ["--dry-run"])
    assert campaign.main() == 0
    first = capsys.readouterr().out
    assert campaign.main() == 0
    assert capsys.readouterr().out == first
    assert json.loads(first)["n_runs"] == 40
    assert not list(isolated_campaign.iterdir())
    output = isolated_campaign / "manifest.json"
    monkeypatch.setattr(sys, "argv", base + ["--out", str(output)])
    assert campaign.main() == 0
    snapshot = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        campaign.main()
    assert error.value.code == 2 and output.read_bytes() == snapshot
    monkeypatch.setattr(sys, "argv", base + ["--dry-run", "--out", str(output)])
    with pytest.raises(SystemExit):
        campaign.main()


def test_cross_arm_drift_rejected(isolated_campaign, monkeypatch):
    """A divergent task/network/schedule must not silently enter the comparison."""
    original = campaign.load_config
    def changed(method):
        cfg = original(method)
        if method == "refine_er":
            cfg.lio.gamma /= 2
        return cfg
    monkeypatch.setattr(campaign, "load_config", changed)
    with pytest.raises(campaign.CampaignError, match="mismatch"):
        campaign.build_campaign("panel", "local", 2)


def test_fresh_process_import_needs_no_tensorflow():
    """Config factories must not execute lio.alg's eager TensorFlow imports."""
    script = """
import sys
from sensys_runs.energy_profile_campaign import load_config
assert load_config('lio_er').alg.n_test > 0
assert load_config('refine_er').lio.energy_weight > 0
assert 'lio.alg' not in sys.modules
assert not any(n == 'tensorflow' or n.startswith('tensorflow.') for n in sys.modules)
print('CONFIG_IMPORT_OK')
"""
    result = subprocess.run([sys.executable, "-B", "-c", script], cwd=ROOT,
                             text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "CONFIG_IMPORT_OK"
