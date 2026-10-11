"""P7 import identity and real-launch cost accounting; synthetic smoke data only.

Exercise the actual launcher in subprocesses, not package-constructed probe agents.
A missing TensorFlow stack is an explicit skip, never an integration pass.
"""
from __future__ import annotations

import ast
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sensys_runs import run_batch
from sensys_runs.energy_profile_campaign import TASK
from sensys_runs.energy_profiles import get_hardware_profile, get_profile

ALIASES = {"lio_agent_er": "lio.alg.lio_agent_er", "REFiNE_er": "lio.alg.REFiNE_er"}


def _launcher_nodes():
    source = (ROOT / "sensys_runs/run_single.py").read_text(encoding="utf-8")
    body = next(n.body for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    trainer = next(i for i, n in enumerate(body)
                   if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "train_mod" for t in n.targets))
    return body, trainer


def test_alias_block_precedes_trainer_import():
    """Keep the validated four statements immediately before trainer import."""
    body, trainer = _launcher_nodes()
    assert [ast.unparse(n) for n in body[trainer - 4:trainer]] == [
        "import lio.alg.lio_agent_er",
        "import lio.alg.REFiNE_er",
        "sys.modules['lio_agent_er'] = lio.alg.lio_agent_er",
        "sys.modules['REFiNE_er'] = lio.alg.REFiNE_er",
    ]


def test_runtime_module_identity():
    """Execute the launcher's real path and alias statements in an isolated process."""
    pytest.importorskip("tensorflow")
    body, trainer = _launcher_nodes()
    path_setup = next(n for n in body if isinstance(n, ast.For)
                      and ast.unparse(n.target) == "path")
    source = ast.unparse(ast.Module(body=[path_setup, *body[trainer - 4:trainer]],
                                   type_ignores=[]))
    script = ("import sys\n"
              f"REPO_ROOT = {str(ROOT)!r}\n"
              f"LIO_ALG = {str(ROOT / 'lio/alg')!r}\n" + source + "\n"
              + "\n".join(f"assert sys.modules[{bare!r}] is sys.modules[{full!r}]"
                          for bare, full in ALIASES.items()))
    result = subprocess.run([sys.executable, "-B", "-c", script], cwd=ROOT,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope="module")
def cost_smokes(tmp_path_factory):
    """Run each real trainer once; reconcile import identity if costs disagree."""
    pytest.importorskip("tensorflow")
    parent = tmp_path_factory.mktemp("p7_import_identity")
    results = {}
    for method in ("lio_er", "refine_er"):
        tree = parent / method
        for directory in ("lio", "sensys_runs"):
            shutil.copytree(ROOT / directory, tree / directory,
                            ignore=shutil.ignore_patterns(
                                "results", "logs", "run_logs", "__pycache__", "*.pyc"))
        spec = dict(run_batch.build_matrix(method, "smoke")[0])
        spec.update(n_agents=TASK[0], min_at_lever=TASK[1], n_eval=TASK[0])
        argv = run_batch.spec_to_argv(spec, use_gpu=False, threads=1, platform="test")
        argv[:2] = [sys.executable, str(tree / "sensys_runs/run_single.py")]
        argv += ["--cost_profile", "paper_ratio_v1",
                 "--hardware_profile", "hetero_2x1p5_v1"]
        result_path = tree / "lio/results" / spec["exp_name"] / spec["dir_name"]
        assert not result_path.exists()
        env = dict(os.environ, PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1",
                   PYTHONPATH=str(tree), TF_ENABLE_ONEDNN_OPTS="0")
        completed = subprocess.run(argv, cwd=tree, env=env, capture_output=True,
                                   text=True, timeout=600)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        with (result_path / "log.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == run_batch.expected_data_rows(spec) > 0
        provenance = json.loads((result_path / "provenance.json").read_text())
        assert provenance["agent_module_aliases"] == ALIASES
        assert provenance["agent_cost_patch_target"] == (
            "lio.alg.lio_agent_er" if method == "lio_er" else "lio.alg.REFiNE_er")
        results[method] = (spec, rows)
    return results


def _registered_energy(spec, row, index):
    lever, move, door = get_profile("paper_ratio_v1").costs
    assert move == door
    vector = get_hardware_profile("hetero_2x1p5_v1").vector(spec["n_agents"])
    steps = spec["n_eval"] * float(row["steps_per_eps"])
    return vector[index] * (move * steps + (lever - move) * float(row[f"A{index + 1}_n_lever"]))


@pytest.mark.parametrize("method", ["lio_er", "refine_er"])
def test_real_launcher_registered_costs(cost_smokes, method):
    """Every logged agent/evaluation must use the registered lambda-weighted table."""
    spec, rows = cost_smokes[method]
    for row in rows:
        for i in range(spec["n_agents"]):
            actual = float(row[f"A{i + 1}_total_energy"])
            assert actual > 0
            assert actual == pytest.approx(_registered_energy(spec, row, i), abs=1e-6, rel=0)
            for legacy_unit in (14.53, 10.09, 3.935):
                multiple = round(actual / legacy_unit)
                assert not (multiple > 0 and abs(actual - multiple * legacy_unit) <= 1e-6)


def test_lio_legacy_table_is_rejected(cost_smokes):
    """Require a nonzero probe that separates registered and legacy accounting."""
    spec, rows = cost_smokes["lio_er"]
    differences = []
    for row in rows:
        steps = spec["n_eval"] * float(row["steps_per_eps"])
        for i in range(spec["n_agents"]):
            prefix = f"A{i + 1}_"
            lever, door = float(row[prefix + "n_lever"]), float(row[prefix + "n_door"])
            legacy = 14.53 * lever + 3.935 * door + 10.09 * (steps - lever - door)
            differences.append(abs(float(row[prefix + "total_energy"]) - legacy))
    assert differences and any(d > 1e-6 for d in differences)
