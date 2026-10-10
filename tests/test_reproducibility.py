"""P3a launcher checks; fix seed ordering rather than relaxing byte equality.

TF smoke runs use copied source trees and fresh pytest temporary result roots.
REFiNE checkpoints are deliberately outside the contract documented in P3a.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sensys_runs import run_batch, run_single


def _main_body() -> list[ast.stmt]:
    source = (ROOT / "sensys_runs/run_single.py").read_text(encoding="utf-8")
    return next(node.body for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == "main")


def _onednn_statement() -> ast.stmt:
    return next(node for node in _main_body()
                if ast.unparse(node) ==
                "os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')")


def test_onednn_default_precedes_tensorflow_import():
    """oneDNN must be configured before TensorFlow can read its environment."""
    body = _main_body()
    tf_import = next(node for node in body if isinstance(node, ast.Import)
                     and any(alias.name == "tensorflow" for alias in node.names))
    assert _onednn_statement().lineno < tf_import.lineno


def test_graph_seed_follows_trainer_import_before_training():
    """Re-seed the graph created by trainer import before configuring or training."""
    body = _main_body()
    index = next(i for i, node in enumerate(body)
                 if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "train_mod"
                         for t in node.targets))
    assert ast.unparse(body[index + 1]) == "tf.compat.v1.set_random_seed(args.seed)"
    train_call = next(node for parent in body for node in ast.walk(parent)
                      if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Name) and node.func.id == "train")
    assert body[index + 1].lineno < train_call.lineno


@pytest.mark.parametrize("explicit", [None, "1"])
def test_onednn_explicit_environment_wins(monkeypatch, explicit):
    """Default to zero without silently overriding an explicit oneDNN choice."""
    monkeypatch.delenv("TF_ENABLE_ONEDNN_OPTS", raising=False)
    if explicit is not None:
        monkeypatch.setenv("TF_ENABLE_ONEDNN_OPTS", explicit)
    module = ast.Module(body=[_onednn_statement()], type_ignores=[])
    exec(compile(module, "run_single.py", "exec"), {"os": os})
    assert os.environ["TF_ENABLE_ONEDNN_OPTS"] == (explicit or "0")


def test_provenance_reads_actual_graph_seed(monkeypatch, tmp_path):
    """Provenance must read TF's graph seed, not echo an unchecked CLI seed."""
    graph = SimpleNamespace(seed=17)
    tf = SimpleNamespace(compat=SimpleNamespace(v1=SimpleNamespace(
        get_default_graph=lambda: graph)))
    monkeypatch.setitem(sys.modules, "tensorflow", tf)
    monkeypatch.setattr(run_single, "_git_field", lambda command: None)
    monkeypatch.setattr(run_single, "_git_dirty", lambda: False)
    monkeypatch.setenv("TF_ENABLE_ONEDNN_OPTS", "1")
    args = SimpleNamespace(platform="test", method="lio_er", exp_name="test",
                           dir_name="test", seed=23, threads=1, cost_profile="legacy")
    cfg = SimpleNamespace(alg=SimpleNamespace(n_episodes=1, period=1, n_eval=1),
                          main=SimpleNamespace(use_gpu=False),
                          env=SimpleNamespace(n_agents=3, min_at_lever=1))
    run_single._write_provenance(str(tmp_path), args, cfg, "stub", "stub")
    record = json.loads((tmp_path / "provenance.json").read_text())
    assert record["tf_graph_seed"] == graph.seed != args.seed
    assert record["onednn_opts"] == "1"


@pytest.fixture(scope="module")
def smoke_runs(tmp_path_factory):
    """Run each requested smoke case once, isolated from repository results."""
    pytest.importorskip("tensorflow")
    parent = tmp_path_factory.mktemp("p3a_smoke")
    outputs = {}

    def launch(method: str, label: str, seed_index: int = 1) -> Path:
        key = (method, label, seed_index)
        if key in outputs:
            return outputs[key]
        tree = parent / (method + "_" + label)
        assert not tree.exists()
        for directory in ("lio", "sensys_runs"):
            shutil.copytree(ROOT / directory, tree / directory,
                            ignore=shutil.ignore_patterns(
                                "results", "logs", "run_logs", "__pycache__", "*.pyc"))
        spec = dict(run_batch.build_matrix(method, "smoke")[0])
        spec["seed"] = run_batch._seed(seed_index)
        argv = run_batch.spec_to_argv(spec, use_gpu=False, threads=1, platform="test")
        argv[:2] = [sys.executable, str(tree / "sensys_runs/run_single.py")]
        result = tree / "lio/results" / spec["exp_name"] / spec["dir_name"]
        assert not result.exists()
        env = os.environ.copy()
        env.pop("TF_ENABLE_ONEDNN_OPTS", None)  # Exercise the launcher's default.
        env.pop("TF_DETERMINISTIC_OPS", None)
        env.update(PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1",
                   PYTHONPATH=str(tree))
        completed = subprocess.run(argv, cwd=tree, env=env, capture_output=True,
                                   text=True, timeout=600)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        record = json.loads((result / "provenance.json").read_text())
        assert record["tf_graph_seed"] == spec["seed"]
        assert record["onednn_opts"] == "0"
        assert record["threads"] == 1
        lines = (result / "log.csv").read_text().splitlines()
        assert len(lines) == run_batch.expected_data_rows(spec) + 1
        outputs[key] = result
        return result

    return launch


def _assert_identical(left: Path, right: Path, filenames: list[str]) -> None:
    for name in filenames:
        assert (left / name).read_bytes() == (right / name).read_bytes(), name


def test_lio_smoke_byte_identity(smoke_runs):
    """Same-seed LIO must preserve CSV, config, and every checkpoint shard byte."""
    left, right = smoke_runs("lio_er", "a"), smoke_runs("lio_er", "b")
    config = json.loads((left / "config.json").read_text())
    model = config["main"]["model_name"]
    shards = sorted(p.name for p in left.glob(model + ".data-*"))
    assert shards and shards == sorted(p.name for p in right.glob(model + ".data-*"))
    _assert_identical(left, right, ["log.csv", "config.json", model + ".index"] + shards)


def test_refine_smoke_log_and_config_identity(smoke_runs):
    """Same-seed REFiNE CSV/config are exact; finding 5 excludes checkpoint bytes."""
    left, right = smoke_runs("refine_er", "a"), smoke_runs("refine_er", "b")
    _assert_identical(left, right, ["log.csv", "config.json"])


def test_lio_different_seeds_change_log(smoke_runs):
    """Reject a vacuous repeatability pass from seed-insensitive smoke outputs."""
    left = smoke_runs("lio_er", "a")
    right = smoke_runs("lio_er", "different_seed", seed_index=2)
    assert (left / "log.csv").read_bytes() != (right / "log.csv").read_bytes()
