"""Manifest execution contracts with real fake children, without TensorFlow.

Git identity/status are injected: tests never create commits. On Windows only,
wait4 is emulated around real Popen children; Linux uses its native wait4.
"""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sensys_runs import energy_profile_runner as runner
from sensys_runs import energy_profile_campaign as campaign


@pytest.fixture
def case(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    stub = repo / "fake_run_single.py"
    stub.write_text('''import json, os, pathlib, sys
a = sys.argv[1:]
def value(k): return a[a.index(k)+1]
target = pathlib.Path("lio/results") / value("--exp_name") / value("--dir_name")
target.mkdir(parents=True)
with open(os.environ["FAKE_EVENTS"], "a") as f:
    f.write(json.dumps({"argv": sys.argv, "hash": os.environ.get("PYTHONHASHSEED"),
                       "onednn": os.environ.get("TF_ENABLE_ONEDNN_OPTS")}) + "\\n")
sys.exit(int(value("--fake-rc")))
''', encoding="utf-8")
    monkeypatch.setattr(runner, "REPO_ROOT", repo)
    monkeypatch.setattr(campaign, "REPO_ROOT", repo)
    # Config factories are read from the actual checkout, never from the fake repo.
    original_root = Path(campaign.__file__).resolve().parents[1]
    original_loader = campaign.load_config
    def load_config(method):
        monkeypatch.setattr(campaign, "REPO_ROOT", original_root)
        try:
            return original_loader(method)
        finally:
            monkeypatch.setattr(campaign, "REPO_ROOT", repo)
    monkeypatch.setattr(campaign, "load_config", load_config)
    monkeypatch.setattr(campaign, "source_identity", lambda: {"git_commit": "fixture-head"})
    manifest = campaign.build_campaign("runner_test", "local", 1)
    for row in manifest["runs"]:
        row["argv"][:2] = [sys.executable, str(stub)]
        row["argv"] += ["--fake-rc", "0", "--literal", "a b;$not-shell"]
    state = {"status": "", "ignored": True}
    def git(*args):
        if args == ("rev-parse", "HEAD"):
            return "fixture-head"
        if args == ("status", "--porcelain"):
            return state["status"]
        assert args[:2] == ("check-ignore", "--quiet")
        if not state["ignored"]:
            raise runner.RunnerError("results root is not git-ignored")
        return ""
    monkeypatch.setattr(runner, "git", git)
    notices = []
    monkeypatch.setattr(runner, "notify", lambda message, **kw: notices.append((message, kw)))
    monkeypatch.setenv("NTFY_TOPIC", "fake-tests-never-send")
    monkeypatch.setenv("TF_ENABLE_ONEDNN_OPTS", "1")
    events = tmp_path / "events.jsonl"
    monkeypatch.setenv("FAKE_EVENTS", str(events))
    if not runner.WAIT4_SUPPORTED:
        processes = {}
        popen = subprocess.Popen
        def launch(*args, **kw):
            p = popen(*args, **kw)
            processes[p.pid] = p
            return p
        def wait4(pid, flags):
            p = processes[pid]
            rc = p.poll() if flags else p.wait()
            return (0, 0, None) if rc is None else (pid, rc << 8, SimpleNamespace(ru_maxrss=8192))
        monkeypatch.setattr(runner.subprocess, "Popen", launch)
        monkeypatch.setattr(os, "wait4", wait4, raising=False)
        monkeypatch.setattr(os, "WNOHANG", 1, raising=False)
        monkeypatch.setattr(os, "waitstatus_to_exitcode", lambda status: status >> 8, raising=False)
        monkeypatch.setattr(runner, "WAIT4_SUPPORTED", True)
    path = tmp_path / "manifest.json"
    console = tmp_path / "console"
    def invoke(*extra):
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return runner.main([str(path), "--console-dir", str(console), *extra])
    return SimpleNamespace(**locals())


@pytest.mark.parametrize("count", range(1, 6))
def test_partition_and_seed_major(case, count):
    rows = case.manifest["runs"]
    ordered = runner.shard_plan(case.manifest, 0, 1)
    for start in range(0, len(ordered), len(campaign.ARMS)):
        block = ordered[start:start + len(campaign.ARMS)]
        assert len({r["seed"] for r in block}) == 1
        assert [(r["method"], r["training_profile"]) for r in block] == list(campaign.ARMS)
    shards = [runner.shard_plan(case.manifest, i, count) for i in range(count)]
    assigned = [r["output_directory"] for shard in shards for r in shard]
    assert len(assigned) == len(set(assigned)) == len(rows)
    assert set(assigned) == {r["output_directory"] for r in rows}
    assert all(shards[i] == ordered[i::count] for i in range(count))


@pytest.mark.parametrize("problem", ["head", "tracked", "untracked", "ignored", "topic", "target", "threads", "argv_threads", "console"])
def test_refusal_creates_nothing(case, monkeypatch, problem):
    if problem == "head": case.manifest["source"]["git_commit"] = "wrong"
    elif problem == "tracked": case.state["status"] = " M tracked.py"
    elif problem == "untracked": case.state["status"] = "?? stray.txt"
    elif problem == "ignored": case.state["ignored"] = False
    elif problem == "topic": monkeypatch.delenv("NTFY_TOPIC")
    elif problem == "target": Path(case.manifest["runs"][-1]["output_directory"]).mkdir(parents=True)
    elif problem == "threads": case.manifest["threads"] = 2
    elif problem == "argv_threads":
        argv = case.manifest["runs"][-1]["argv"]
        argv[argv.index("--threads") + 1] = "2"
    elif problem == "console":
        assert case.invoke("--console-dir", str(case.repo / "console")) == 2
        assert not (case.repo / "console").exists()
        return
    assert case.invoke() == 2
    assert not case.console.exists()
    assert not case.events.exists()
    assert not case.notices


def test_dry_run_creates_nothing(case, capsys):
    case.path.write_text(json.dumps(case.manifest), encoding="utf-8")
    before = sorted(str(p) for p in case.tmp_path.rglob("*"))
    assert runner.main([str(case.path), "--console-dir", str(case.console), "--dry-run"]) == 0
    assert sorted(str(p) for p in case.tmp_path.rglob("*")) == before
    assert capsys.readouterr().out.count("PLAN ") == case.manifest["n_runs"]
    assert not case.notices


def test_exact_argv_environment_timing_and_failure_continuation(case):
    # Three real child processes; the middle failure must not drop the third.
    case.manifest["runs"] = runner.shard_plan(case.manifest, 0, 1)[:3]
    case.manifest["n_runs"] = 3
    argv = case.manifest["runs"][1]["argv"]
    argv[argv.index("--fake-rc") + 1] = "7"
    assert case.invoke() == 1
    events = [json.loads(s) for s in case.events.read_text().splitlines()]
    assert [e["argv"] for e in events] == [r["argv"][1:] for r in case.manifest["runs"]]
    assert all(e["hash"] == "0" and e["onednn"] == "1" for e in events)
    with (case.console / "shard-0-of-1.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert reader.fieldnames == runner.TIMING_FIELDS
    assert [r["rc"] for r in rows] == ["0", "7", "0"]
    assert all(float(r["wall_seconds"]) > 0 and float(r["peak_rss_mb"]) > 0 for r in rows)
    log = (case.console / "shard-0-of-1.log").read_text()
    assert 'CHILD_ENV_ADDITIONS {"PYTHONHASHSEED": "0"}' in log
    assert "SHARD_SUMMARY shard-0-of-1 runs=3 failed=1" in log
    assert [kw["priority"] for _, kw in case.notices] == ["default", "high", "default", "high"]


@pytest.mark.parametrize("problem", ["dirty", "target"])
def test_per_run_gate_failure_continues(case, monkeypatch, problem):
    case.manifest["runs"] = runner.shard_plan(case.manifest, 0, 1)[:2]
    case.manifest["n_runs"] = 2
    original = runner.check_fresh
    def gate(row):
        if row is case.manifest["runs"][0]:
            raise AssertionError("JSON reload should copy rows")
        if row["output_directory"] == case.manifest["runs"][0]["output_directory"]:
            if problem == "dirty": case.state["status"] = "?? late.txt"
            else: Path(row["output_directory"]).mkdir(parents=True)
        else:
            case.state["status"] = ""
        original(row)
    monkeypatch.setattr(runner, "check_fresh", gate)
    assert case.invoke() == 1
    assert len(case.events.read_text().splitlines()) == 1
    with (case.console / "shard-0-of-1.csv").open() as f:
        assert [r["rc"] for r in csv.DictReader(f)] == ["2", "0"]


def test_startup_barrier_does_not_accept_old_token(case, monkeypatch):
    case.console.mkdir()
    old = "0" * 32
    (case.console / (old + "-shard-1-of-2.ready")).write_text("fixture-head")
    monkeypatch.setattr(runner, "STARTUP_TIMEOUT_SECONDS", 0)
    assert case.invoke("--of", "2", "--start-together", "1" * 32) == 2
    assert not case.events.exists()
    assert case.notices[-1][1]["priority"] == "high"


def test_successful_shard_releases_current_barrier(case):
    case.manifest["runs"] = case.manifest["runs"][:1]
    case.manifest["n_runs"] = 1
    assert case.invoke("--start-together", "2" * 32) == 0
    assert len(case.events.read_text().splitlines()) == 1
    assert "failed=0" in case.notices[-1][0]


def test_timeout_reaps_with_wait4_and_retains_rss(case, monkeypatch):
    process = SimpleNamespace(pid=123, returncode=None)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: process)
    ticks = iter([0.0, 1.0, 2.0])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(ticks))
    waits = iter([(0, 0, None), (123, 0, SimpleNamespace(ru_maxrss=3072))])
    monkeypatch.setattr(os, "wait4", lambda *a: next(waits))
    kills = []
    monkeypatch.setattr(os, "kill", lambda *a: kills.append(a))
    monkeypatch.setattr(runner.signal, "SIGKILL", 9, raising=False)
    assert runner.measure(["fake"], None, 0.5) == (124, 2.0, 3.0)
    assert kills == [(123, 9)]
