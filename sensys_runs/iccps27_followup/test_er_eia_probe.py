"""Synthetic contract tests; no real checkpoint inference or validation claim."""
from copy import deepcopy
import csv
import json
from types import SimpleNamespace

import pytest

from sensys_runs import energy_profile_campaign as base
from sensys_runs.energy_profiles import get_profile, get_hardware_profile
from sensys_runs.iccps27_followup import er_eia_probe as er
from sensys_runs.iccps27_followup import source_identity_v2 as identity

HEAD = "c" * 40
TAG = er.RELEASE_TAG
SOURCE = {"synthetic_source.txt": "d" * 64}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def attack(pair):
    return dict(schema_version=1, configured_weights=dict(lever=pair[0], door=pair[1]),
        action_order=["lever", "start", "door"], exploitative_giver_id=1,
        training_active=True, evaluation_active=True,
        effective_multipliers=[dict(giver_id=g, recipient_id=r, delivered=g != r,
            multipliers=[pair[0], 1.0, pair[1]] if g == 1 and r != 1 else [1.0]*3)
            for g in range(4) for r in range(4)])


def matrix_fixture():
    runs, inventory = [], []
    for pair in er.WEIGHTS:
        label = ",".join(str(float(v)) for v in pair)
        for seed in er.SEEDS:
            cfg = base.load_config(er.adapter.METHOD)
            cfg.main.exp_name = er.CAMPAIGN
            cfg.main.dir_name = "lio_eia_er_energy_reward_w" + label + "_s" + str(seed)
            cfg.main.seed = seed
            cfg.main.use_gpu = False
            cfg.env = er.rp.ENVIRONMENT.copy()
            cfg.lio.min_at_lever = 2
            cfg.lio.eia_w_lever, cfg.lio.eia_w_door = pair
            cfg.cost_profile = get_profile("uniform_v1").snapshot()
            hardware = get_hardware_profile("homogeneous_v1")
            cfg.hardware_profile = hardware.snapshot(4)
            cfg.hardware_profile_lambda = list(hardware.vector(4))
            er.adapter.configure(cfg)
            rid = cfg.main.exp_name + "/" + cfg.main.dir_name
            row = dict(method=er.adapter.METHOD, objective=er.adapter.OBJECTIVE,
                followup_arm=er.adapter.ARM, weight_pair=list(pair), weight=label, seed=seed,
                effective_config=cfg, arm="energy_reward_" + label,
                argv=["SYNTHETIC_PYTHON", "er_eia_adapter.py", "--energy_in_reward"])
            runs.append(row)
            inventory.append(dict(run_id=rid, arm=row["arm"], method=row["method"], seed=str(seed),
                checkpoint_prefix="lio/results/" + rid + "/model.ckpt", evidence="SYNTHETIC TEST ONLY"))
    manifest = dict(campaign=er.CAMPAIGN, mode="campaign", n_runs=30, planned_formal_runs=30,
        n_seeds_per_arm=10, methods=[er.adapter.METHOD], weight_pairs=[list(w) for w in er.WEIGHTS],
        source=dict(git_commit=HEAD, git_describe=TAG, sha256=SOURCE), runs=runs)
    return manifest, inventory


def test_registered_matrix_accepts_exact_thirty_rows():
    manifest, inventory = matrix_fixture()
    assert len(er.validate_matrix(manifest, inventory, 30)) == 30


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "wrong_seed", "wrong_count", "energy"])
def test_missing_extra_duplicate_seed_count_and_energy_rows_fail(mutation):
    manifest, inventory = matrix_fixture()
    count = 30
    if mutation == "missing":
        manifest["runs"].pop()
    elif mutation == "extra":
        inventory.append(deepcopy(inventory[0]))
    elif mutation == "duplicate":
        manifest["runs"][1] = deepcopy(manifest["runs"][0])
    elif mutation == "wrong_seed":
        manifest["runs"][0]["seed"] = 7
    elif mutation == "wrong_count":
        count = 29
    else:
        manifest["runs"][0]["effective_config"]["energy_in_reward"] = False
    with pytest.raises(ValueError):
        er.validate_matrix(manifest, inventory, count)


@pytest.fixture
def probe_fixture(tmp_path, monkeypatch):
    manifest, inventory = matrix_fixture()
    results = tmp_path / "results"
    for row, inv in zip(manifest["runs"], inventory):
        cfg = deepcopy(row["effective_config"])
        cfg["eia_attack"] = attack(row["weight_pair"])
        provenance = dict(er.ENERGY_FIELDS, method=er.adapter.METHOD, seed=row["seed"],
            git_commit=HEAD, er_eia_adapter=er.adapter.ADAPTER_ID,
            energy_reward=er.adapter.energy_provenance(deepcopy(cfg)),
            er_eia_source_sha256=SOURCE, er_eia_launch_argv=row["argv"], eia_attack=cfg["eia_attack"])
        run = results / inv["run_id"]
        write(run/"config.json", cfg)
        write(run/"provenance.json", provenance)
        (run/"model.ckpt.index").write_bytes(b"SYNTHETIC INDEX")
        (run/"model.ckpt.data-00000-of-00001").write_bytes(b"SYNTHETIC DATA")
    manifest_path, inventory_path = tmp_path/"campaign.json", tmp_path/"inventory.csv"
    write(manifest_path, manifest)
    with inventory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)
    protected = tmp_path/"protected_manifest.txt"
    protected.write_bytes(b"SYNTHETIC MANIFEST")
    monkeypatch.setattr(er.rp, "PROTECTED_MANIFEST_SHA256", er.rp.sha(protected))
    monkeypatch.setattr(er.adapter, "source_hashes", lambda: SOURCE.copy())
    historical = dict(commit=er.INFERENCE_REF, blobs={"SYNTHETIC_CORE": "e"*40})
    wrapper = dict(commit=HEAD, blobs={"SYNTHETIC_WRAPPER": "f"*40})
    def verified(**kwargs):
        result = historical if kwargs["source_ref"] == er.INFERENCE_REF else wrapper
        if kwargs.get("expected_identity") is not None and kwargs["expected_identity"] != result:
            raise identity.SourceIdentityError("Synthetic receipt identity mismatch")
        return deepcopy(result)
    monkeypatch.setattr(er, "verified_source_identity", verified)
    monkeypatch.setattr(identity, "verified_source_identity", verified)
    monkeypatch.setattr(er, "git", lambda *a: "" if a[0] == "status" else TAG if a[0] == "describe" else HEAD)
    runtime = {"execution": "SYNTHETIC_RUNTIME_ONLY"}
    monkeypatch.setattr(er.rp, "tensorflow_runtime", lambda: (None, runtime))
    expected, checks = [], []
    for method in ("lio_er", "refine_er"):
        for seed in er.rp.VALIDATION_SEEDS:
            rid = "synthetic_clean/" + method + str(seed)
            run = results/rid
            write(run/"config.json", {"synthetic_only": True})
            write(run/"provenance.json", {"synthetic_only": True})
            (run/"model.ckpt.index").write_bytes(b"SYNTHETIC CLEAN INDEX")
            (run/"model.ckpt.data-00000-of-00001").write_bytes(b"SYNTHETIC CLEAN DATA")
            expected.append(dict(run_id=rid, method=method, seed=seed,
                arm="A3" if method == "lio_er" else "A8", expected_P_tau_1=.9,
                reference_result_sha256="0"*64))
            checks.append(dict(run_id=rid, method=method, seed=seed, expected_P_tau_1=.9,
                P1=.9, absolute_difference=0.0, passed=True, runtime=runtime,
                checkpoint_hashes=er.rp.checkpoint_hashes(run),
                config_provenance_hashes={n: er.rp.sha(run/n) for n in ("config.json", "provenance.json")}))
    reference = tmp_path/"synthetic_reference_table.json"
    write(reference, dict(runs=expected, tolerance=1e-5))
    receipt_path = tmp_path/"synthetic_reader_fixture_receipt.json"
    receipt = dict(status="PASS", execution_kind="actual_tensorflow_checkpoint_restore",
        expected=expected, tolerance=1e-5, checks=checks, source_hashes=er.rp.source_hashes(),
        source_git_identity=historical, reference_table=str(reference), reference_table_sha256=er.rp.sha(reference))
    write(receipt_path, receipt)
    calls = []
    def fake_probe(results_root, row, *, eia):
        calls.append(row["run_id"])
        assert eia is True
        return dict(run_id=row["run_id"], P1=.99, E_T_lower=1.01, E_T_upper=1.04, E_T_bound_width=.03,
                    execution_kind="SYNTHETIC_PROBE_TEST_ONLY")
    monkeypatch.setattr(er.rp, "probe_one", fake_probe)
    args = SimpleNamespace(campaign_manifest=manifest_path, inventory=inventory_path, results_root=results,
        validation_receipt=receipt_path, protected_manifest=protected, output_dir=tmp_path/"fresh_probe_output",
        expected_count=30, release_tag=TAG)
    return args, calls, historical, wrapper


def test_reuses_unchanged_batch_and_binds_old_core_new_wrapper_separately(probe_fixture):
    args, calls, historical, wrapper = probe_fixture
    old_function = er.rp.source_git_identity
    result = er.run_probe(args)
    assert len(calls) == 30 and result["status"] == "COMPLETE"
    assert result["inference_source_git_identity"] == historical
    assert result["wrapper_source_git_identity"] == wrapper
    assert result["training_release"] == dict(git_commit=HEAD, git_describe=TAG)
    assert len(result["run_inputs"]) == 30
    assert er.rp.json_read(args.output_dir/"manifest.json")["source_git_identity"] == historical
    assert er.rp.source_git_identity is old_function
    assert er.rp.json_read(args.output_dir/"er_context.json") == result


@pytest.mark.parametrize("fault", ["receipt_fail", "receipt_identity", "receipt_source", "runtime", "checkpoint", "ref", "source", "dirty", "wrong_tag"])
def test_real_data_gates_fail_before_any_inference(probe_fixture, monkeypatch, fault):
    args, calls, _, _ = probe_fixture
    receipt = er.rp.json_read(args.validation_receipt)
    if fault == "receipt_fail":
        receipt["status"] = "FAIL"
    elif fault == "receipt_identity":
        receipt["source_git_identity"]["commit"] = HEAD
    elif fault == "receipt_source":
        receipt["source_hashes"]["changed"] = "0"*64
    elif fault == "runtime":
        receipt["checks"][0]["runtime"] = {"different": True}
    elif fault == "checkpoint":
        (args.results_root/receipt["expected"][0]["run_id"]/"model.ckpt.index").write_bytes(b"CHANGED")
    elif fault == "ref":
        monkeypatch.setattr(er, "git", lambda *a: "" if a[0] == "status" else "wrong-tag" if a[0] == "describe" else HEAD)
    elif fault == "dirty":
        monkeypatch.setattr(er, "git", lambda *a: " M real_source.py" if a[0] == "status" else TAG if a[0] == "describe" else HEAD)
    elif fault == "wrong_tag":
        args.release_tag = "unapproved-tag"
    else:
        def reject(**kwargs):
            raise identity.SourceIdentityError("Genuine source change")
        monkeypatch.setattr(er, "verified_source_identity", reject)
    write(args.validation_receipt, receipt)
    with pytest.raises(ValueError):
        er.run_probe(args)
    assert calls == [] and not args.output_dir.exists()


def test_changed_energy_provenance_fails_before_inference(probe_fixture):
    args, calls, _, _ = probe_fixture
    first = er.rp.json_read(args.campaign_manifest)["runs"][0]["effective_config"]["main"]
    path = args.results_root/first["exp_name"]/first["dir_name"]/"provenance.json"
    record = er.rp.json_read(path)
    record["energy_reward"]["episode_kinds"] = ["ordinary"]
    write(path, record)
    with pytest.raises(ValueError, match="energy provenance"):
        er.run_probe(args)
    assert calls == [] and not args.output_dir.exists()


def test_existing_output_is_not_overwritten(probe_fixture):
    args, calls, _, _ = probe_fixture
    args.output_dir.mkdir()
    sentinel = args.output_dir/"sentinel.txt"
    sentinel.write_bytes(b"UNCHANGED")
    with pytest.raises(ValueError, match="must be new"):
        er.run_probe(args)
    assert calls == [] and sentinel.read_bytes() == b"UNCHANGED"
