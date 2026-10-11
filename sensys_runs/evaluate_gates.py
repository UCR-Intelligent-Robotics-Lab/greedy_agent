"""Evaluate registered gates from seed summaries only; never read trajectories.

Thresholds are pinned to the gates tag. Missing data are not evidence of failure.
All input bytes used are hashed; duplicate/conflicting seed records are refused.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import io
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GATES_TAG = "iccps2027-energy-gates"
GATES_COMMIT = "4755ea5bd00c10ff937fde8b5efc3648d55dc497"
GATES_PATH = "docs/iccps2027-energy/gates.md"
# One registry, including sampling settings and the complete primary family.
REGISTRY = {
    "seeds": {"value": list(range(12341, 12351)), "section": "Statistics used by every gate"},
    "evaluation_episodes": {"value": 100, "section": "Statistics used by every gate"},
    "alpha": {"value": 0.05, "section": "Statistics used by every gate"},
    "primary": {"value": ["G1", "G3a", "G3b"], "section": "Statistics used by every gate"},
    "confidence": {"value": 0.95, "section": "Statistics used by every gate"},
    "resamples": {"value": 10000, "section": "Statistics used by every gate"},
    "generator_seed": {"value": 20261010, "section": "Statistics used by every gate"},
    "relative_cost": {"value": -0.05, "section": "G1"},
    "success_margin": {"value": -0.05, "section": "G1"},
    "support_reference": {"value": 0.5, "section": "G3"},
    "support_effect": {"value": 0.65, "section": "G3"},
    "max_excluded": {"value": 2, "section": "G3"},
    "sanity_range": {"value": [0.35, 0.65], "section": "G3"},
}
HOMO, HETERO = "homogeneous_v1", "hetero_2x1p5_v1"
IDENTITY = ("method", "training_profile", "objective", "training_hardware_profile",
            "evaluation_profile", "evaluation_hardware_profile")
COST = "mean_agent_mission_cost"
WORST = "worst_agent_mean_mission_cost"
SUPPORT = "support_share"
METRICS = (COST, WORST, "success_rate", "support_cost_component", SUPPORT)


class GateError(ValueError):
    """Invalid or inconsistent gate inputs; CLI exits 2 without a report."""


def setting(name: str):
    return REGISTRY[name]["value"]


def finite(values) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise GateError("Expected a nonempty finite seed vector")
    if len(result) > len(setting("seeds")):
        raise GateError("More observations than registered training seeds")
    return result


def wilcoxon(differences, direction: str) -> dict:
    """Enumerate every sign assignment of average ranks; no asymptotic fallback."""
    d = finite(differences)
    if direction not in ("less", "greater"):
        raise GateError("Expected a one-sided direction")
    nonzero = d[d != 0]
    ranks = np.empty(len(nonzero))
    magnitudes = np.abs(nonzero)
    for magnitude in np.unique(magnitudes):
        positions = np.flatnonzero(np.sort(magnitudes) == magnitude) + 1
        ranks[magnitudes == magnitude] = np.mean(positions)
    # Doubled ranks make tied-rank sums exact integers.
    ranks2 = (2 * ranks).astype(int)
    observed = int(ranks2[nonzero > 0].sum())
    extreme = 0
    for signs in itertools.product((0, 1), repeat=len(nonzero)):
        total = sum(rank * sign for rank, sign in zip(ranks2, signs))
        extreme += total <= observed if direction == "less" else total >= observed
    return dict(p=float(extreme / (2 ** len(nonzero))), direction=direction,
                zero_count=int(len(d) - len(nonzero)), n_nonzero=len(nonzero),
                positive_rank_sum=observed / 2, sign_assignments=2 ** len(nonzero))


def holm(pvalues: dict) -> dict:
    """Missing primary tests occupy their family slot with p=1, never shrink m."""
    family = setting("primary")
    if set(pvalues) - set(family):
        raise GateError("Unexpected primary test")
    ordered = sorted(family, key=lambda k: pvalues.get(k, 1.0))
    adjusted, previous = {}, 0.0
    for i, name in enumerate(ordered):
        p = pvalues.get(name, 1.0)
        if not 0 <= p <= 1:
            raise GateError("Invalid p value")
        previous = min(1.0, max(previous, (len(family) - i) * p))
        adjusted[name] = previous if name in pvalues else None
    return adjusted


def bootstrap_median(values) -> list[float]:
    d = finite(values)
    rng = np.random.default_rng(setting("generator_seed"))
    medians = np.median(rng.choice(d, size=(setting("resamples"), len(d)), replace=True), axis=1)
    tail = (1 - setting("confidence")) / 2
    return np.quantile(medians, [tail, 1 - tail], method="linear").tolist()


def statistics(values, direction: str, seeds: list[int]) -> dict:
    d = finite(values)
    if len(d) != len(seeds):
        raise GateError("Seed/vector length mismatch")
    wins = int(np.count_nonzero(d < 0 if direction == "less" else d > 0))
    return dict(median=float(np.median(d)), interval=bootstrap_median(d),
                probability_of_improvement=wins / len(d), improved_seed_count=wins,
                n_seeds=len(d), seeds=seeds, differences=d.tolist(), **wilcoxon(d, direction))


def not_evaluated(reason: str) -> dict:
    return dict(status="NOT_EVALUATED", reason=reason)


def load_summaries(directories: list[Path], hashes: dict) -> dict:
    """Merge seed and hardware columns, never aggregate episode or summary rows."""
    records = {}
    for directory in directories:
        found = False
        for filename in ("seed_metrics.csv", "hardware_seed_metrics.csv"):
            path = directory / filename
            if not path.exists():
                continue
            found = True
            raw = path.read_bytes()
            hashes[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
            reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
            required = {"seed", "method", "training_profile", "evaluation_profile",
                        "training_hardware_profile", "evaluation_hardware_profile"}
            if not required <= set(reader.fieldnames or ()):
                raise GateError(f"Missing explicit seed/arm/hardware columns: {path}")
            seen = set()
            for row in reader:
                if None in row or any(v is None for v in row.values()):
                    raise GateError(f"Malformed row: {path}")
                if "n_episodes" in row and int(row["n_episodes"]) != setting("evaluation_episodes"):
                    raise GateError("Gate inputs require the registered final evaluation episode count")
                row.setdefault("objective", "env_reward")
                key = tuple(row[k] for k in IDENTITY)
                seed = int(row["seed"])
                identity = (*key, seed)
                if identity in seen:
                    raise GateError(f"Duplicate seed within arm: {path}")
                seen.add(identity)
                if seed not in setting("seeds") or any(not part for part in key):
                    raise GateError(f"Unregistered seed or missing arm identity: {path}")
                if row["objective"] not in ("env_reward", "energy_reward"):
                    raise GateError("Unknown objective")
                target = records.setdefault(key, {}).setdefault(seed, {})
                for field in (*METRICS, "support_share_status", "run_id"):
                    if field not in row:
                        continue
                    value = row[field]
                    if field == SUPPORT and row.get("support_share_status") == "no lever actions" and value != "":
                        raise GateError("No-lever seed must not have a defined support share")
                    if field in METRICS and value != "":
                        number = float(value)
                        if not np.isfinite(number):
                            raise GateError(f"Non-finite {field}: {path}")
                        if field in (SUPPORT, "success_rate") and not 0 <= number <= 1:
                            raise GateError(f"Out-of-range {field}: {path}")
                        if field in (COST, WORST, "support_cost_component") and number < 0:
                            raise GateError(f"Negative cost: {path}")
                    if field in target and target[field] != value:
                        raise GateError(f"Conflicting seed metric {field}: {path}")
                    target[field] = value
        if not found:
            raise GateError(f"No seed-level input files: {directory}")
    return records


def select(records: dict, method: str, profile: str, training_hardware: str,
           table: str, evaluation_hardware: str, objective="env_reward", accounting=False):
    matches = [rows for key, rows in records.items()
               if key[0] == method and key[2] == objective and key[4:] == (table, evaluation_hardware)
               and ((accounting and method == "lio_er" and objective == "env_reward")
                    or (key[1] == profile and key[3] == training_hardware))]
    if len(matches) > 1:
        raise GateError("Ambiguous reference arm; supply one accounting-only LIO reference")
    return matches[0] if matches else None


def paired_seeds(left: dict, right: dict) -> list[int]:
    if left.keys() != right.keys():
        raise GateError("Paired arms have different seed sets")
    if set(left) != set(setting("seeds")):
        raise GateError("Expected the complete registered ten-seed set before exclusions")
    return sorted(left)


def differences(left, right, metric, seeds):
    # CSV decimals avoid representation artifacts at inclusive gate boundaries.
    return [float(Decimal(str(left[s][metric])) - Decimal(str(right[s][metric]))) for s in seeds]


def comparison(left, right, metric, direction="less", support=False) -> dict:
    if left is None or right is None:
        return not_evaluated("Required arm absent")
    seeds = paired_seeds(left, right)
    excluded = {"method": [], "reference": []}
    if support:
        for label, arm in (("method", left), ("reference", right)):
            for s in seeds:
                if arm[s].get("support_share_status") == "no lever actions":
                    excluded[label].append(s)
                elif arm[s].get("support_share_status") != "defined" or arm[s].get(metric, "") == "":
                    return not_evaluated("Support share requires heterogeneous evaluation rescoring")
        seeds = [s for s in seeds if s not in excluded["method"] and s not in excluded["reference"]]
    if not seeds:
        return dict(status="NOT_EVALUATED", reason="No paired defined shares", excluded=excluded)
    if any(arm[s].get(metric, "") == "" for arm in (left, right) for s in seeds):
        return not_evaluated(f"Missing seed metric: {metric}")
    values = differences(left, right, metric, seeds)
    return dict(status="REPORTED", metric=metric, excluded=excluded,
                **statistics(values, direction, seeds))


def g1_conditions(cost: dict, success: dict, adjusted: float) -> dict:
    return dict(test=adjusted < setting("alpha"),
                effect=cost["relative_median"] <= setting("relative_cost"),
                success_noninferiority=success["interval"][0] >= setting("success_margin"))


def decision(g1: str, g3a: str, g3b: str) -> str:
    if g1 == "NOT_DETECTED":
        return "Discuss re-scoping or venue with the advisor."
    if "INCONCLUSIVE" in (g3a, g3b):
        return "G3 INCONCLUSIVE: report and discuss; no claim."
    if g1 == "PASS" and g3a == g3b == "PASS":
        return "The main line stands."
    if g1 == "PASS" and "NOT_DETECTED" in (g3a, g3b):
        return "Narrow the claim to cost reduction; no role-routing claim."
    return "NOT_EVALUATED: await missing gate evidence."


def evaluate(waves: dict) -> dict:
    w1, w2, w3 = (waves.get(k, {}) for k in ("wave1", "wave2", "wave3"))
    gates = {g: not_evaluated("Required wave or arm absent") for g in ("G1", "G2", "G3a", "G3b", "G4")}
    primary = {}
    descriptive = {}
    for profile in ("uniform_v1", "low_contrast_v1", "paper_ratio_v1"):
        ref = select(w1, "lio_er", "uniform_v1", HOMO, profile, HOMO)
        method = select(w1, "refine_er", profile, HOMO, profile, HOMO)
        cost = comparison(method, ref, COST)
        success = comparison(method, ref, "success_rate", "greater")
        if cost["status"] == "REPORTED":
            seeds = cost["seeds"]
            if any(Decimal(str(ref[s][COST])) <= 0 for s in seeds):
                raise GateError("Relative cost needs positive reference costs")
            relative = [float((Decimal(str(method[s][COST])) - Decimal(str(ref[s][COST])))
                              / Decimal(str(ref[s][COST]))) for s in seeds]
            cost.update(relative_median=float(np.median(relative)), relative_differences=relative,
                        relative_interval=bootstrap_median(relative))
        result = dict(status="REPORTED" if cost["status"] == success["status"] == "REPORTED"
                      else "NOT_EVALUATED", cost=cost, success=success)
        if profile == "uniform_v1":
            gates["G1"] = result
            if result["status"] == "REPORTED":
                primary["G1"] = cost["p"]
        else:
            descriptive[profile] = result
    gates["G2"] = comparison(select(w1, "refine_er", "paper_ratio_v1", HOMO, "paper_ratio_v1", HOMO),
                             select(w1, "refine_er", "uniform_v1", HOMO, "paper_ratio_v1", HOMO),
                             "support_cost_component")
    a = select(w2, "refine_er", "paper_ratio_v1", HETERO, "paper_ratio_v1", HETERO)
    b = select(w2, "refine_er", "uniform_v1", HETERO, "paper_ratio_v1", HETERO)
    excluded = {}
    for name, arm in (("paper_ratio_v1", a), ("uniform_v1", b)):
        if arm is not None:
            paired_seeds(arm, arm)
            excluded[name] = [s for s in sorted(arm) if arm[s].get("support_share_status") == "no lever actions"]
            for s in set(arm) - set(excluded[name]):
                if arm[s].get("support_share_status") != "defined" or arm[s].get(SUPPORT, "") == "":
                    raise GateError("G3 requires defined share or explicit no lever actions")
    if a is not None and b is not None:
        paired_seeds(a, b)
    if any(len(seeds) > setting("max_excluded") for seeds in excluded.values()):
        gates["G3a"] = gates["G3b"] = dict(status="INCONCLUSIVE", excluded=excluded,
                                               reason="More than two seeds excluded in an arm")
    else:
        if a is not None:
            seeds = sorted(set(a) - set(excluded["paper_ratio_v1"]))
            shares = [float(a[s][SUPPORT]) for s in seeds]
            values = [float(Decimal(str(a[s][SUPPORT])) - Decimal(str(setting("support_reference")))) for s in seeds]
            gates["G3a"] = dict(status="REPORTED", excluded=excluded,
                                 median_share=float(np.median(shares)), **statistics(values, "greater", seeds))
            primary["G3a"] = gates["G3a"]["p"]
        gates["G3b"] = comparison(a, b, SUPPORT, "greater", support=True)
        if gates["G3b"]["status"] == "REPORTED":
            primary["G3b"] = gates["G3b"]["p"]
    worst = comparison(a, b, WORST)
    sanity = {}
    for method in ("lio_er", "refine_er"):
        arm = select(w1, method, "uniform_v1", HOMO, "uniform_v1", HETERO)
        if arm is None:
            sanity[method] = not_evaluated("Provide homogeneous training rescored with hetero evaluation hardware")
            continue
        paired_seeds(arm, arm)
        omitted = [s for s in arm if arm[s].get("support_share_status") == "no lever actions"]
        valid = sorted(set(arm) - set(omitted))
        if not valid:
            sanity[method] = not_evaluated("No lever actions in any seed")
            sanity[method]["excluded"] = sorted(omitted)
            continue
        if any(arm[s].get("support_share_status") != "defined" or arm[s].get(SUPPORT, "") == "" for s in valid):
            raise GateError("Sanity reference requires defined agent-group shares")
        median = float(np.median([float(arm[s][SUPPORT]) for s in valid]))
        low, high = setting("sanity_range")
        sanity[method] = dict(status="REPORTED", median_share=median, n_seeds=len(valid), seeds=valid,
                              excluded=sorted(omitted), agent_index_asymmetry=not low <= median <= high)
    # References may come from another wave; retain each training/evaluation identity.
    combined = {}
    for wave in (w1, w2, w3):
        for key, rows in wave.items():
            if key in combined and combined[key] != rows:
                raise GateError("Conflicting copies of an arm across waves")
            combined[key] = rows
    g4 = {}
    for hardware in (HOMO, HETERO):
        method = select(w3, "lio_er", "paper_ratio_v1", hardware, "paper_ratio_v1", hardware, "energy_reward")
        for reference in ("lio_er", "refine_er"):
            ref = select(combined, reference, "paper_ratio_v1", hardware, "paper_ratio_v1", hardware,
                         accounting=reference == "lio_er")
            metrics = {}
            for metric in (COST, WORST, "success_rate", *((SUPPORT,) if hardware == HETERO else ())):
                # G4 has no registered prediction: report both tails explicitly.
                metrics[metric] = {direction: comparison(method, ref, metric, direction, metric == SUPPORT)
                                   for direction in ("less", "greater")}
            g4[hardware + "/" + reference] = metrics
    gates["G4"] = dict(status="REPORTED" if any(r["less"]["status"] == "REPORTED"
                          for metrics in g4.values() for r in metrics.values()) else "NOT_EVALUATED",
                       direction="Not pre-registered; both one-sided directions reported, no pass line", comparisons=g4)
    adjusted = holm(primary)
    for name in setting("primary"):
        if name not in primary:
            continue
        gate = gates[name]
        gate["holm_p"] = adjusted[name]
        if name == "G1":
            conditions = g1_conditions(gate["cost"], gate["success"], adjusted[name])
        else:
            conditions = dict(test=adjusted[name] < setting("alpha"))
            if name == "G3a":
                conditions["effect"] = gate["median_share"] >= setting("support_effect")
        gate.update(conditions=conditions, status="PASS" if all(conditions.values()) else "NOT_DETECTED")
    return dict(gates=gates, g1_descriptive=descriptive, g3_worst_agent_cost=worst,
                sanity_reference=sanity, holm=dict(adjusted=adjusted, raw=primary,
                missing_policy="Missing primary p values occupy their fixed family slot as 1"),
                decision=decision(*(gates[k]["status"] for k in ("G1", "G3a", "G3b"))))


def gate_provenance() -> dict:
    commit = subprocess.check_output(["git", "rev-parse", GATES_TAG + "^{commit}"], cwd=ROOT, text=True).strip()
    if commit != GATES_COMMIT:
        raise GateError("Gates tag differs from the registered commit")
    raw = subprocess.check_output(["git", "show", commit + ":" + GATES_PATH], cwd=ROOT)
    return dict(tag=GATES_TAG, commit=commit, path=GATES_PATH, sha256=hashlib.sha256(raw).hexdigest())


def markdown(report: dict) -> str:
    # Full nested detail avoids a terse status table hiding exclusions or missing tests.
    lines = ["# ICCPS 2027 registered gates", "", report["decision"], "",
             "A non-detection is not evidence of no effect. Sanity flags precede G3 interpretation.", ""]
    for name in ("sanity_reference", "gates", "g1_descriptive", "g3_worst_agent_cost", "holm",
                 "registration", "registry", "input_sha256"):
        lines += ["## " + name, "", "```json", json.dumps(report[name], indent=2, sort_keys=True), "```", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for wave in ("wave1", "wave2", "wave3"):
        parser.add_argument("--" + wave, type=Path, nargs="+", default=[], help="Summary directories only")
    parser.add_argument("--out", type=Path, required=True, help="Fresh report directory")
    args = parser.parse_args(argv)
    try:
        if args.out.exists() or args.out.is_symlink():
            raise GateError("Output directory must not exist")
        registration = gate_provenance()
        hashes = {}
        waves = {wave: load_summaries(getattr(args, wave), hashes) for wave in ("wave1", "wave2", "wave3")}
        report = evaluate(waves)
        report.update(registration=registration, registry=REGISTRY, input_sha256=hashes)
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / "gates_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        (args.out / "gates_report.md").write_text(markdown(report), encoding="utf-8")
        print("Wrote gates_report.json and gates_report.md to " + str(args.out))
        return 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print("CONFIG_ERROR: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
