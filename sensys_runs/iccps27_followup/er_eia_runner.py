"""Strict ER-LIO+EIA release gate around unchanged EIA dispatch and execution."""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from copy import deepcopy
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sensys_runs import energy_profile_runner as core
from sensys_runs.iccps27_followup import eia_runner as existing
from sensys_runs.iccps27_followup import er_eia_campaign as campaign
from sensys_runs.iccps27_followup.eia_adapter import rebind

ORIGINAL_TMUX_PLAN = existing.tmux_plan
expected_shards = existing.expected_shards


def validate_er_eia(manifest, console, original_validate):
    """Validate actual rows exactly, then project a private copy for the old gate."""
    if manifest.get("mode") not in ("campaign", "smoke"):
        raise core.RunnerError("Unknown ER-LIO+EIA mode")
    if manifest.get("fallback_declared") is not False:
        raise core.RunnerError("ER-LIO+EIA has no declared fallback")
    tag = core.git("describe", "--tags", "--exact-match", "HEAD")
    if tag != campaign.RELEASE_TAG or manifest["source"].get("git_describe") != tag:
        raise core.RunnerError("ER-LIO+EIA requires the exact approved new release tag")
    if manifest["source"].get("git_status") != "":
        raise core.RunnerError("ER-LIO+EIA manifest must have been generated from a clean release")
    expected = campaign.build_campaign(manifest["campaign"], manifest["platform"],
                                       manifest["threads"], manifest["mode"] == "smoke")
    fields = ("n_runs", "n_seeds_per_arm", "weight_pairs", "planned_formal_runs", "methods",
              "objective", "followup_arm", "display_label", "smoke_weight_pair",
              "fallback_declared", "use_gpu", "cost_units")
    for field in fields:
        if manifest.get(field) != expected[field]:
            raise core.RunnerError("ER-LIO+EIA matrix declaration differs: "+field)
    canonical = lambda value: json.dumps(value, sort_keys=True, allow_nan=False)
    if canonical(manifest["runs"]) != canonical(expected["runs"]):
        raise core.RunnerError("ER-LIO+EIA actual run/config/argv differs from the registered matrix")
    if manifest["source"].get("sha256") != expected["source"]["sha256"]:
        raise core.RunnerError("ER-LIO+EIA release source hashes are incomplete or changed")

    # The old validator recognizes energy buffers only for clean lio_er. Validate
    # that same energy contract on a deep copy after the exact EIA rows have been
    # checked. Execution and shard selection retain the original recorded rows.
    projected = deepcopy(manifest)
    for row in projected["runs"]:
        row["method"] = "lio_er"
        index = row["argv"].index("--method")+1
        row["argv"][index] = "lio_er"
    original_validate(projected, console)


@contextmanager
def runner_scope():
    original_validate, original_plan = core.validate, core.shard_plan
    def validate(manifest, console):
        return validate_er_eia(manifest, console, original_validate)
    def plan(manifest, shard, count):
        if count != expected_shards(manifest):
            raise core.RunnerError("ER-LIO+EIA requires K=2 smoke or K=5 formal")
        return original_plan(manifest, shard, count)
    with ExitStack() as stack:
        stack.enter_context(rebind(core, "ARMS", (*core.ARMS, ("lio_eia_er", "uniform_v1"))))
        stack.enter_context(rebind(core, "validate", validate))
        stack.enter_context(rebind(core, "shard_plan", plan))
        yield


def tmux_plan(manifest_path, manifest, count, session, console, token):
    # Reuse the exact command construction, environment forwarding and startup
    # barrier token. Only its child entry-point path changes inside this scope.
    with rebind(existing, "__file__", str(Path(__file__))):
        return ORIGINAL_TMUX_PLAN(manifest_path, manifest, count, session, console, token)


def main(argv=None):
    # Existing main/dispatch keep freshness, CPU, tmux, Linux wait4 and exclusive
    # logs. Existing core.execute receives the original ER-LIO+EIA argv unchanged.
    with rebind(existing, "runner_scope", runner_scope), rebind(existing, "tmux_plan", tmux_plan):
        return existing.main(argv)


if __name__ == "__main__":
    sys.exit(main())
