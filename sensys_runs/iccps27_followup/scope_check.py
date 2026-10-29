"""Static local-import closure for the registered A1-A8 training entry points."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLEAN_ROOTS = (
    "sensys_runs/run_single.py",
    "sensys_runs/energy_profile_runtime.py",
    "sensys_runs/energy_reward_runtime.py",
    "lio/alg/config_room_lio.py",
    "lio/alg/config_room_REFiNE.py",
    "lio/alg/train_lio_er.py",
    "lio/alg/train_REFiNE_er.py",
)
C03 = "lio/alg/REFiNE_eia_er.py"


def clean_training_imports(root=ROOT):
    """Resolve static imports with run_single's repo and lio/alg search roots.

    The two dynamic trainer/config selections are explicit CLEAN_ROOTS. External
    packages are not traversed. This is a static closure, not a runtime trace.
    """
    root = Path(root).resolve()
    pending, seen = list(CLEAN_ROOTS), set()

    def candidates(module):
        relative = module.replace(".", "/")
        for prefix in (root, root / "lio/alg"):
            for path in (prefix / (relative + ".py"), prefix / relative / "__init__.py"):
                if path.is_file() and path.resolve().is_relative_to(root):
                    yield path.relative_to(root).as_posix()

    while pending:
        relative = pending.pop()
        if relative in seen:
            continue
        seen.add(relative)
        path = root / relative
        if not path.is_file():
            raise ValueError("Missing clean training source: " + relative)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=relative)
        package = relative.removesuffix(".py").split("/")[:-1]
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package) - node.level + 1] if node.level else []
                base = ".".join(prefix + ([node.module] if node.module else []))
                modules = [base] + [base + "." + a.name if base else a.name
                                    for a in node.names if a.name != "*"]
            for module in modules:
                pending.extend(candidates(module))
    return sorted(seen)


def permitted_c03_exception(root, changed):
    """Allow only the approved EIA file after checking clean-path isolation."""
    reachable = set(clean_training_imports(root))
    if C03 in reachable:
        raise AssertionError("C03 source is reachable from an A1-A8 training path")
    return [p for p in changed if p != C03]
