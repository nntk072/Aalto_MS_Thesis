from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = (
    "configuration",
    "artifacts",
    "market_data",
    "features",
    "trading",
    "simulation",
    "models",
    "decisioning",
    "learning",
    "training",
    "evaluation",
    "experiments",
    "live",
    "integrations",
    "compute",
    "orchestra",
)
LEGACY_PREFIXES = (
    "quant_rl",
    "mt5_trading",
    "evaluation.analysis",
    "simulation.envs",
    "training.train",
)


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _private_cross_domain_imports(tree: ast.Module, package: str, source: str) -> list[str]:
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if (
                    parts[0] in PRODUCTION
                    and parts[0] != package
                    and any(part.startswith("_") for part in parts[1:])
                ):
                    violations.append(f"{source} -> {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            if parts[0] not in PRODUCTION or parts[0] == package:
                continue
            private_modules = [part for part in parts[1:] if part.startswith("_")]
            private_names = [alias.name for alias in node.names if alias.name.startswith("_")]
            details = private_modules + private_names
            if details:
                violations.append(f"{source} -> {node.module}: {', '.join(details)}")
    return violations


def _production_files() -> list[Path]:
    return sorted(
        p
        for package in PRODUCTION
        for p in (ROOT / package).rglob("*.py")
        if "__pycache__" not in p.parts
    )


def test_production_never_imports_scripts() -> None:
    violations = []
    for path in _production_files():
        for name in _imports(path):
            if name == "scripts" or name.startswith("scripts."):
                violations.append(f"{path.relative_to(ROOT)} -> {name}")
    assert not violations, "production imports scripts:\n" + "\n".join(violations)


def test_production_import_graph_is_acyclic() -> None:
    graph: dict[str, set[str]] = {package: set() for package in PRODUCTION}
    for package in PRODUCTION:
        for path in sorted((ROOT / package).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            for name in _imports(path):
                dependency = name.split(".", 1)[0]
                if dependency in graph and dependency != package:
                    graph[package].add(dependency)

    visiting: list[str] = []
    visited: set[str] = set()
    cycles: list[str] = []

    def visit(package: str) -> None:
        if package in visiting:
            cycle = visiting[visiting.index(package) :] + [package]
            cycles.append(" -> ".join(cycle))
            return
        if package in visited:
            return
        visiting.append(package)
        for dependency in sorted(graph[package]):
            visit(dependency)
        visiting.pop()
        visited.add(package)

    for package in PRODUCTION:
        visit(package)
    assert not cycles, "production package dependency cycles:\n" + "\n".join(cycles)


def test_production_cross_domain_imports_are_public() -> None:
    violations: list[str] = []
    for package in PRODUCTION:
        for path in sorted((ROOT / package).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            violations.extend(
                _private_cross_domain_imports(
                    tree,
                    package,
                    path.relative_to(ROOT).as_posix(),
                )
            )
    assert not violations, "private cross-domain imports remain:\n" + "\n".join(violations)


def test_private_cross_domain_import_rule_covers_module_paths() -> None:
    tree = ast.parse("import features._normalizer\nfrom trading.orders._geometry import stop\n")

    assert _private_cross_domain_imports(tree, "training", "training/example.py") == [
        "training/example.py -> features._normalizer",
        "training/example.py -> trading.orders._geometry: _geometry",
    ]


def test_production_has_no_legacy_imports() -> None:
    violations = []
    for path in _production_files():
        for name in _imports(path):
            if any(name == prefix or name.startswith(prefix + ".") for prefix in LEGACY_PREFIXES):
                violations.append(f"{path.relative_to(ROOT)} -> {name}")
    assert not violations, "legacy imports remain:\n" + "\n".join(violations)


DEPENDENCY_RULES = {
    "configuration": {"training", "evaluation", "experiments", "live"},
    "artifacts": {"training", "evaluation", "experiments", "live"},
    "market_data": {"features", "trading", "training", "evaluation"},
    "features": {"training", "evaluation", "live", "integrations"},
    "trading": {"training", "evaluation", "experiments", "scripts"},
    "simulation": {"training", "evaluation", "scripts"},
    "models": {"training", "learning", "simulation", "compute"},
    "live": {"integrations"},
}

FORBIDDEN_IMPORT_PREFIXES = {
    "features": ("trading",),
    "trading": ("simulation.environment",),
    "simulation": ("models",),
    "models": ("simulation", "compute"),
    "evaluation": ("training", "learning.rl", "learning.supervised"),
    "compute": ("models",),
    "decisioning": ("training", "learning", "simulation.environment"),
}

ALLOWED_COMPOSITION_IMPORTS = {
    "evaluation/reanalysis/legacy_sb3.py": ("learning.rl",),
}

RUNTIME_BACKEND_FORBIDDEN = {
    "live": ("learning.rl", "learning.supervised", "training"),
}
EVALUATION_BACKEND_COMPOSITION_ROOTS = {"evaluation/reanalysis/legacy_sb3.py"}


def test_evaluation_and_live_are_backend_neutral() -> None:
    violations: list[str] = []
    for package, forbidden in RUNTIME_BACKEND_FORBIDDEN.items():
        for path in sorted((ROOT / package).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            for name in _imports(path):
                if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden):
                    violations.append(f"{path.relative_to(ROOT)} -> {name}")
    for path in sorted((ROOT / "evaluation").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(ROOT).as_posix()
        for name in _imports(path):
            is_backend_import = name.startswith(("learning.rl", "learning.supervised"))
            if is_backend_import and relative not in EVALUATION_BACKEND_COMPOSITION_ROOTS:
                violations.append(f"{relative} -> {name}")
    assert not violations, "runtime backend coupling remains:\n" + "\n".join(violations)


def test_supervised_learning_does_not_depend_on_rl_or_simulation() -> None:
    forbidden = ("stable_baselines3", "gymnasium", "simulation", "learning.rl", "training")
    violations = []
    for path in sorted((ROOT / "learning" / "supervised").rglob("*.py")):
        for name in _imports(path):
            if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden):
                violations.append(f"{path.relative_to(ROOT)} -> {name}")
    assert not violations, "supervised backend imports RL/simulation code:\n" + "\n".join(
        violations
    )


def test_package_dependency_direction() -> None:
    violations: list[str] = []
    for package, forbidden in DEPENDENCY_RULES.items():
        for path in sorted((ROOT / package).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            for name in _imports(path):
                if name.split(".", 1)[0] in forbidden:
                    violations.append(f"{path.relative_to(ROOT)} -> {name}")
    assert not violations, "forbidden package dependencies:\n" + "\n".join(violations)


def test_scientific_dependency_seams_are_acyclic() -> None:
    violations: list[str] = []
    for package, forbidden_prefixes in FORBIDDEN_IMPORT_PREFIXES.items():
        for path in sorted((ROOT / package).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            for name in _imports(path):
                relative = path.relative_to(ROOT).as_posix()
                forbidden = any(
                    name == prefix or name.startswith(prefix + ".") for prefix in forbidden_prefixes
                )
                allowed_composition = any(
                    name == prefix or name.startswith(prefix + ".")
                    for prefix in ALLOWED_COMPOSITION_IMPORTS.get(relative, ())
                )
                if forbidden and not allowed_composition:
                    violations.append(f"{path.relative_to(ROOT)} -> {name}")
    assert not violations, "reverse scientific dependencies remain:\n" + "\n".join(violations)


TEMPORARY_MODEL_FRAMEWORK_IMPORTS: dict[str, set[str]] = {}


def test_models_framework_imports_are_limited_to_checkpoint_adapters() -> None:
    framework_roots = {"gymnasium", "stable_baselines3"}
    violations: list[str] = []
    observed: set[str] = set()
    for path in sorted((ROOT / "models").rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        for name in _imports(path):
            root = name.split(".", 1)[0]
            if root not in framework_roots:
                continue
            observed.add(relative)
            if root not in TEMPORARY_MODEL_FRAMEWORK_IMPORTS.get(relative, set()):
                violations.append(f"{relative} -> {name}")
    stale_allowlist = sorted(set(TEMPORARY_MODEL_FRAMEWORK_IMPORTS) - observed)
    assert not violations, "model framework imports escaped their adapter allowlist:\n" + "\n".join(
        violations
    )
    assert not stale_allowlist, "resolved model adapter imports remain allowlisted:\n" + "\n".join(
        stale_allowlist
    )


WORKFLOW_PACKAGES = {"training", "evaluation", "experiments", "live", "integrations", "orchestra"}
LEAF_PACKAGES = {
    "configuration",
    "artifacts",
    "market_data",
    "features",
    "trading",
    "simulation",
    "models",
    "decisioning",
    "learning",
    "compute",
}


def test_leaf_packages_do_not_import_application_workflows() -> None:
    violations = []
    for package in LEAF_PACKAGES:
        for path in sorted((ROOT / package).rglob("*.py")):
            for name in _imports(path):
                if name.split(".", 1)[0] in WORKFLOW_PACKAGES:
                    violations.append(f"{path.relative_to(ROOT)} -> {name}")
    assert not violations, "leaf package imports workflow code:\n" + "\n".join(violations)


def test_simulation_environment_uses_explicit_service_imports() -> None:
    wildcard_imports: list[str] = []
    for path in sorted((ROOT / "simulation/environment").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
                wildcard_imports.append(path.relative_to(ROOT).as_posix())
    assert not wildcard_imports, "simulation service wildcard imports remain:\n" + "\n".join(
        wildcard_imports
    )


def test_evaluation_metrics_do_not_import_renderers() -> None:
    violations = []
    for path in sorted((ROOT / "evaluation" / "metrics").rglob("*.py")):
        for name in _imports(path):
            if name.startswith(("evaluation.plotting", "evaluation.reporting")):
                violations.append(f"{path.relative_to(ROOT)} -> {name}")
    assert not violations, "metric implementations import renderers:\n" + "\n".join(violations)


COMPOSITION_ROOT_BUDGETS = {
    "training/runner/data_stage.py": 8,
    "training/runner/seed_runtime.py": 8,
    "training/runner/baselines.py": 8,
    "training/runner/lstm_baseline.py": 8,
    "training/runner/backtest.py": 8,
    "evaluation/reanalysis/runner.py": 8,
    "experiments/variants/runner.py": 8,
}


def test_domain_fanout_is_limited_to_declared_composition_roots() -> None:
    violations: list[str] = []
    stale_roots: list[str] = []
    for package in PRODUCTION:
        for path in sorted((ROOT / package).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(ROOT).as_posix()
            domains = {
                name.split(".", 1)[0]
                for name in _imports(path)
                if name.split(".", 1)[0] in PRODUCTION and name.split(".", 1)[0] != package
            }
            budget = COMPOSITION_ROOT_BUDGETS.get(relative, 4)
            if len(domains) > budget:
                violations.append(f"{relative}: {len(domains)} domains exceeds budget {budget}")
            if relative in COMPOSITION_ROOT_BUDGETS and len(domains) <= 4:
                stale_roots.append(relative)
    assert not violations, "domain import fan-out exceeds its budget:\n" + "\n".join(violations)
    assert not stale_roots, "resolved composition roots remain allowlisted:\n" + "\n".join(
        stale_roots
    )


def test_domain_maps_reference_existing_source_paths() -> None:
    map_files = sorted((ROOT / ".agents/domain_maps").glob("*.md"))
    missing_sections: list[str] = []
    missing_paths: list[str] = []
    missing_axes: list[str] = []
    missing_change_maps: list[str] = []
    for map_path in map_files:
        lines = map_path.read_text(encoding="utf-8").splitlines()
        if "## Axis of Change" not in lines:
            missing_axes.append(map_path.relative_to(ROOT).as_posix())
        if "## Change Map" not in lines and "## Common Change Maps" not in lines:
            missing_change_maps.append(map_path.relative_to(ROOT).as_posix())
        in_sources = False
        paths: list[str] = []
        for line in lines:
            if line == "## Verified Source Paths":
                in_sources = True
                continue
            if in_sources and line.startswith("## "):
                break
            if not in_sources:
                continue
            if line.startswith("| `"):
                paths.append(line.split("|", 2)[1].strip().strip("`"))
            elif line.startswith("- `"):
                paths.append(line[3:].split("`", 1)[0])
        if not paths:
            missing_sections.append(map_path.relative_to(ROOT).as_posix())
        missing_paths.extend(
            f"{map_path.relative_to(ROOT)} -> {path}"
            for path in paths
            if not (ROOT / path).is_file()
        )
    assert not missing_sections, "domain maps need verified source paths:\n" + "\n".join(
        missing_sections
    )
    assert not missing_axes, "domain maps need axis-of-change guidance:\n" + "\n".join(missing_axes)
    assert not missing_change_maps, "domain maps need task-level change guidance:\n" + "\n".join(
        missing_change_maps
    )
    assert not missing_paths, "domain maps reference missing files:\n" + "\n".join(missing_paths)


def test_training_workflow_contexts_keep_preparation_staged() -> None:
    stages_tree = ast.parse((ROOT / "training/runner/stages.py").read_text(encoding="utf-8"))
    setup_classes = {node.name: node for node in stages_tree.body if isinstance(node, ast.ClassDef)}
    preparation = setup_classes["TrainingPreparation"]
    preparation_fields = [
        node.target.id
        for node in preparation.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    ]
    assert preparation_fields == ["configuration", "data", "run_assets"]

    seed_tree = ast.parse((ROOT / "training/runner/seed_runtime.py").read_text(encoding="utf-8"))
    seed_context = next(
        node
        for node in seed_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SeedRunContext"
    )
    seed_fields = [
        node.target.id
        for node in seed_context.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    ]
    assert seed_fields == ["preparation"]
    seed_runner = next(
        node
        for node in seed_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SeedRunner"
    )
    train_method = next(
        node
        for node in seed_runner.body
        if isinstance(node, ast.FunctionDef) and node.name == "train_model"
    )
    direct_learn_calls = [
        node
        for node in ast.walk(train_method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "learn"
    ]
    assert not direct_learn_calls, "seed workflow must fit through Learner.fit()"

    stage_tree = ast.parse((ROOT / "training/runner/seed_stages.py").read_text(encoding="utf-8"))
    stage_names = {node.name for node in stage_tree.body if isinstance(node, ast.ClassDef)}
    assert stage_names >= {
        "PreparedSeedRun",
        "TrainedSeedRun",
        "EvaluatedSeedRun",
        "PersistedSeedRun",
    }
    method_returns = {
        node.name: ast.unparse(node.returns)
        for node in seed_runner.body
        if isinstance(node, ast.FunctionDef) and node.returns is not None
    }
    assert method_returns["prepare_training"] == "PreparedSeedRun"
    assert method_returns["train_model"] == "TrainedSeedRun"
    assert method_returns["evaluate_train_split"] == "EvaluatedSeedRun"
    assert method_returns["write_artifacts"] == "PersistedSeedRun"
