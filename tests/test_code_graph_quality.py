from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "dev" / "code_graph" / "graph_quality.py"
spec = importlib.util.spec_from_file_location("graph_quality", MODULE_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def _db() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(
        """
        create table nodes(
          name text, qualified_name text, file_path text,
          parent_name text, kind text
        );
        create table edges(
          source_qualified text, target_qualified text, file_path text,
          line integer, target_resolution text, extra text, kind text
        );
        """
    )
    return c


def test_config_path_resolves_literal_path_join_and_constant() -> None:
    tree = mod.ast.parse('ROOT / "config" / "base" / "default.yaml"', mode="eval")
    assert mod._config_path(tree.body, {}) == "config/base/default.yaml"
    literal = mod.ast.parse('"config/2_po3_ifvg.yaml"', mode="eval")
    assert mod._config_path(literal.body, {}) == "config/2_po3_ifvg.yaml"
    expr = mod.ast.parse("CONFIG", mode="eval")
    assignment = mod.ast.parse('CONFIG = ROOT / "config" / "3_distribution.yaml"').body[0]
    assert isinstance(assignment, mod.ast.Assign)
    assert mod._config_path(expr.body, {"CONFIG": assignment.value}) == "config/3_distribution.yaml"


def test_config_path_rejects_non_config_text() -> None:
    expr = mod.ast.parse('Path("README.md")', mode="eval")
    assert mod._config_path(expr.body, {}) is None


def test_registry_relations_extract_literal_mapping_and_constant() -> None:
    tree = mod.ast.parse(
        'ROOT = Path(".")\n'
        'QUALITY = ROOT / "config" / "quality_experiments.yaml"\n'
        'STRATEGIES = {"distribution": ROOT / "config" / "3_distribution.yaml"}\n'
    )
    assignments = mod._module_assignments(tree)
    rels = mod._registry_relations("training/configuration/example.py", tree, assignments)
    keyed = {(r["relation_type"], r["config_path"], r["selector_or_key"]) for r in rels}
    assert ("CONFIG_CONSTANT", "config/quality_experiments.yaml", "QUALITY") in keyed
    assert ("REGISTRY_ENTRY", "config/3_distribution.yaml", "distribution") in keyed


def test_function_default_config_path_resolves_constant() -> None:
    tree = mod.ast.parse(
        'DEFAULT = ROOT / "config" / "quality_experiments.yaml"\n'
        "def load(path=DEFAULT):\n    return path\n"
    )
    defaults = mod._function_default_config_paths(tree, mod._module_assignments(tree))
    assert defaults[("load", "path")] == "config/quality_experiments.yaml"
    wrapped = mod.ast.parse("Path(path)", mode="eval")
    assert mod._default_param_ref(wrapped.body) == "path"


def test_registry_relations_include_dynamic_selection_read() -> None:
    tree = mod.ast.parse(
        'STRATEGIES = {"distribution": ROOT / "config" / "3_distribution.yaml"}\n'
        "def select(name):\n    return STRATEGIES[name]\n"
    )
    rels = mod._registry_relations(
        "training/configuration/cli.py", tree, mod._module_assignments(tree)
    )
    selections = [r for r in rels if r["relation_type"] == "REGISTRY_SELECTION"]
    assert len(selections) == 1
    assert selections[0]["config_path"] == "config/3_distribution.yaml"
    assert selections[0]["selector_or_key"] == "<dynamic>"
    assert selections[0]["source_symbol"] == "select"


def test_cfg_accesses_extracts_attribute_and_subscript() -> None:
    tree = mod.ast.parse('x = cfg.env.reward_mode\ny = config["seed"]\n')
    assert mod._cfg_accesses(tree) == {"env.reward_mode", "seed"}


def test_classify_calls_distinguishes_direct_builtin_external_and_dynamic_get(monkeypatch) -> None:
    c = _db()
    src = str((mod.ROOT / "pkg" / "a.py").resolve())
    c.executemany(
        "insert into nodes values (?,?,?,?,?)",
        [
            ("only_here", f"{src}::only_here", src, "", "Function"),
            ("get", f"{src}::Store.get", src, "Store", "Function"),
        ],
    )
    c.executemany(
        "insert into edges values (?,?,?,?,?,?,?)",
        [
            (f"{src}::f", f"{src}::only_here", src, 1, "direct", "{}", "CALLS"),
            (f"{src}::f", "len", src, 2, "unresolved", "{}", "CALLS"),
            (
                f"{src}::f",
                "array",
                src,
                3,
                "unresolved",
                '{"receiver":"np","receiver_binding":"class","receiver_class":"np"}',
                "CALLS",
            ),
            (
                f"{src}::f",
                "get",
                src,
                4,
                "unresolved",
                '{"receiver":"data","receiver_binding":"unknown"}',
                "CALLS",
            ),
        ],
    )
    monkeypatch.setattr(mod, "_module_imports", lambda path: {"np": "numpy"})
    monkeypatch.setattr(mod, "_repo_top_modules", lambda: {"pkg"})
    monkeypatch.setattr(mod, "_parse_file", lambda path: ("", mod.ast.parse("")))
    audits, metrics, _ = mod.classify_calls(c)
    assert [x.category for x in audits] == [
        "internal_exact",
        "python_builtin",
        "external_library",
        "dynamic_or_indeterminate",
    ]
    assert metrics["counts"]["dynamic_or_indeterminate"] == 1


def test_same_module_unique_unqualified_call_becomes_exact(monkeypatch) -> None:
    c = _db()
    src = str((mod.ROOT / "pkg" / "a.py").resolve())
    c.execute(
        "insert into nodes values (?,?,?,?,?)", ("helper", f"{src}::helper", src, "", "Function")
    )
    c.execute(
        "insert into edges values (?,?,?,?,?,?,?)",
        (f"{src}::f", "helper", src, 10, "unresolved", "{}", "CALLS"),
    )
    monkeypatch.setattr(mod, "_module_imports", lambda path: {})
    monkeypatch.setattr(mod, "_repo_top_modules", lambda: {"pkg"})
    monkeypatch.setattr(
        mod, "_parse_file", lambda path: ("", mod.ast.parse("def f():\n    helper()\n"))
    )
    audits, _, _ = mod.classify_calls(c)
    assert audits[0].category == "internal_exact"
    assert "same-module" in audits[0].resolution_reason


def test_self_method_resolves_to_enclosing_class(monkeypatch) -> None:
    c = _db()
    src = str((mod.ROOT / "pkg" / "a.py").resolve())
    other = str((mod.ROOT / "pkg" / "b.py").resolve())
    c.executemany(
        "insert into nodes values (?,?,?,?,?)",
        [
            ("step", f"{src}::A.step", src, "A", "Function"),
            ("step", f"{other}::B.step", other, "B", "Function"),
        ],
    )
    c.execute(
        "insert into edges values (?,?,?,?,?,?,?)",
        (
            f"{src}::A.run",
            "step",
            src,
            3,
            "unresolved",
            '{"receiver":"self","receiver_binding":"class","receiver_class":"A"}',
            "CALLS",
        ),
    )
    tree = mod.ast.parse("class A:\n    def run(self):\n        self.step()\n")
    monkeypatch.setattr(mod, "_module_imports", lambda path: {})
    monkeypatch.setattr(mod, "_repo_top_modules", lambda: {"pkg"})
    monkeypatch.setattr(mod, "_parse_file", lambda path: ("", tree))
    audits, _, _ = mod.classify_calls(c)
    assert audits[0].category == "internal_exact"
    assert audits[0].candidate_targets == [f"{src}::A.step"]


def test_inherited_self_method_resolves_unique_direct_base(monkeypatch) -> None:
    c = _db()
    child = str((mod.ROOT / "pkg" / "child.py").resolve())
    base = str((mod.ROOT / "pkg" / "base.py").resolve())
    c.executemany(
        "insert into nodes values (?,?,?,?,?)",
        [
            ("run", f"{base}::Base.run", base, "Base", "Function"),
            (
                "run",
                f"{mod.ROOT / 'other.py'}::Other.run",
                str(mod.ROOT / "other.py"),
                "Other",
                "Function",
            ),
        ],
    )
    c.executemany(
        "insert into edges values (?,?,?,?,?,?,?)",
        [
            (
                f"{child}::Child.call",
                "run",
                child,
                3,
                "unresolved",
                '{"receiver":"self","receiver_binding":"class","receiver_class":"Child"}',
                "CALLS",
            ),
            (f"{child}::Child", "Base", child, 1, None, "{}", "INHERITS"),
        ],
    )
    tree = mod.ast.parse("class Child(Base):\n    def call(self):\n        self.run()\n")
    monkeypatch.setattr(mod, "_module_imports", lambda path: {})
    monkeypatch.setattr(mod, "_repo_top_modules", lambda: {"pkg", "other"})
    monkeypatch.setattr(mod, "_parse_file", lambda path: ("", tree))
    monkeypatch.setattr(mod, "_reexport_map", lambda: {})
    audits, _, _ = mod.classify_calls(c)
    assert audits[0].category == "internal_exact"
    assert audits[0].candidate_targets == [f"{base}::Base.run"]
    assert "direct base class" in audits[0].resolution_reason


def test_explicit_import_unique_candidate_becomes_exact(monkeypatch) -> None:
    c = _db()
    src = str((mod.ROOT / "pkg" / "caller.py").resolve())
    target = str((mod.ROOT / "pkg" / "helpers.py").resolve())
    c.executemany(
        "insert into nodes values (?,?,?,?,?)",
        [
            ("build", f"{target}::build", target, "", "Function"),
            (
                "build",
                f"{mod.ROOT / 'other.py'}::build",
                str(mod.ROOT / "other.py"),
                "",
                "Function",
            ),
        ],
    )
    c.execute(
        "insert into edges values (?,?,?,?,?,?,?)",
        (f"{src}::run", "build", src, 2, "unresolved", "{}", "CALLS"),
    )
    monkeypatch.setattr(mod, "_module_imports", lambda path: {"build": "pkg.helpers.build"})
    monkeypatch.setattr(mod, "_repo_top_modules", lambda: {"pkg", "other"})
    monkeypatch.setattr(mod, "_parse_file", lambda path: ("", mod.ast.parse("")))
    audits, _, _ = mod.classify_calls(c)
    assert audits[0].category == "internal_exact"
    assert "explicit import" in audits[0].resolution_reason


def test_reexport_chain_is_followed_for_import_resolution(monkeypatch) -> None:
    c = _db()
    src = str((mod.ROOT / "caller.py").resolve())
    target = str((mod.ROOT / "features" / "pipeline" / "builder.py").resolve())
    c.execute(
        "insert into nodes values (?,?,?,?,?)",
        ("build_features", f"{target}::build_features", target, "", "Function"),
    )
    c.execute(
        "insert into edges values (?,?,?,?,?,?,?)",
        (f"{src}::run", "build_features", src, 2, "unresolved", "{}", "CALLS"),
    )
    monkeypatch.setattr(
        mod, "_module_imports", lambda path: {"build_features": "features.build_features"}
    )
    monkeypatch.setattr(
        mod,
        "_reexport_map",
        lambda: {
            "features.build_features": "features.pipeline.build_features",
            "features.pipeline.build_features": "features.pipeline.builder.build_features",
        },
    )
    monkeypatch.setattr(mod, "_repo_top_modules", lambda: {"features"})
    monkeypatch.setattr(mod, "_parse_file", lambda path: ("", mod.ast.parse("")))
    audits, _, _ = mod.classify_calls(c)
    assert audits[0].category == "internal_exact"
    assert audits[0].candidate_targets == [f"{target}::build_features"]


def test_verify_critical_fails_unresolved_required_config(monkeypatch) -> None:
    c = _db()
    monkeypatch.setattr(mod, "graph_file_set", lambda conn: {"a.py", "x.sh"})
    monkeypatch.setattr(
        mod, "tracked_files", lambda suffixes: ["a.py"] if suffixes == (".py",) else ["x.sh"]
    )
    monkeypatch.setattr(mod, "CRITICAL_SYMBOLS", ())
    monkeypatch.setattr(
        mod,
        "CRITICAL_CONFIG_POLICY",
        {"config/3_distribution.yaml": {"directly_traceable", "indirectly_traceable"}},
    )
    cfg = {
        "config_files": ["config/3_distribution.yaml"],
        "reachability": {
            "config/3_distribution.yaml": {"state": "unresolved", "evidence": [], "note": "missing"}
        },
    }
    result = mod.verify_critical(c, cfg)
    assert result["ok"] is False
    assert result["critical_configs"][0]["result"] is False


def test_verify_critical_allows_explicit_inactive_policy(monkeypatch) -> None:
    c = _db()
    monkeypatch.setattr(mod, "graph_file_set", lambda conn: {"a.py", "x.sh"})
    monkeypatch.setattr(
        mod, "tracked_files", lambda suffixes: ["a.py"] if suffixes == (".py",) else ["x.sh"]
    )
    monkeypatch.setattr(mod, "CRITICAL_SYMBOLS", ())
    monkeypatch.setattr(
        mod, "CRITICAL_CONFIG_POLICY", {"config/1_baseline.yaml": {"tracked_but_inactive"}}
    )
    cfg = {
        "config_files": ["config/1_baseline.yaml"],
        "reachability": {
            "config/1_baseline.yaml": {
                "state": "tracked_but_inactive",
                "evidence": [],
                "note": "intentional",
            }
        },
    }
    result = mod.verify_critical(c, cfg)
    assert result["ok"] is True
