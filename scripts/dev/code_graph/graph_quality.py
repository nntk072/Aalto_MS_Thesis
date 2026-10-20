#!/usr/bin/env python3
"""Repository-owned code-review-graph quality and configuration dependency audits."""

from __future__ import annotations

import argparse
import ast
import builtins
import json
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / ".code-review-graph" / "graph.db"
OUT = ROOT / "outputs" / "code_graph_quality"

CRITICAL_CONFIG_POLICY: dict[str, set[str]] = {
    "pyproject.toml": {"directly_traceable", "indirectly_traceable"},
    "config/base/default.yaml": {"directly_traceable", "indirectly_traceable"},
    "config/experiments.yaml": {"directly_traceable", "indirectly_traceable"},
    "config/quality_experiments.yaml": {"directly_traceable", "indirectly_traceable"},
    "config/1_baseline.yaml": {"tracked_but_inactive"},
    "config/2_po3_ifvg.yaml": {"directly_traceable", "indirectly_traceable"},
    "config/3_distribution.yaml": {"directly_traceable", "indirectly_traceable"},
    "config/features/full_po3_mtf.yaml": {"directly_traceable", "indirectly_traceable"},
}
INACTIVE_CONFIG_EVIDENCE = {
    "config/1_baseline.yaml": (
        "The file documents that baseline is not auto-merged; training/configuration/cli.py "
        "uses baseline as the default branch and leaves config/base/default.yaml untouched."
    )
}
CRITICAL_SYMBOLS = (
    ("configuration/loader.py", "load_config"),
    ("training/configuration/cli.py", "load_merged_config"),
    ("features/pipeline/builder.py", "build_features"),
    ("simulation/environment/env.py", "TradingEnv.step"),
)
CRITICAL_KEY_PREFIXES = (
    "env.",
    "reward",
    "features.",
    "strategy.",
    "algo.",
    "risk",
    "guardrail",
    "action",
    "tp",
    "sl",
    "seed",
    "timesteps",
    "n_envs",
    "train",
    "test",
)
CRITICAL_AREAS = (
    "configuration/",
    "training/configuration/",
    "training/runner/",
    "features/pipeline/",
    "simulation/environment/",
    "learning/rl/",
    "experiments/campaigns/",
    "evaluation/rollout/",
    "evaluation/reanalysis/",
)
GENERIC_CONTAINER_METHODS = {
    "get",
    "items",
    "keys",
    "values",
    "update",
    "setdefault",
    "pop",
    "append",
    "extend",
    "insert",
    "index",
    "count",
    "copy",
    "clear",
    "remove",
    "sort",
    "reverse",
    "read",
    "write",
    "close",
    "flush",
    "fileno",
}


def _run(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def tracked_files(suffixes: tuple[str, ...]) -> list[str]:
    items = _run("git", "ls-files").splitlines()
    return [p for p in items if p.endswith(suffixes)]


def git_state() -> dict[str, Any]:
    return {
        "git_commit": _run("git", "rev-parse", "HEAD"),
        "dirty_tree": _run("git", "status", "--short").splitlines(),
    }


def graph_status() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(_run(str(ROOT / "tools" / "crgraph"), "status", "--json")),
    )


def connect() -> sqlite3.Connection:
    if not DB.exists():
        raise FileNotFoundError(f"graph database not found: {DB}")
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


def graph_schema_version(c: sqlite3.Connection) -> dict[str, str]:
    return {str(r["key"]): str(r["value"]) for r in c.execute("select key,value from metadata")}


def _rel_path(value: str | Path) -> str:
    p = Path(value)
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except (ValueError, OSError):
        return p.as_posix()


def graph_file_set(c: sqlite3.Connection) -> set[str]:
    return {_rel_path(str(row[0])) for row in c.execute("select distinct file_path from nodes")}


def _parse_file(path: Path) -> tuple[str, ast.Module] | None:
    try:
        source = path.read_text(encoding="utf-8")
        return source, ast.parse(source)
    except (SyntaxError, UnicodeDecodeError, OSError):
        return None


def _module_imports(path: Path) -> dict[str, str]:
    parsed = _parse_file(path)
    if parsed is None:
        return {}
    _, tree = parsed
    out: dict[str, str] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                out[a.asname or a.name.split(".")[0]] = a.name
        elif isinstance(n, ast.ImportFrom):
            rel = _rel_path(path)
            module_path = rel.removesuffix(".py")
            package = (
                module_path.rsplit("/", 1)[0]
                if not module_path.endswith("/__init__")
                else module_path.removesuffix("/__init__")
            )
            package_parts = package.split("/") if package else []
            if n.level:
                keep = max(0, len(package_parts) - (n.level - 1))
                prefix = package_parts[:keep]
                if n.module:
                    prefix.extend(n.module.split("."))
                base_module = ".".join(prefix)
            else:
                base_module = n.module or ""
            for a in n.names:
                target = f"{base_module}.{a.name}" if base_module else a.name
                out[a.asname or a.name] = target
    return out


def _reexport_map() -> dict[str, str]:
    """Map package-level public names to their statically imported origins."""
    result: dict[str, str] = {}
    for rel in tracked_files(("/__init__.py",)):
        parsed = _parse_file(ROOT / rel)
        if parsed is None:
            continue
        _, tree = parsed
        package = rel.removesuffix("/__init__.py").replace("/", ".")
        imports = _module_imports(ROOT / rel)
        for public_name, origin in imports.items():
            result[f"{package}.{public_name}"] = origin
    return result


def _resolve_reexport(target: str, reexports: dict[str, str]) -> str:
    seen: set[str] = set()
    while target in reexports and target not in seen:
        seen.add(target)
        target = reexports[target]
    return target


def _stdlib_roots() -> set[str]:
    roots = set(getattr(sys, "stdlib_module_names", ()))
    roots.update(
        {"pathlib", "typing", "dataclasses", "collections", "datetime", "json", "os", "sys"}
    )
    return roots


def _repo_top_modules() -> set[str]:
    return {p.split("/", 1)[0].removesuffix(".py") for p in tracked_files((".py",))}


def _call_name(n: ast.Call) -> str:
    parts: list[str] = []
    cur: ast.AST = n.func
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))


def _string_parts(expr: ast.AST, assignments: dict[str, ast.AST]) -> list[str]:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return [expr.value]
    if isinstance(expr, ast.Name) and expr.id in assignments:
        return _string_parts(assignments[expr.id], assignments)
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Div):
        return _string_parts(expr.left, assignments) + _string_parts(expr.right, assignments)
    if isinstance(expr, ast.Call):
        name = _call_name(expr)
        if name.endswith("Path") and expr.args:
            return _string_parts(expr.args[0], assignments)
        if name.endswith("read_text") or name.endswith("open"):
            if isinstance(expr.func, ast.Attribute):
                return _string_parts(expr.func.value, assignments)
        return []
    return []


def _config_path(expr: ast.AST, assignments: dict[str, ast.AST]) -> str | None:
    parts = [p for p in _string_parts(expr, assignments) if p not in {".", ".."}]
    if not parts:
        return None
    candidate = "/".join(p.strip("/\\") for p in parts if p.strip("/\\"))
    marker_positions = [
        candidate.find(x) for x in ("config/", "pyproject.toml") if candidate.find(x) >= 0
    ]
    if marker_positions:
        candidate = candidate[min(marker_positions) :]
    if candidate.endswith((".yaml", ".yml", ".toml")):
        return candidate
    return None


def _enclosing_symbol(tree: ast.AST, target: ast.AST) -> str:
    best = "<module>"
    best_span = 10**12
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            end = getattr(n, "end_lineno", n.lineno)
            if n.lineno <= getattr(target, "lineno", 0) <= end:
                span = end - n.lineno
                if span < best_span:
                    best, best_span = n.name, span
    return best


def _enclosing_class(tree: ast.AST, line: int) -> str | None:
    best: str | None = None
    best_span = 10**12
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef):
            end = getattr(n, "end_lineno", n.lineno)
            if n.lineno <= line <= end and end - n.lineno < best_span:
                best = n.name
                best_span = end - n.lineno
    return best


def _class_ranges(tree: ast.AST) -> list[tuple[int, int, str]]:
    ranges: list[tuple[int, int, str]] = []
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef):
            ranges.append((n.lineno, getattr(n, "end_lineno", n.lineno), n.name))
    ranges.sort(key=lambda x: (x[1] - x[0], x[0]))
    return ranges


def _class_at_line(ranges: list[tuple[int, int, str]], line: int) -> str | None:
    return next((name for start, end, name in ranges if start <= line <= end), None)


def _module_assignments(tree: ast.Module) -> dict[str, ast.AST]:
    assignments: dict[str, ast.AST] = {}
    for n in tree.body:
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            if n.value is None:
                continue
            for t in targets:
                if isinstance(t, ast.Name):
                    assignments[t.id] = n.value
    return assignments


def _cfg_accesses(tree: ast.AST) -> set[str]:
    found: set[str] = set()
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute):
            parent = parents.get(n)
            if isinstance(parent, ast.Attribute) and parent.value is n:
                continue
            parts: list[str] = []
            cur: ast.AST = n
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name) and cur.id.lower() in {"cfg", "config", "conf"}:
                found.add(".".join(reversed(parts)))
        elif (
            isinstance(n, ast.Subscript)
            and isinstance(n.value, ast.Name)
            and n.value.id.lower() in {"cfg", "config", "conf"}
        ):
            if isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str):
                found.add(n.slice.value)
    return found


def _function_default_config_paths(
    tree: ast.Module, assignments: dict[str, ast.AST]
) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for n in ast.walk(tree):
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        positional = list(n.args.posonlyargs) + list(n.args.args)
        defaults = [None] * (len(positional) - len(n.args.defaults)) + list(n.args.defaults)
        pairs = [(a.arg, d) for a, d in zip(positional, defaults) if d is not None]
        pairs.extend(
            (a.arg, d) for a, d in zip(n.args.kwonlyargs, n.args.kw_defaults) if d is not None
        )
        for param, default in pairs:
            path = _config_path(default, assignments)
            if path:
                result[(n.name, param)] = path
    return result


def _default_param_ref(expr: ast.AST | None) -> str | None:
    """Return a parameter name wrapped by harmless path/open helpers."""
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Call) and expr.args:
        name = _call_name(expr)
        if name.endswith("Path") or name.endswith("open"):
            return _default_param_ref(expr.args[0])
    return None


def _registry_relations(
    rel: str, tree: ast.Module, assignments: dict[str, ast.AST]
) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    registries: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for n in tree.body:
        if not isinstance(n, (ast.Assign, ast.AnnAssign)) or n.value is None:
            continue
        targets = n.targets if isinstance(n, ast.Assign) else [n.target]
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        if not names:
            continue
        mapping_name = names[0]
        direct = _config_path(n.value, assignments)
        if direct:
            relations.append(
                {
                    "config_path": direct,
                    "relation_type": "CONFIG_CONSTANT",
                    "source_file": rel,
                    "source_symbol": "<module>",
                    "line": n.lineno,
                    "selector_or_key": mapping_name,
                    "consumer_or_next_hop": mapping_name,
                    "confidence": "exact",
                    "resolution_reason": "module-level config path constant",
                }
            )
        if isinstance(n.value, ast.Dict):
            for key, value in zip(n.value.keys, n.value.values):
                if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                    continue
                path = _config_path(value, assignments)
                if path:
                    registries[mapping_name].append((str(key.value), path))
                    relations.append(
                        {
                            "config_path": path,
                            "relation_type": "REGISTRY_ENTRY",
                            "source_file": rel,
                            "source_symbol": "<module>",
                            "line": getattr(value, "lineno", n.lineno),
                            "selector_or_key": str(key.value),
                            "consumer_or_next_hop": mapping_name,
                            "confidence": "exact",
                            "resolution_reason": f"literal mapping {mapping_name}[{key.value!r}]",
                        }
                    )
    for node in ast.walk(tree):
        registry_name: str | None = None
        selector_key = "<dynamic>"
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            registry_name = node.value.id
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                selector_key = node.slice.value
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
        ):
            registry_name = node.func.value.id
            if (
                node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                selector_key = node.args[0].value
        if registry_name is None or registry_name not in registries:
            continue
        for registry_key, config_path in registries[registry_name]:
            if selector_key != "<dynamic>" and selector_key != registry_key:
                continue
            relations.append(
                {
                    "config_path": config_path,
                    "relation_type": "REGISTRY_SELECTION",
                    "source_file": rel,
                    "source_symbol": _enclosing_symbol(tree, node),
                    "line": getattr(node, "lineno", 0),
                    "selector_or_key": selector_key,
                    "consumer_or_next_hop": registry_name,
                    "confidence": "high" if selector_key == "<dynamic>" else "exact",
                    "resolution_reason": f"registry {registry_name} is read at this site",
                }
            )
    return relations


def config_dependencies() -> dict[str, Any]:
    consumers: list[dict[str, Any]] = []
    dynamic_loads: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    key_consumers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    loader_suffixes = (
        "OmegaConf.load",
        "load_config",
        "yaml.safe_load",
        "tomllib.load",
        "tomllib.loads",
    )

    for rel in tracked_files((".py",)):
        parsed = _parse_file(ROOT / rel)
        if parsed is None:
            continue
        source_text, tree = parsed
        assignments = _module_assignments(tree)
        defaults = _function_default_config_paths(tree, assignments)
        relations.extend(_registry_relations(rel, tree, assignments))

        for key in sorted(_cfg_accesses(tree)):
            if any(key.startswith(p) for p in CRITICAL_KEY_PREFIXES):
                key_consumers[key].append({"file": rel})

        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            name = _call_name(n)
            is_loader = any(name.endswith(x) for x in loader_suffixes)
            is_read = name.endswith("read_text")
            if not (is_loader or is_read):
                continue
            expr: ast.AST | None = n.args[0] if n.args else None
            if is_read and isinstance(n.func, ast.Attribute):
                expr = n.func.value
            symbol = _enclosing_symbol(tree, n)
            resolved = _config_path(expr, assignments) if expr is not None else None
            relation_type = "DIRECT_LOAD"
            default_param = _default_param_ref(expr)
            if resolved is None and default_param:
                resolved = defaults.get((symbol, default_param))
                if resolved:
                    relation_type = "DEFAULT_PARAMETER_LOAD"
            record = {
                "file": rel,
                "line": getattr(n, "lineno", 0),
                "symbol": symbol,
                "loader": name,
                "resolved_config": resolved,
                "relation_type": relation_type if resolved else "DYNAMIC_LOAD",
                "source": ast.get_source_segment(source_text, n) or name,
            }
            if resolved:
                consumers.append(record)
                relations.append(
                    {
                        "config_path": resolved,
                        "relation_type": relation_type,
                        "source_file": rel,
                        "source_symbol": symbol,
                        "line": getattr(n, "lineno", 0),
                        "selector_or_key": default_param
                        if relation_type == "DEFAULT_PARAMETER_LOAD"
                        else None,
                        "consumer_or_next_hop": name,
                        "confidence": "high"
                        if relation_type == "DEFAULT_PARAMETER_LOAD"
                        else "exact",
                        "resolution_reason": "loader argument resolved statically",
                    }
                )
            elif is_loader:
                dynamic_loads.append(record)

    config_files = sorted(_run("git", "ls-files", "*.yaml", "*.yml", "*.toml").splitlines())
    reachability: dict[str, dict[str, Any]] = {}
    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for relation in relations:
        by_config[str(relation["config_path"])].append(relation)
    for cfg in config_files:
        evidence = by_config.get(cfg, [])
        direct = [x for x in evidence if x["relation_type"] == "DIRECT_LOAD"]
        indirect = [x for x in evidence if x["relation_type"] != "DIRECT_LOAD"]
        if cfg in INACTIVE_CONFIG_EVIDENCE:
            state = "tracked_but_inactive"
            note = INACTIVE_CONFIG_EVIDENCE[cfg]
        elif direct:
            state = "directly_traceable"
            note = "direct static loader relationship found"
        elif indirect:
            state = "indirectly_traceable"
            note = "static registry/default-parameter relationship found"
        else:
            state = "unresolved"
            note = "no static direct or indirect relationship found"
        reachability[cfg] = {"state": state, "evidence": evidence, "note": note}
    if "pyproject.toml" in reachability and reachability["pyproject.toml"]["state"] == "unresolved":
        reachability["pyproject.toml"] = {
            "state": "indirectly_traceable",
            "evidence": [
                {
                    "config_path": "pyproject.toml",
                    "relation_type": "TOOLING_CONFIG",
                    "source_file": "pyproject.toml",
                    "source_symbol": "<tooling>",
                    "line": 1,
                    "selector_or_key": None,
                    "consumer_or_next_hop": "pytest/ruff/mypy/build tooling",
                    "confidence": "exact",
                    "resolution_reason": "project tooling configuration",
                }
            ],
            "note": "project tooling configuration",
        }

    return {
        "config_files": config_files,
        "consumers": consumers,
        "dynamic_loads": dynamic_loads,
        "relations": relations,
        "reachability": reachability,
        "critical_key_consumers": dict(sorted(key_consumers.items())),
    }


@dataclass
class CallAudit:
    source_file: str
    source_symbol: str
    call_target: str
    line: int
    category: str
    candidate_targets: list[str]
    resolution_reason: str
    ambiguity_class: str | None = None
    critical_path: bool = False


def _node_rows(c: sqlite3.Connection) -> dict[str, list[dict[str, str]]]:
    by_name: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in c.execute("select name,qualified_name,file_path,parent_name,kind from nodes"):
        by_name[str(r["name"])].append(
            {
                "name": str(r["name"]),
                "qualified_name": str(r["qualified_name"]),
                "file_path": _rel_path(str(r["file_path"])),
                "parent_name": str(r["parent_name"] or ""),
                "kind": str(r["kind"]),
            }
        )
    return by_name


def _module_file_candidates(module: str) -> tuple[str, str]:
    path = module.replace(".", "/")
    return f"{path}.py", f"{path}/__init__.py"


def _candidate_matches_import(candidate: dict[str, str], imported_target: str, short: str) -> bool:
    parts = imported_target.split(".")
    if not parts or parts[-1] != short:
        return False
    module = ".".join(parts[:-1])
    return candidate["file_path"] in _module_file_candidates(module)


def _inheritance_map(c: sqlite3.Connection) -> dict[tuple[str, str], set[str]]:
    result: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in c.execute(
        "select source_qualified,target_qualified,file_path from edges where kind='INHERITS'"
    ):
        source_class = str(row["source_qualified"]).rsplit("::", 1)[-1].rsplit(".", 1)[-1]
        target_class = str(row["target_qualified"]).rsplit("::", 1)[-1].rsplit(".", 1)[-1]
        result[(_rel_path(str(row["file_path"])), source_class)].add(target_class)
    return result


def _ambiguity_class(
    short: str, receiver: str, receiver_binding: str, candidates: list[dict[str, str]]
) -> str:
    if short in GENERIC_CONTAINER_METHODS and receiver_binding in {"unknown", "builtin"}:
        return "receiver_type_unknown"
    if receiver in {"self", "cls"}:
        return "instance_method" if receiver == "self" else "class_method"
    parents = {x["parent_name"] for x in candidates if x["parent_name"]}
    if len(parents) > 1:
        return "same_short_name_across_modules"
    return "receiver_type_unknown" if receiver else "same_short_name_across_modules"


def classify_calls(c: sqlite3.Connection) -> tuple[list[CallAudit], dict[str, Any], dict[str, Any]]:
    nodes_by_name = _node_rows(c)
    imports_cache: dict[str, dict[str, str]] = {}
    class_ranges_cache: dict[str, list[tuple[int, int, str]]] = {}
    inheritance = _inheritance_map(c)
    reexports = _reexport_map()
    stdlib = _stdlib_roots()
    repo_roots = _repo_top_modules()
    builtin_names = set(dir(builtins))
    audits: list[CallAudit] = []

    rows = c.execute(
        "select source_qualified,target_qualified,file_path,line,target_resolution,extra "
        "from edges where kind='CALLS'"
    )
    for r in rows:
        source = str(r["source_qualified"])
        target = str(r["target_qualified"])
        source_file = _rel_path(str(r["file_path"]))
        line = int(r["line"] or 0)
        critical = any(source_file.startswith(prefix) for prefix in CRITICAL_AREAS)
        short = target.rsplit("::", 1)[-1].rsplit(".", 1)[-1]
        candidates = list(nodes_by_name.get(short, ()))
        try:
            extra = json.loads(r["extra"] or "{}")
        except json.JSONDecodeError:
            extra = {}
        receiver = str(extra.get("receiver") or "")
        receiver_binding = str(extra.get("receiver_binding") or "")
        receiver_class = str(extra.get("receiver_class") or "")
        ambiguity: str | None = None

        if r["target_resolution"] == "direct":
            category, selected, reason = (
                "internal_exact",
                [target],
                "graph target_resolution=direct",
            )
        else:
            if source_file not in imports_cache:
                imports_cache[source_file] = _module_imports(ROOT / source_file)
            imports = imports_cache[source_file]
            receiver_module = imports.get(receiver, receiver)
            receiver_root = receiver_module.split(".", 1)[0] if receiver_module else ""
            imported_target = _resolve_reexport(imports.get(short, ""), reexports)
            imported_root = imported_target.split(".", 1)[0] if imported_target else ""
            same_file = [x for x in candidates if x["file_path"] == source_file]
            imported_candidates = [
                x
                for x in candidates
                if imported_target and _candidate_matches_import(x, imported_target, short)
            ]

            if source_file not in class_ranges_cache:
                parsed = _parse_file(ROOT / source_file)
                class_ranges_cache[source_file] = _class_ranges(parsed[1]) if parsed else []
            enclosing_class = _class_at_line(class_ranges_cache[source_file], line)
            self_candidates = (
                [x for x in candidates if x["parent_name"] == enclosing_class]
                if enclosing_class
                else []
            )
            inherited_candidates: list[dict[str, str]] = []
            if enclosing_class:
                bases = inheritance.get((source_file, enclosing_class), set())
                inherited_candidates = [x for x in candidates if x["parent_name"] in bases]

            if short in builtin_names and not candidates:
                category, selected, reason = "python_builtin", [], "target is a Python builtin"
            elif (
                short in GENERIC_CONTAINER_METHODS
                and (
                    receiver_binding in {"unknown", "builtin"}
                    or receiver_class
                    in {"Any", "dict", "Dict", "Mapping", "MutableMapping", "DictConfig", "Path"}
                )
                and not self_candidates
            ):
                category, selected, reason = (
                    "dynamic_or_indeterminate",
                    [],
                    "generic container/file method with unknown or builtin receiver",
                )
                ambiguity = "receiver_type_unknown"
            elif (
                receiver_root
                and receiver_root not in repo_roots
                and (receiver_root in stdlib or receiver in imports)
            ):
                category, selected, reason = (
                    "external_library",
                    [],
                    f"receiver resolves to imported/stdlib module {receiver_module!r}",
                )
            elif imported_root and imported_root not in repo_roots:
                category, selected, reason = (
                    "external_library",
                    [],
                    f"target imported from {imported_target!r}",
                )
            elif len(imported_candidates) == 1:
                category = "internal_exact"
                selected = [imported_candidates[0]["qualified_name"]]
                reason = f"explicit import resolves uniquely to {imported_target!r}"
            elif receiver in {"self", "cls"} and len(self_candidates) == 1:
                category = "internal_exact"
                selected = [self_candidates[0]["qualified_name"]]
                reason = f"{receiver} call resolves uniquely in enclosing class {enclosing_class}"
            elif (
                receiver in {"self", "cls"}
                and not self_candidates
                and len(inherited_candidates) == 1
            ):
                category = "internal_exact"
                selected = [inherited_candidates[0]["qualified_name"]]
                reason = (
                    f"{receiver} call resolves uniquely in direct base class of {enclosing_class}"
                )
            elif not receiver and len(same_file) == 1:
                category = "internal_exact"
                selected = [same_file[0]["qualified_name"]]
                reason = "one same-module candidate matches unqualified call"
            elif len(candidates) == 1:
                category, selected, reason = (
                    "internal_unresolved",
                    [candidates[0]["qualified_name"]],
                    "one indexed repository symbol has the same short name but receiver provenance is insufficient",
                )
            elif len(candidates) > 1:
                category = "internal_ambiguous"
                selected = [x["qualified_name"] for x in candidates[:20]]
                ambiguity = _ambiguity_class(short, receiver, receiver_binding, candidates)
                reason = f"{len(candidates)} indexed repository symbols share the short name"
            elif receiver_root in repo_roots:
                category, selected, reason = (
                    "internal_unresolved",
                    [],
                    f"receiver root {receiver_root!r} is repository-owned",
                )
            else:
                category, selected, reason = (
                    "dynamic_or_indeterminate",
                    [],
                    "no reliable static repository or external target evidence",
                )

        audits.append(
            CallAudit(
                source_file=source_file,
                source_symbol=source,
                call_target=target,
                line=line,
                category=category,
                candidate_targets=selected,
                resolution_reason=reason,
                ambiguity_class=ambiguity,
                critical_path=critical,
            )
        )

    counts = Counter(a.category for a in audits)
    denom = counts["internal_exact"] + counts["internal_ambiguous"] + counts["internal_unresolved"]
    critical_counts = Counter(a.category for a in audits if a.critical_path)
    critical_denom = (
        critical_counts["internal_exact"]
        + critical_counts["internal_ambiguous"]
        + critical_counts["internal_unresolved"]
    )
    metrics = {
        "counts": dict(sorted(counts.items())),
        "internal_resolution_rate": counts["internal_exact"] / denom if denom else 1.0,
        "internal_unresolved_rate": counts["internal_unresolved"] / denom if denom else 0.0,
        "internal_ambiguous_rate": counts["internal_ambiguous"] / denom if denom else 0.0,
        "raw_exact_calls_rate": counts["internal_exact"] / len(audits) if audits else 1.0,
        "total_calls": len(audits),
        "internal_call_candidates": denom,
        "critical_counts": dict(sorted(critical_counts.items())),
        "critical_internal_candidates": critical_denom,
        "critical_internal_resolution_rate": critical_counts["internal_exact"] / critical_denom
        if critical_denom
        else 1.0,
    }
    ambiguous = [a for a in audits if a.category == "internal_ambiguous"]
    by_class = Counter(a.ambiguity_class or "other" for a in ambiguous)
    by_area = Counter(
        next(
            (p.rstrip("/") for p in CRITICAL_AREAS if a.source_file.startswith(p)),
            a.source_file.split("/", 1)[0],
        )
        for a in ambiguous
    )
    top_targets = Counter(a.call_target for a in ambiguous)
    breakdown = {
        "count": len(ambiguous),
        "by_root_cause": dict(by_class.most_common()),
        "by_area": dict(by_area.most_common()),
        "top_target_names": top_targets.most_common(50),
        "critical_path_count": sum(1 for a in ambiguous if a.critical_path),
        "samples": [asdict(a) for a in ambiguous[:100]],
    }
    return audits, metrics, breakdown


def verify_critical(c: sqlite3.Connection, config: dict[str, Any]) -> dict[str, Any]:
    graph_files = graph_file_set(c)
    tracked_py = set(tracked_files((".py",)))
    tracked_sh = set(tracked_files((".sh",)))
    indexed_py = tracked_py & graph_files
    indexed_sh = tracked_sh & graph_files
    config_paths = set(config["config_files"])

    symbol_checks = []
    for rel, symbol in CRITICAL_SYMBOLS:
        rows = list(
            c.execute(
                "select qualified_name from nodes where file_path like ? and (name=? or qualified_name like ?)",
                (f"%/{rel}", symbol.rsplit(".", 1)[-1], f"%::{symbol}"),
            )
        )
        symbol_checks.append(
            {
                "file": rel,
                "symbol": symbol,
                "present": bool(rows),
                "matches": [str(r[0]) for r in rows[:10]],
            }
        )

    cfg_checks = []
    for cfg, allowed in CRITICAL_CONFIG_POLICY.items():
        reach = config["reachability"].get(
            cfg, {"state": "unresolved", "evidence": [], "note": "missing reachability record"}
        )
        actual = str(reach["state"])
        result = cfg in config_paths and actual in allowed
        cfg_checks.append(
            {
                "config": cfg,
                "tracked": cfg in config_paths,
                "state": actual,
                "expected_states": sorted(allowed),
                "result": result,
                "evidence": reach.get("evidence", []),
                "note": reach.get("note", ""),
            }
        )

    ok = (
        indexed_py == tracked_py
        and indexed_sh == tracked_sh
        and all(x["present"] for x in symbol_checks)
        and all(x["result"] for x in cfg_checks)
    )
    return {
        "ok": ok,
        "source_coverage": {
            "tracked_python": len(tracked_py),
            "indexed_python": len(indexed_py),
            "missing_python": sorted(tracked_py - graph_files),
            "tracked_shell": len(tracked_sh),
            "indexed_shell": len(indexed_sh),
            "missing_shell": sorted(tracked_sh - graph_files),
        },
        "critical_symbols": symbol_checks,
        "critical_configs": cfg_checks,
    }


def write_json(name: str, value: Any) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_ambiguity_markdown(breakdown: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "ambiguous_call_breakdown.md"
    lines = [
        "# Ambiguous CALLS Breakdown",
        "",
        f"Total ambiguous internal calls: {breakdown['count']}",
        "",
        "## Root causes",
        "",
    ]
    for key, count in breakdown["by_root_cause"].items():
        lines.append(f"- `{key}`: {count}")
    lines += ["", "## Areas", ""]
    for key, count in breakdown["by_area"].items():
        lines.append(f"- `{key}`: {count}")
    lines += ["", f"Critical-path ambiguous calls: {breakdown['critical_path_count']}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_markdown(name: str, report: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    m = report["call_metrics"]
    v = report["verification"]
    lines = [
        "# Code Graph Quality Report",
        "",
        f"- Git commit: `{report['git_commit']}`",
        f"- Graph tool: `{report['graph_tool_version']}`",
        f"- Graph nodes/edges/files: {report['graph_status']['nodes']} / {report['graph_status']['edges']} / {report['graph_status']['files']}",
        f"- Internal call resolution: {m['internal_resolution_rate']:.2%} ({m['counts'].get('internal_exact', 0)} exact / {m['internal_call_candidates']} internal candidates)",
        f"- Critical-path internal resolution: {m['critical_internal_resolution_rate']:.2%} ({m['critical_counts'].get('internal_exact', 0)} exact / {m['critical_internal_candidates']} candidates)",
        f"- Raw exact/all CALLS rate: {m['raw_exact_calls_rate']:.2%}",
        f"- Python indexed: {v['source_coverage']['indexed_python']}/{v['source_coverage']['tracked_python']}",
        f"- Shell indexed: {v['source_coverage']['indexed_shell']}/{v['source_coverage']['tracked_shell']}",
        f"- Static config consumers: {len(report['config_dependencies']['consumers'])}",
        f"- Indirect config relations: {sum(1 for x in report['config_dependencies']['relations'] if x['relation_type'] != 'DIRECT_LOAD')}",
        f"- Dynamic config loads requiring source review: {len(report['config_dependencies']['dynamic_loads'])}",
        "",
        "## CALLS classification",
        "",
    ]
    for k, count in sorted(m["counts"].items()):
        lines.append(f"- `{k}`: {count}")
    lines += ["", "## Critical configuration checks", ""]
    for x in v["critical_configs"]:
        lines.append(
            f"- `{x['config']}`: state={x['state']}, result={x['result']}, expected={','.join(x['expected_states'])}"
        )
    lines += ["", "## Critical symbol checks", ""]
    for x in v["critical_symbols"]:
        lines.append(f"- `{x['file']}::{x['symbol']}`: present={x['present']}")
    lines += [
        "",
        "> TESTED_BY resolution only means the recorded relation resolves to an indexed node; it is not a coverage measurement.",
        "",
    ]
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def build_report(label: str) -> dict[str, Any]:
    with connect() as c:
        calls, metrics, breakdown = classify_calls(c)
        config = config_dependencies()
        verification = verify_critical(c, config)
        status = graph_status()
        state = git_state()
        try:
            version = _run("code-review-graph", "--version")
        except Exception:
            version = "unknown"
        edge_counts = {
            str(r[0]): int(r[1]) for r in c.execute("select kind,count(*) from edges group by kind")
        }
        report = {
            **state,
            "timestamp": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
            "label": label,
            "graph_tool_version": version,
            "graph_status": status,
            "graph_metadata": graph_schema_version(c),
            "edge_counts": edge_counts,
            "call_metrics": metrics,
            "ambiguity_breakdown": breakdown,
            "config_dependencies": config,
            "verification": verification,
        }
        write_json(f"{label}.json", report)
        write_markdown(f"{label}.md", report)
        write_json(
            "unresolved_internal_calls.json",
            [
                asdict(x)
                for x in calls
                if x.category in {"internal_unresolved", "internal_ambiguous"}
            ],
        )
        write_json("config_dependencies.json", config)
        write_json("ambiguous_call_breakdown.json", breakdown)
        write_ambiguity_markdown(breakdown)
        return report


def compare_reports(before: Path, after: Path) -> dict[str, Any]:
    b = json.loads(before.read_text(encoding="utf-8"))
    a = json.loads(after.read_text(encoding="utf-8"))
    bm, am = b["call_metrics"], a["call_metrics"]
    result = {
        "before_commit": b["git_commit"],
        "after_commit": a["git_commit"],
        "internal_resolution_before": bm["internal_resolution_rate"],
        "internal_resolution_after": am["internal_resolution_rate"],
        "internal_resolution_delta": am["internal_resolution_rate"]
        - bm["internal_resolution_rate"],
        "internal_ambiguous_before": bm["counts"].get("internal_ambiguous", 0),
        "internal_ambiguous_after": am["counts"].get("internal_ambiguous", 0),
        "critical_resolution_before": bm.get("critical_internal_resolution_rate"),
        "critical_resolution_after": am.get("critical_internal_resolution_rate"),
        "source_coverage_ok": a["verification"]["source_coverage"]["missing_python"] == []
        and a["verification"]["source_coverage"]["missing_shell"] == [],
        "critical_verification_ok": a["verification"]["ok"],
    }
    write_json("comparison.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_report = sub.add_parser("report")
    p_report.add_argument("--label", default="final")
    p_check = sub.add_parser("check")
    p_check.add_argument("--baseline", type=Path)
    p_check.add_argument("--max-resolution-regression", type=float, default=0.01)
    p_check.add_argument("--min-resolution", type=float)
    p_check.add_argument("--min-critical-resolution", type=float)
    p_compare = sub.add_parser("compare")
    p_compare.add_argument("before", type=Path)
    p_compare.add_argument("after", type=Path)
    args = parser.parse_args()

    if args.cmd == "report":
        report = build_report(args.label)
        print(
            json.dumps(
                {
                    "report": str(OUT / f"{args.label}.json"),
                    "metrics": report["call_metrics"],
                    "verification_ok": report["verification"]["ok"],
                },
                indent=2,
            )
        )
        return 0
    if args.cmd == "compare":
        print(json.dumps(compare_reports(args.before, args.after), indent=2))
        return 0

    report = build_report("check")
    ok = bool(report["verification"]["ok"])
    if not ok:
        for item in report["verification"]["critical_configs"]:
            if not item["result"]:
                print(
                    f"FAIL {item['config']} state={item['state']} expected={{{','.join(item['expected_states'])}}} note={item['note']}",
                    file=sys.stderr,
                )
    current = report["call_metrics"]["internal_resolution_rate"]
    current_critical = report["call_metrics"].get("critical_internal_resolution_rate")
    if args.min_resolution is not None and current < args.min_resolution:
        print(
            f"internal resolution below CI floor: {current:.4f} < {args.min_resolution:.4f}",
            file=sys.stderr,
        )
        ok = False
    if (
        args.min_critical_resolution is not None
        and current_critical is not None
        and current_critical < args.min_critical_resolution
    ):
        print(
            "critical internal resolution below CI floor: "
            f"{current_critical:.4f} < {args.min_critical_resolution:.4f}",
            file=sys.stderr,
        )
        ok = False
    if args.baseline and args.baseline.exists():
        base = json.loads(args.baseline.read_text(encoding="utf-8"))
        prior = base["call_metrics"]["internal_resolution_rate"]
        if current + args.max_resolution_regression < prior:
            print(f"internal resolution regression: {prior:.4f} -> {current:.4f}", file=sys.stderr)
            ok = False
        prior_critical = base["call_metrics"].get("critical_internal_resolution_rate")
        if (
            prior_critical is not None
            and current_critical is not None
            and current_critical + args.max_resolution_regression < prior_critical
        ):
            print(
                f"critical internal resolution regression: {prior_critical:.4f} -> {current_critical:.4f}",
                file=sys.stderr,
            )
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
