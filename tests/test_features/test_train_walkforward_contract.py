"""Source contract for walk-forward feature inputs, without launching training.

The feature builder's numerical fold-isolation tests live in
test_feature_cache_and_fit_scope.py; this test covers the training call site.
"""

from __future__ import annotations

import ast
import os
import unittest
from pathlib import Path

TRAIN_MODULE = Path(
    os.environ.get(
        "AALTO_WF_TRAIN_SOURCE",
        str(Path(__file__).resolve().parents[2] / "quant_rl/train/train_rl.py"),
    )
)


class TestWalkForwardSourceContract(unittest.TestCase):
    def test_walk_forward_uses_retained_train_scoped_sources(self) -> None:
        tree = ast.parse(TRAIN_MODULE.read_text(encoding="utf-8"))
        fold_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "build_walkforward_fold_features"
        ]
        self.assertEqual(len(fold_calls), 1, "Expected one training-driver fold call")
        call = fold_calls[0]
        self.assertGreaterEqual(len(call.args), 2)
        assert isinstance(call.args[0], ast.Name)
        assert isinstance(call.args[1], ast.Name)
        self.assertEqual(
            (call.args[0].id, call.args[1].id),
            ("train_bars", "train_sec"),
            "Fold features must not use deleted full-history or OOS data frames",
        )

    def test_train_scoped_inputs_survive_parent_cleanup(self) -> None:
        tree = ast.parse(TRAIN_MODULE.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "build_walkforward_fold_features"
        ]
        self.assertEqual(len(calls), 1)
        call = calls[0]
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent
        deleted_before: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Delete) or node.lineno >= call.lineno:
                continue
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                deleted_before.add(target.id)
                if target.id == "train_bars":
                    ancestors: list[ast.AST] = []
                    cursor: ast.AST = node
                    while cursor in parents:
                        cursor = parents[cursor]
                        ancestors.append(cursor)
                    self.assertTrue(
                        any(
                            isinstance(parent, ast.If)
                            and "not args.walk_forward" in ast.unparse(parent.test)
                            for parent in ancestors
                        ),
                        "train_bars may only be deleted when walk-forward is disabled",
                    )
        self.assertNotIn("train_sec", deleted_before)
        self.assertIn("primary_m1", deleted_before)
        self.assertIn("secondary_m1", deleted_before)


if __name__ == "__main__":
    unittest.main()
