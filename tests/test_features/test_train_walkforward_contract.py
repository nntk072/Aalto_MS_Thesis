"""Source contracts for retained walk-forward feature inputs.

Numerical fold-isolation coverage lives in test_feature_cache_and_fit_scope.py.
These tests protect the training composition boundary and the dedicated
walk-forward runtime after the runner decomposition.
"""

from __future__ import annotations

import ast
import os
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_MODULE = Path(
    os.environ.get(
        "AALTO_WF_TRAIN_SOURCE",
        str(_REPO_ROOT / "training/runner/training.py"),
    )
)
WALKFORWARD_MODULE = Path(
    os.environ.get(
        "AALTO_WF_RUNTIME_SOURCE",
        str(_REPO_ROOT / "training/runner/walkforward_runtime.py"),
    )
)


def _named_calls(module: Path, name: str) -> list[ast.Call]:
    tree = ast.parse(module.read_text(encoding="utf-8"))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]


class TestWalkForwardSourceContract(unittest.TestCase):
    def test_walk_forward_runtime_uses_retained_train_scoped_sources(self) -> None:
        fold_calls = _named_calls(WALKFORWARD_MODULE, "build_walkforward_fold_features")
        self.assertEqual(len(fold_calls), 1, "Expected one walk-forward fold feature call")
        call = fold_calls[0]
        self.assertGreaterEqual(len(call.args), 2)
        assert isinstance(call.args[0], ast.Name)
        assert isinstance(call.args[1], ast.Name)
        self.assertEqual(
            (call.args[0].id, call.args[1].id),
            ("train_bars", "train_sec"),
            "Fold features must use the retained training-scope market sources",
        )

    def test_training_composition_passes_retained_sources_to_walk_forward(self) -> None:
        calls = _named_calls(TRAIN_MODULE, "run_walk_forward_validation")
        self.assertEqual(len(calls), 1, "Expected one training composition call")
        keywords = {kw.arg: kw.value for kw in calls[0].keywords if kw.arg is not None}

        for keyword, attr in (("train_bars", "train_bars"), ("train_sec", "train_sec")):
            value = keywords.get(keyword)
            self.assertIsInstance(value, ast.Attribute)
            assert isinstance(value, ast.Attribute)
            self.assertIsInstance(value.value, ast.Name)
            assert isinstance(value.value, ast.Name)
            self.assertEqual(
                (value.value.id, value.attr),
                ("prep", attr),
                f"{keyword} must remain owned by prepared training state until walk-forward",
            )


if __name__ == "__main__":
    unittest.main()
