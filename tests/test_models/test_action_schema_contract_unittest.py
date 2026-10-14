"""Action layout invariants, without requiring the training environment."""

import itertools
import unittest

from quant_rl.action_schema import build_action_schema


class ActionSchemaContractTest(unittest.TestCase):
    def test_supported_layouts_have_explicit_coordinate_names(self):
        cases = (
            (
                {
                    "has_direction": False,
                    "has_sl_mode": True,
                    "has_tp_mode": False,
                    "has_multi_tp": True,
                    "has_simplex": False,
                },
                8,
            ),
            (
                {
                    "has_direction": False,
                    "has_sl_mode": True,
                    "has_tp_mode": True,
                    "has_multi_tp": True,
                    "has_simplex": False,
                },
                9,
            ),
            (
                {
                    "has_direction": False,
                    "has_sl_mode": True,
                    "has_tp_mode": True,
                    "has_multi_tp": True,
                    "has_simplex": True,
                },
                11,
            ),
            (
                {
                    "has_direction": True,
                    "has_sl_mode": True,
                    "has_tp_mode": True,
                    "has_multi_tp": True,
                    "has_simplex": True,
                },
                12,
            ),
        )
        for flags, expected_width in cases:
            with self.subTest(flags=flags):
                schema = build_action_schema(**flags)
                self.assertEqual(schema.width, expected_width)
                self.assertEqual(schema.indices["exit"], expected_width - 1)
                self.assertEqual(len(schema.names), len(set(schema.names)))
                self.assertEqual(schema.indices["target"], schema.indices["tp1_sel"])
                self.assertEqual(
                    schema.names[0], "direction" if flags["has_direction"] else "intensity"
                )

    def test_schema_collision_by_width_preserves_independent_semantics(self):
        width_to_layouts: dict[int, set[tuple[str, ...]]] = {}
        for direction, sl, tp, multi, simplex in itertools.product((False, True), repeat=5):
            if simplex and not multi:
                with self.assertRaises(ValueError):
                    build_action_schema(
                        has_direction=direction,
                        has_sl_mode=sl,
                        has_tp_mode=tp,
                        has_multi_tp=multi,
                        has_simplex=simplex,
                    )
                continue
            schema = build_action_schema(
                has_direction=direction,
                has_sl_mode=sl,
                has_tp_mode=tp,
                has_multi_tp=multi,
                has_simplex=simplex,
            )
            self.assertEqual(
                tuple(schema.indices[name] for name in schema.names), tuple(range(schema.width))
            )
            self.assertEqual(schema.names[-1], "exit")
            width_to_layouts.setdefault(schema.width, set()).add(schema.names)
        self.assertGreater(len(width_to_layouts[8]), 1)


if __name__ == "__main__":
    unittest.main()
