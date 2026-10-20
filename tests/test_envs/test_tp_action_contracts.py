from __future__ import annotations

import math

import pytest

from trading.orders.selection import (
    DISABLED,
    decode_tp_fractions,
    decode_tp_selections,
    validate_tp_levels,
)


class TestSimplexDecoder:
    def test_simplex_decoder_sums_to_one(self) -> None:
        f1, f2, f3 = decode_tp_fractions(0.0, 0.0)
        assert pytest.approx(f1 + f2 + f3, rel=1e-6) == 1.0
        assert pytest.approx(f1, rel=1e-6) == 1.0 / 3.0
        assert pytest.approx(f2, rel=1e-6) == 1.0 / 3.0
        assert pytest.approx(f3, rel=1e-6) == 1.0 / 3.0

    def test_simplex_renormalize_active_levels(self) -> None:
        fracs = decode_tp_fractions(1.0, -1.0, active_slots=(True, True, True))
        assert len(fracs) == 3
        assert pytest.approx(sum(fracs), rel=1e-6) == 1.0

    def test_simplex_renormalize_when_tp1_disabled(self) -> None:
        # TP1 disabled, TP2 and TP3 active
        z1, z2 = 0.5, 1.0
        # raw unnormalized fractions:
        m = max(0.5, 1.0, 0.0)
        e1 = math.exp(0.5 - m)
        e2 = math.exp(1.0 - m)
        e3 = math.exp(-m)
        denom = e1 + e2 + e3
        raw_f2 = e2 / denom
        raw_f3 = e3 / denom
        expected_f2 = raw_f2 / (raw_f2 + raw_f3)
        expected_f3 = raw_f3 / (raw_f2 + raw_f3)

        fracs = decode_tp_fractions(z1, z2, active_slots=(False, True, True))
        assert len(fracs) == 2
        assert pytest.approx(fracs[0], rel=1e-6) == expected_f2
        assert pytest.approx(fracs[1], rel=1e-6) == expected_f3
        assert pytest.approx(sum(fracs), rel=1e-6) == 1.0

    def test_simplex_renormalize_when_tp2_disabled(self) -> None:
        # TP2 disabled, TP1 and TP3 active
        z1, z2 = 0.5, 1.0
        m = max(0.5, 1.0, 0.0)
        e1 = math.exp(0.5 - m)
        e2 = math.exp(1.0 - m)
        e3 = math.exp(-m)
        denom = e1 + e2 + e3
        raw_f1 = e1 / denom
        raw_f3 = e3 / denom
        expected_f1 = raw_f1 / (raw_f1 + raw_f3)
        expected_f3 = raw_f3 / (raw_f1 + raw_f3)

        fracs = decode_tp_fractions(z1, z2, active_slots=(True, False, True))
        assert len(fracs) == 2
        assert pytest.approx(fracs[0], rel=1e-6) == expected_f1
        assert pytest.approx(fracs[1], rel=1e-6) == expected_f3
        assert pytest.approx(sum(fracs), rel=1e-6) == 1.0

    def test_simplex_decoder_handles_large_logits(self) -> None:
        fracs = decode_tp_fractions(1000.0, 2000.0, active_slots=(True, True, True))
        assert len(fracs) == 3
        assert pytest.approx(sum(fracs), rel=1e-6) == 1.0
        assert fracs[1] > fracs[0]
        assert fracs[1] > fracs[2]

        fracs_neg = decode_tp_fractions(-1000.0, -2000.0, active_slots=(True, True, True))
        assert len(fracs_neg) == 3
        assert pytest.approx(sum(fracs_neg), rel=1e-6) == 1.0


class TestTPSelection:
    def test_tp_selection_disabled_middle_slot(self) -> None:
        # TP1 active, TP2 disabled, TP3 active
        s1, s2, s3 = decode_tp_selections(0, DISABLED, 2, menu_size=5)
        assert s1 == 0
        assert s2 is None
        assert s3 == 2

    def test_tp_selection_duplicate_resolution(self) -> None:
        # Duplicate 1, 1, 2 -> disable nearer slot (TP1)
        s1, s2, s3 = decode_tp_selections(1, 1, 2, menu_size=5)
        assert s1 is None
        assert s2 == 1
        assert s3 == 2

        # Duplicate 1, 2, 2 -> disable nearer slot (TP2)
        s1, s2, s3 = decode_tp_selections(1, 2, 2, menu_size=5)
        assert s1 == 1
        assert s2 is None
        assert s3 == 2

        # Triplicate 1, 1, 1 -> only TP3 remains
        s1, s2, s3 = decode_tp_selections(1, 1, 1, menu_size=5)
        assert s1 is None
        assert s2 is None
        assert s3 == 1

    def test_tp_selection_invalid_tp3_repair(self) -> None:
        # TP3 is DISABLED, but TP1=1 and TP2=2 are valid.
        # Since TP3 is required, promote furthest candidate (2) to TP3.
        s1, s2, s3 = decode_tp_selections(1, 2, DISABLED, menu_size=5)
        assert s1 == 1
        assert s2 is None
        assert s3 == 2

        # TP3 out of bounds clamped to menu_size - 1
        s1, s2, s3 = decode_tp_selections(0, 1, 999, menu_size=5)
        assert s1 == 0
        assert s2 == 1
        assert s3 == 4

    def test_tp_selection_boundary_values(self) -> None:
        # menu_size = 1 -> only index 0 is valid
        s1, s2, s3 = decode_tp_selections(0, 0, 0, menu_size=1)
        assert s1 is None
        assert s2 is None
        assert s3 == 0

        # Boundary continuous values: -1.0 is disabled, 0.0 is index 0, 1.0 is menu_size-1
        s1, s2, s3 = decode_tp_selections(-1.0, 0.0, 1.0, menu_size=4, continuous=True)
        assert s1 is None
        assert s2 == 0
        assert s3 == 3

    def test_tp_selection_all_disabled_falls_back_to_nearest(self) -> None:
        # All DISABLED -> activates nearest valid target as sole TP (TP3)
        s1, s2, s3 = decode_tp_selections(DISABLED, DISABLED, DISABLED, menu_size=5)
        assert s1 is None
        assert s2 is None
        assert s3 == 0

    def test_tp_selection_preserves_maximum_valid_subset(self) -> None:
        # raw={1, 3, 2}: s2(3) >= s3(2) -> disable s2 -> {1, None, 2}
        s1, s2, s3 = decode_tp_selections(1, 3, 2, menu_size=5)
        assert s1 == 1
        assert s2 is None
        assert s3 == 2

        # raw={2, 1, 3}: s1(2) >= s2(1) -> disable s1 -> {None, 1, 3}
        s1, s2, s3 = decode_tp_selections(2, 1, 3, menu_size=5)
        assert s1 is None
        assert s2 == 1
        assert s3 == 3

        # raw={3, 2, 1}: descending -> {None, None, 1}
        s1, s2, s3 = decode_tp_selections(3, 2, 1, menu_size=5)
        assert s1 is None
        assert s2 is None
        assert s3 == 1

    def test_tp_fraction_mapping_when_tp1_disabled(self) -> None:
        s1, s2, s3 = decode_tp_selections(DISABLED, 1, 3, menu_size=5)
        assert s1 is None
        assert s2 == 1
        assert s3 == 3
        fracs = decode_tp_fractions(0.0, 0.0, active_slots=(s1, s2, s3))
        assert len(fracs) == 2
        assert pytest.approx(fracs[0] + fracs[1], rel=1e-6) == 1.0
        # When z1=z2=0, f1=f2=f3=1/3. Normalizing f2 and f3 gives 0.5 each
        assert pytest.approx(fracs[0], rel=1e-6) == 0.5
        assert pytest.approx(fracs[1], rel=1e-6) == 0.5

    def test_tp_fraction_mapping_when_tp2_disabled(self) -> None:
        s1, s2, s3 = decode_tp_selections(0, DISABLED, 3, menu_size=5)
        assert s1 == 0
        assert s2 is None
        assert s3 == 3
        fracs = decode_tp_fractions(0.0, 0.0, active_slots=(s1, s2, s3))
        assert len(fracs) == 2
        assert pytest.approx(fracs[0], rel=1e-6) == 0.5
        assert pytest.approx(fracs[1], rel=1e-6) == 0.5

    def test_multi_tp_preserves_fraction_slot_mapping(self) -> None:
        # Verify slot identity: fraction for TP1 always couples to TP1,
        # fraction for TP3 always couples to TP3.
        # If z1 has high logit (f1 high) and z2 low:
        z1, z2 = 5.0, -5.0
        fracs_all = decode_tp_fractions(z1, z2, active_slots=(0, 1, 2))
        f1_raw, f2_raw, f3_raw = fracs_all
        assert f1_raw > 0.9  # TP1 gets the large fraction

        # Now suppose TP2 is disabled:
        fracs_sub = decode_tp_fractions(z1, z2, active_slots=(0, None, 2))
        f1_sub, f3_sub = fracs_sub
        # TP1 must still retain the dominant share, not remapped to TP3
        assert f1_sub > 0.9
        assert f3_sub < 0.1


class TestTPValidation:
    def test_validate_tp_levels_long(self) -> None:
        entry = 100.0
        sl = 98.0
        # Valid: SL < Entry <= TP1 < TP2 < TP3
        assert validate_tp_levels(1, entry, sl, [102.0, 104.0, 106.0])
        # Valid with middle disabled: SL < Entry <= TP1 < TP3
        assert validate_tp_levels(1, entry, sl, [102.0, None, 106.0])
        # Invalid: TP below entry
        assert not validate_tp_levels(1, entry, sl, [99.0, 104.0, 106.0])
        # Invalid: non-monotonic
        assert not validate_tp_levels(1, entry, sl, [105.0, 103.0, 106.0])
        # Invalid: TP below SL
        assert not validate_tp_levels(1, entry, sl, [97.0, None, 106.0])

    def test_validate_tp_levels_short(self) -> None:
        entry = 100.0
        sl = 102.0
        # Valid: TP3 < TP2 < TP1 <= Entry < SL
        # In slot order [TP1, TP2, TP3]: [98.0, 96.0, 94.0]
        assert validate_tp_levels(-1, entry, sl, [98.0, 96.0, 94.0])
        # Valid with middle disabled:
        assert validate_tp_levels(-1, entry, sl, [98.0, None, 94.0])
        # Invalid: TP above entry
        assert not validate_tp_levels(-1, entry, sl, [101.0, 96.0, 94.0])
        # Invalid: non-monotonic
        assert not validate_tp_levels(-1, entry, sl, [95.0, 97.0, 94.0])
        # Invalid: TP above SL
        assert not validate_tp_levels(-1, entry, sl, [103.0, None, 94.0])
