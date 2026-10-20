"""Tests for DistributionReward and the Idea 2 respect gate."""

from __future__ import annotations

import pandas as pd
import pytest

from simulation.environment.distribution_reward import DistributionReward
from simulation.environment.env import _previous_session_audit
from trading.strategies.distribution import DistributionStrategy


def reward() -> DistributionReward:
    return DistributionReward(
        entry_bonus=1.0,
        sweep_penalty=0.5,
        distribution_bonus=0.5,
    )


def test_consistent_long_chain_is_rewarded() -> None:
    """A distribution bar after the opposing gap failed is the long entry."""
    r = reward()
    out = r(
        position_changed=True,
        direction=1,
        distribution_after_ifvg=True,
        distribution_phase=True,
    )
    # +entry_bonus +distribution_bonus = 1.5
    assert out == pytest.approx(1.5)


def test_consistent_short_chain_is_rewarded() -> None:
    """A distribution bar after the opposing gap failed is the short entry."""
    r = reward()
    out = r(
        position_changed=True,
        direction=-1,
        distribution_after_ifvg=True,
        distribution_phase=False,
    )
    assert out == pytest.approx(1.0)


def test_inconsistent_mapping_is_penalised() -> None:
    """An entry without a respected IFVG is not the distribution chain."""
    r = reward()
    out = r(
        position_changed=True,
        direction=1,
        distribution_after_ifvg=False,
        distribution_phase=True,
    )
    # -sweep_penalty +distribution_bonus = 0.0
    assert out == pytest.approx(0.0)


def test_no_context_is_penalised() -> None:
    r = reward()
    out = r(
        position_changed=True,
        direction=1,
        distribution_after_ifvg=False,
        distribution_phase=False,
    )
    assert out == pytest.approx(-0.5)


def test_no_position_transition_gives_no_signal() -> None:
    r = reward()
    out = r(
        position_changed=False,
        direction=0,
        distribution_after_ifvg=True,
        distribution_phase=True,
    )
    assert out == 0.0


def test_reset_is_noop() -> None:
    r = reward()
    r(
        position_changed=True,
        direction=1,
        distribution_after_ifvg=True,
        distribution_phase=True,
    )
    r.reset()
    assert (
        r(
            position_changed=False,
            direction=0,
            distribution_after_ifvg=False,
            distribution_phase=False,
        )
        == 0.0
    )


def _long_leg(**extra: float) -> pd.Series:
    row = {
        "po3_distribution": 1.0,
        "po3_distribution_direction": 1.0,
        "htf_day_bias": 1.0,
        "po3_distribution_after_ifvg": 1.0,
    }
    row.update(extra)
    return pd.Series(row)


def test_distribution_after_the_gap_fails_sets_the_side() -> None:
    strategy = DistributionStrategy(enforce_gate=False)
    assert strategy.context_direction(_long_leg(bos_up=1.0)) == 1
    short_row = pd.Series(
        {
            "po3_distribution": 1.0,
            "po3_distribution_direction": -1.0,
            "htf_day_bias": -1.0,
            "po3_distribution_after_ifvg": 1.0,
        }
    )
    assert strategy.context_direction(short_row) == -1


def test_failure_bar_and_a_bias_mismatch_hold() -> None:
    strategy = DistributionStrategy(enforce_gate=True)
    before = _long_leg(po3_distribution_after_ifvg=0.0, bos_up=1.0)
    assert strategy.context_direction(before) == 0
    assert strategy.validate_entry(direction=1, row=before) is False
    against = _long_leg(htf_day_bias=-1.0)
    assert strategy.context_direction(against) == 0


def test_idea2_requires_all_direction_and_ifvg_gate_inputs() -> None:
    required = set(DistributionStrategy.required_features)
    assert {
        "po3_distribution",
        "po3_distribution_direction",
        "htf_day_bias",
        "po3_distribution_after_ifvg",
    }.issubset(required)


def test_idea2_accepts_post_confirmation_distribution_without_retest() -> None:
    strategy = DistributionStrategy(enforce_gate=True)
    row = _long_leg()
    assert "ifvg_retest_bull" not in row.index
    assert strategy.validate_entry(direction=1, row=row)
    assert not strategy.validate_entry(direction=-1, row=row)


def test_idea2_confirmed_mode_is_eligible_on_confirmation_close() -> None:
    strategy = DistributionStrategy(enforce_gate=True, entry_mode="confirmed_first_eligible")
    row = _long_leg(po3_distribution_after_ifvg=0.0, po3_ifvg_confirmed=1.0)
    assert strategy.validate_entry(direction=1, row=row)


def test_previous_session_pd_gate_is_strict_per_entry_and_reports_reasons() -> None:
    strategy = DistributionStrategy(
        enforce_gate=True,
        entry_mode="confirmed_first_eligible",
        previous_session_pd_gate=True,
    )
    long_discount = _long_leg(
        po3_distribution_after_ifvg=0.0,
        po3_ifvg_confirmed=1.0,
        previous_session_midpoint=108.0,
        decision_close=107.0,
    )
    assert strategy.validate_entry(direction=1, row=long_discount)
    assert strategy.context_direction(long_discount) == 1

    premium = long_discount.copy()
    premium["decision_close"] = 112.0
    assert not strategy.validate_entry(direction=1, row=premium)
    # Context still proposes the setup direction; validate_entry rejects that
    # specific new order and preserves the explicit premium_block reason.
    assert strategy.context_direction(premium) == 1
    assert strategy.entry_rejection_reason(direction=1, row=premium) == "premium_block"

    equality = long_discount.copy()
    equality["decision_close"] = 108.0
    assert not strategy.validate_entry(direction=1, row=equality)
    assert strategy.entry_rejection_reason(direction=1, row=equality) == "premium_block"

    missing = long_discount.drop(labels=["previous_session_midpoint"])
    assert not strategy.validate_entry(direction=1, row=missing)
    assert strategy.entry_rejection_reason(direction=1, row=missing) == "missing_previous_session"


def test_previous_session_pd_gate_mirrors_short_and_can_be_disabled() -> None:
    short = pd.Series(
        {
            "po3_distribution": 1.0,
            "po3_distribution_direction": -1.0,
            "htf_day_bias": -1.0,
            "po3_ifvg_confirmed": 1.0,
            "previous_session_midpoint": 108.0,
            "decision_close": 109.0,
        }
    )
    strategy = DistributionStrategy(
        entry_mode="confirmed_first_eligible", previous_session_pd_gate=True
    )
    assert strategy.validate_entry(direction=-1, row=short)
    short["decision_close"] = 108.0
    assert not strategy.validate_entry(direction=-1, row=short)
    assert strategy.entry_rejection_reason(direction=-1, row=short) == "discount_block"

    baseline = DistributionStrategy(entry_mode="confirmed_first_eligible")
    assert baseline.validate_entry(direction=-1, row=short)


def test_previous_session_audit_records_reference_and_decision_price() -> None:
    strategy = DistributionStrategy(previous_session_pd_gate=True)
    row = pd.Series(
        {
            "previous_session_high": 120.0,
            "previous_session_low": 96.0,
            "previous_session_midpoint": 108.0,
            "previous_session_code": 3.0,
            "previous_session_completed_at_ns": 1_800_000_000_000_000_000.0,
        }
    )
    audit = _previous_session_audit(strategy, row, 112.0)
    assert audit["decision_close"] == 112.0
    assert audit["previous_session_label"] == "ny"
    assert audit["previous_session_midpoint"] == 108.0
    assert audit["previous_session_completed_at_ns"] > 0


def test_idea2_retest_mode_requires_later_touch_inside_candidate_window() -> None:
    strategy = DistributionStrategy(enforce_gate=True, entry_mode="confirmed_retest")
    same_bar = _long_leg(
        po3_distribution_after_ifvg=0.0,
        ifvg_retest_bull=1.0,
        ifvg_bull_age=0.0,
    )
    later = _long_leg(ifvg_retest_bull=1.0, ifvg_bull_age=3.0)
    expired = _long_leg(ifvg_retest_bull=1.0, ifvg_bull_age=6.0)

    assert not strategy.validate_entry(direction=1, row=same_bar)
    assert strategy.validate_entry(direction=1, row=later)
    assert not strategy.validate_entry(direction=1, row=expired)


def test_stop_uses_a_respected_lower_gap_and_an_aligned_higher_one() -> None:
    strategy = DistributionStrategy(enforce_gate=False)
    row = pd.Series(
        {
            "last_swing_low": 10.0,
            "sweep_low_level": 11.0,
            "asian_low": 12.0,
            "london_low": 13.0,
            "ifvg_bull_active": 1.0,
            "ifvg_bull_low": 9.0,
            "ifvg_bull_origin": 8.5,
            "M15_ifvg_bull_active": 0.0,
            "M15_ifvg_bull_low": 5.0,
            "M15_ifvg_bull_origin": 4.5,
            "H1_ifvg_bull_active": 1.0,
            "H1_ifvg_bull_low": 7.0,
            "H1_ifvg_bull_origin": 6.5,
        }
    )
    names = [name for name, _ in strategy.sl_candidates(direction=1, row=row)]
    assert names == [
        "last_swing_low",
        "sweep_low_level",
        "asian_low",
        "london_low",
        "ifvg_bull_low",
        "ifvg_bull_origin",
        "H1_ifvg_bull_low",
        "H1_ifvg_bull_origin",
    ]
