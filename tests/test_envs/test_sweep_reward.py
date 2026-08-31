"""Tests for sweep confirmation reward."""

import pytest

from quant_rl.envs.sweep_reward import CompositeReward, SweepConfirmationReward


class TestSweepConfirmationReward:
    """Tests for SweepConfirmationReward class."""

    def test_london_sweep_confirmed(self) -> None:
        """Test that London sweep held for 3 bars returns C_t = +1.0."""
        reward_fn = SweepConfirmationReward(alpha=0.1, beta=0.01, hold_bars=3)
        reward_fn.reset()

        # Start below London High, then cross and hold above
        prices = [94, 96, 97, 98, 99]
        for i, price in enumerate(prices):
            r = reward_fn(
                0,
                0,
                price,
                london_high=95,
                london_low=90,
                asian_high=98,
                asian_low=92,
                minutes_since_open=0,
            )

        # After crossing and holding for 3 bars, should get +1.0 * alpha = 0.1
        assert r == pytest.approx(0.1, abs=0.01)

    def test_false_breakout(self) -> None:
        """Test that false breakout returns C_t = -0.5."""
        reward_fn = SweepConfirmationReward(alpha=0.1, beta=0.01, hold_bars=3)
        reward_fn.reset()

        # Start below, cross above, then drop back below
        reward_fn(
            0,
            0,
            94,
            london_high=100,
            london_low=90,
            asian_high=98,
            asian_low=92,
            minutes_since_open=0,
        )
        reward_fn(
            0,
            0,
            101,
            london_high=100,
            london_low=90,
            asian_high=98,
            asian_low=92,
            minutes_since_open=1,
        )
        r = reward_fn(
            0,
            0,
            99,
            london_high=100,
            london_low=90,
            asian_high=98,
            asian_low=92,
            minutes_since_open=2,
        )

        assert r == pytest.approx(-0.05, abs=0.01)

    def test_asian_sweep_confirmed(self) -> None:
        """Test that Asian sweep held for 3 bars returns C_t = +0.5."""
        reward_fn = SweepConfirmationReward(alpha=0.1, beta=0.01, hold_bars=3)
        reward_fn.reset()

        # Start below Asian High, then cross and hold above
        # London High is far above, so won't interfere
        prices = [93, 96, 97, 98, 99]
        for i, price in enumerate(prices):
            r = reward_fn(
                0,
                0,
                price,
                london_high=105,
                london_low=90,
                asian_high=95,
                asian_low=92,
                minutes_since_open=0,
            )

        # After crossing and holding for 3 bars, should get +0.5 * alpha = 0.05
        assert r == pytest.approx(0.05, abs=0.01)

    def test_london_priority(self) -> None:
        """Test that London sweep takes priority over Asian sweep."""
        reward_fn = SweepConfirmationReward(alpha=0.1, beta=0.01, hold_bars=3)
        reward_fn.reset()

        reward_fn(
            0,
            0,
            98,
            london_high=100,
            london_low=90,
            asian_high=95,
            asian_low=92,
            minutes_since_open=0,
        )
        reward_fn(
            0,
            0,
            101,
            london_high=100,
            london_low=90,
            asian_high=95,
            asian_low=92,
            minutes_since_open=1,
        )
        for i in range(3):
            r = reward_fn(
                0,
                0,
                102 + i,
                london_high=100,
                london_low=90,
                asian_high=95,
                asian_low=92,
                minutes_since_open=2 + i,
            )

        assert r == pytest.approx(0.1, abs=0.01)

    def test_reset(self) -> None:
        """Test that reset() properly clears all state."""
        reward_fn = SweepConfirmationReward()
        reward_fn.prev_price = 100.0
        reward_fn.in_london_long_sweep = True
        reward_fn.london_long_counter = 5
        reward_fn.reset()

        assert reward_fn.prev_price is None
        assert reward_fn.in_london_long_sweep is False
        assert reward_fn.london_long_counter == 0

    def test_time_decay(self) -> None:
        """Test that time decay penalty is applied after 20 minutes."""
        reward_fn = SweepConfirmationReward(alpha=0.1, beta=0.01, hold_bars=3)
        reward_fn.reset()

        r1 = reward_fn(
            0,
            0,
            100,
            london_high=95,
            london_low=90,
            asian_high=98,
            asian_low=92,
            minutes_since_open=10,
        )
        r2 = reward_fn(
            0,
            0,
            100,
            london_high=95,
            london_low=90,
            asian_high=98,
            asian_low=92,
            minutes_since_open=30,
        )

        assert r1 == pytest.approx(0.0, abs=0.01)
        assert r2 == pytest.approx(-0.1, abs=0.01)

    def test_nan_handling(self) -> None:
        """Test that NaN values in liquidity levels are handled gracefully."""
        reward_fn = SweepConfirmationReward()
        reward_fn.reset()

        r = reward_fn(
            0, 0, 100, float("nan"), float("nan"), float("nan"), float("nan"), minutes_since_open=0
        )

        assert r == 0.0

    def test_position_changed_cost(self) -> None:
        """Test that position changes add cost_per_trade."""
        reward_fn = SweepConfirmationReward(cost_per_trade=1.0)
        reward_fn.reset()

        r1 = reward_fn(10.0, 0.5, 100, 95, 90, 98, 92, 0, position_changed=False)
        r2 = reward_fn(10.0, 0.5, 100, 95, 90, 98, 92, 0, position_changed=True)

        assert r2 == pytest.approx(r1 - 1.0, abs=0.01)


class TestCompositeReward:
    """Tests for CompositeReward class."""

    def test_composite_combines_rewards(self) -> None:
        """Test that composite reward combines DSR and sweep rewards."""
        sweep_reward = SweepConfirmationReward(alpha=0.1, beta=0.01)
        composite = CompositeReward(sweep_reward=sweep_reward, dsr_weight=0.5, sweep_weight=0.5)
        composite.reset()

        r = composite(
            pnl_step=10.0,
            cost=0.0,
            price=105,
            london_high=100,
            london_low=90,
            asian_high=98,
            asian_low=92,
            minutes_since_open=0,
            position_changed=False,
            dsr_reward=2.0,
        )

        assert r > 0.5 * 2.0

    def test_composite_fallback_to_dsr(self) -> None:
        """Test that composite falls back to DSR when sweep params not provided."""
        sweep_reward = SweepConfirmationReward()
        composite = CompositeReward(sweep_reward=sweep_reward, dsr_weight=0.7, sweep_weight=0.3)
        composite.reset()

        r = composite(
            pnl_step=10.0,
            daily_loss=0.0,
            daily_loss_limit=5000.0,
            initial_balance=100000.0,
            breach=False,
        )

        assert isinstance(r, float)

    def test_composite_reset(self) -> None:
        """Test that composite reset() resets the sweep reward."""
        sweep_reward = SweepConfirmationReward()
        composite = CompositeReward(sweep_reward=sweep_reward)

        sweep_reward.prev_price = 100.0
        sweep_reward.in_london_long_sweep = True

        composite.reset()

        assert sweep_reward.prev_price is None
        assert sweep_reward.prev_price is None
        assert sweep_reward.in_london_long_sweep is False

    def test_composite_dsr_state_persists(self) -> None:
        """DSR EMA state must accumulate across steps, not reset each call.

        The old bug created a fresh DSRReward() inside __call__, zeroing the
        EMA (A, B) every step so the DSR signal was always ~0. With a
        persistent instance, feeding repeated positive PnL should produce a
        non-zero, evolving DSR component.
        """
        sweep_reward = SweepConfirmationReward()
        composite = CompositeReward(sweep_reward=sweep_reward, dsr_weight=1.0, sweep_weight=0.0)
        composite.reset()

        rewards = []
        for _ in range(20):
            r = composite(
                pnl_step=50.0,
                daily_loss=0.0,
                daily_loss_limit=5000.0,
                initial_balance=100_000.0,
                breach=False,
            )
            rewards.append(r)

        assert any(abs(r) > 1e-6 for r in rewards), "DSR component is dead (all zeros)"
        assert len(set(round(r, 8) for r in rewards)) > 1, (
            "DSR signal is static — EMA not accumulating"
        )

    def test_composite_dsr_reset_clears_state(self) -> None:
        """reset() must clear the DSR EMA so episodes start cold.

        Note: a cold-start first step does NOT yield ~0 — with a single sample
        the DSR denominator (B - A²) is tiny, so the ratio is large. The right
        invariant is cold-start equivalence: after reset(), the reward must be
        identical to what a fresh instance produces on its first step.
        """
        sweep_reward = SweepConfirmationReward()
        composite = CompositeReward(sweep_reward=sweep_reward, dsr_weight=1.0, sweep_weight=0.0)
        composite.reset()

        for _ in range(10):
            composite(
                pnl_step=50.0,
                daily_loss=0.0,
                daily_loss_limit=5000.0,
                initial_balance=100_000.0,
                breach=False,
            )

        composite.reset()
        first_after_reset = composite(
            pnl_step=50.0,
            daily_loss=0.0,
            daily_loss_limit=5000.0,
            initial_balance=100_000.0,
            breach=False,
        )

        # Reference: a fresh instance's first step (same cold EMA state A=B=0)
        fresh = CompositeReward(
            sweep_reward=SweepConfirmationReward(), dsr_weight=1.0, sweep_weight=0.0
        )
        fresh_first = fresh(
            pnl_step=50.0,
            daily_loss=0.0,
            daily_loss_limit=5000.0,
            initial_balance=100_000.0,
            breach=False,
        )

        assert first_after_reset == pytest.approx(fresh_first, abs=1e-12)
        assert sweep_reward.in_london_long_sweep is False
