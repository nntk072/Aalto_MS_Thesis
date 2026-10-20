"""Execution-cost reserve used by risk-aware position sizing."""

from __future__ import annotations


def modeled_execution_reserve_usd(
    *, lots: float, slippage_points: float, commission_per_lot: float, contract_size: float
) -> float:
    lots_f = max(0.0, float(lots))
    adverse_stop_slippage = float(slippage_points) * lots_f * float(contract_size)
    round_trip_commission = 2.0 * float(commission_per_lot) * lots_f
    return max(0.0, adverse_stop_slippage + round_trip_commission)
