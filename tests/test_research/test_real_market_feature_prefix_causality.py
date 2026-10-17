from __future__ import annotations

import importlib.util
from pathlib import Path

from omegaconf import OmegaConf

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/matrix/audit_real_market_feature_prefix_causality.py"
)
SPEC = importlib.util.spec_from_file_location("real_market_feature_prefix_causality", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_feature_group_key_tracks_source_clock_model() -> None:
    cfg = OmegaConf.create(
        {
            "features": {},
            "data": {"tz": "Etc/GMT-3", "timestamp_clock": "native"},
            "session": {"tz": "Etc/GMT-3"},
            "strategy": {"session": {"timezone": "Etc/GMT-3"}},
        }
    )

    native_key = AUDIT.feature_config_key(cfg)
    cfg.data.timestamp_clock = "ftmo_mt5_us_dst"

    assert AUDIT.feature_config_key(cfg) != native_key
