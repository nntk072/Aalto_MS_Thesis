from __future__ import annotations

import pytest

from features.contracts import InputKind, InputSchema, InputView


@pytest.mark.parametrize(
    "kind",
    ["handcrafted_features", "raw_market", "learned_representation"],
)
def test_distinct_input_representations_share_one_schema_contract(kind: InputKind) -> None:
    schema = InputSchema("market-input", "v1", kind, ("x0", "x1"))

    view = InputView(schema, "dataset-1", ((0.1, 0.2), (0.3, 0.4)))

    assert view.schema.kind == kind
    assert len(view.values) == 2


def test_input_view_rejects_width_schema_mismatch() -> None:
    schema = InputSchema("market-input", "v1", "raw_market", ("close", "volume"))
    with pytest.raises(ValueError, match="schema width"):
        InputView(schema, "dataset-1", ((1.0,),))
