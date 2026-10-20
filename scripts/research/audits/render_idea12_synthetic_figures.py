#!/usr/bin/env python3
"""Render the standalone synthetic Idea 1_2 timing and gate diagrams as SVG."""

from __future__ import annotations

import json

from configuration.paths import repository_root

ROOT = repository_root(__file__)
SOURCE = ROOT / "docs/decisions/fixtures/idea12_synthetic_fixtures.json"
OUT = ROOT / "docs/thesis/figures"


def _svg(body: str, title: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="560" viewBox="0 0 1000 560">
<title>{title}</title><rect width="1000" height="560" fill="#fbfcfe"/>
<style>text{{font:14px sans-serif;fill:#152536}} .small{{font-size:12px}} .axis{{stroke:#8091a0;stroke-width:1}} .bull{{stroke:#18794e;fill:#18794e}} .bear{{stroke:#b42318;fill:#b42318}} .ref{{stroke:#7a5af8;stroke-dasharray:6 5}} .zone{{fill:#fdb022;fill-opacity:.19;stroke:#dc6803}} .gate{{stroke:#175cd3;stroke-dasharray:7 5}}</style>
<text x="44" y="34" font-size="20" font-weight="bold">{title}</text>
<text x="44" y="54" class="small">SYNTHETIC EXPLANATORY EXAMPLE — not historical data or a strategy result</text>
{body}</svg>"""


def render_fixture_a(data: dict[str, object]) -> str:
    fixture = data["fixture_a_confirmed_long"]
    assert isinstance(fixture, dict)
    bars = fixture["bars"]
    assert isinstance(bars, list)
    x_positions = [210, 390, 570, 750]
    low_y, high_y = 440, 110
    pmin, pmax = 90.0, 155.0

    def y(price: float) -> float:
        return high_y + (pmax - price) / (pmax - pmin) * (low_y - high_y)

    marks: list[str] = []
    for i, bar in enumerate(bars):
        assert isinstance(bar, dict)
        x = x_positions[i]
        marks.append(
            f'<line class="axis" x1="{x}" y1="{y(bar["high"]):.1f}" x2="{x}" y2="{y(bar["low"]):.1f}"/>'
        )
        color = "bull" if bar["close"] >= bar["open"] else "bear"
        top = y(max(bar["open"], bar["close"]))
        height = max(4, abs(y(float(bar["open"])) - y(float(bar["close"]))))
        marks.append(
            f'<rect class="{color}" x="{x - 22}" y="{top:.1f}" width="44" height="{height:.1f}" opacity=".78"/>'
        )
        marks.append(f'<text x="{x - 5}" y="462">{bar["bar"]}</text>')
    stop, target = float(fixture["stop"]), float(fixture["target"])
    zone_top, zone_bottom = y(116), y(112)
    marks.extend(
        [
            f'<rect class="zone" x="120" y="{zone_top:.1f}" width="720" height="{zone_bottom - zone_top:.1f}"/>',
            f'<line class="ref" x1="100" y1="{y(100):.1f}" x2="850" y2="{y(100):.1f}"/>',
            f'<line x1="110" y1="{y(stop):.1f}" x2="850" y2="{y(stop):.1f}" stroke="#b42318"/>',
            f'<line x1="110" y1="{y(target):.1f}" x2="850" y2="{y(target):.1f}" stroke="#18794e"/>',
            f'<text x="860" y="{y(100) + 4:.1f}">confirmed M5 low 100</text>',
            f'<text x="860" y="{y(114) + 4:.1f}">bearish FVG [112,116]</text>',
            f'<text x="860" y="{y(stop) + 4:.1f}">SL 97</text>',
            f'<text x="860" y="{y(target) + 4:.1f}">TP 150</text>',
            '<text x="520" y="90" fill="#175cd3">IFVG confirmation close</text>',
            '<line class="gate" x1="600" y1="96" x2="600" y2="485"/>',
            '<text x="710" y="500">next bar open: earliest fill 118</text>',
            '<text x="75" y="520" class="small">Causal order: pivot confirmation → sweep/reclaim (20) → IFVG close (21) → submit after close → fill (22)</text>',
        ]
    )
    return _svg("\n".join(marks), "Figure 17 — Confirmed PO3–IFVG entry timing")


def render_fixture_b(data: dict[str, object]) -> str:
    fixture = data["fixture_b_idea2_gate"]
    assert isinstance(fixture, dict)
    session = fixture["previous_completed_session"]
    assert isinstance(session, dict)
    hi, lo, mid = (float(session[k]) for k in ("high", "low", "midpoint"))
    close = float(fixture["decision_close"])
    x0, x1 = 220, 790
    y_top, y_bottom = 120, 420

    def y(price: float) -> float:
        return y_bottom - (price - lo) / (hi - lo) * (y_bottom - y_top)

    body = [
        f'<rect x="{x0}" y="{y(hi):.1f}" width="{x1 - x0}" height="{y(lo) - y(hi):.1f}" fill="#d1e9ff" stroke="#175cd3"/>',
        f'<line class="gate" x1="{x0}" y1="{y(mid):.1f}" x2="{x1}" y2="{y(mid):.1f}"/>',
        f'<line x1="{x0}" y1="{y(close):.1f}" x2="{x1}" y2="{y(close):.1f}" stroke="#b42318" stroke-width="3"/>',
        f'<text x="810" y="{y(hi) + 4:.1f}">previous completed {session["label"]} high 120</text>',
        f'<text x="810" y="{y(mid) + 4:.1f}">midpoint 108</text>',
        f'<text x="810" y="{y(close) + 4:.1f}">decision close 112</text>',
        f'<text x="810" y="{y(lo) + 4:.1f}">low 96</text>',
        '<text x="230" y="470" fill="#b42318">New long blocked: close is at premium</text>',
        '<text x="230" y="495">An already open long continues under its existing exit rules.</text>',
        '<text x="65" y="535" class="small">Session calendar: Asia 19–23, London 02–05, NY 08:30–15:00 America/Chicago; DST-aware.</text>',
    ]
    return _svg("\n".join(body), "Figure 18 — Idea 2 previous-session midpoint gate")


def render_retest(data: dict[str, object]) -> str:
    fixture = data["fixture_a_confirmed_long"]
    assert isinstance(fixture, dict)
    retest = fixture["optional_retest_example"]
    assert isinstance(retest, dict)
    body = [
        '<rect x="110" y="160" width="780" height="105" rx="8" fill="#eef4ff" stroke="#175cd3"/>',
        '<rect x="110" y="315" width="780" height="105" rx="8" fill="#ecfdf3" stroke="#18794e"/>',
        '<text x="140" y="195" font-weight="bold">Main: confirmed first eligible</text>',
        '<text x="140" y="230">IFVG confirmed at bar 21 close → submit → earliest fill at bar 22 open.</text>',
        '<text x="140" y="350" font-weight="bold">Optional: confirmed retest</text>',
        f'<text x="140" y="385">Later touch at {retest["touch_price"]} inside [112,116] on bar {retest["touch_bar"]} → submit after close → fill bar {retest["earliest_fill_bar"]}.</text>',
        '<text x="110" y="470" class="small">The interior touch is one synthetic example. The retest mode accepts any touch in the zone within its expiry.</text>',
        '<text x="110" y="495" class="small">The early exploratory variant is unconfirmed and is not depicted or labeled as an IFVG confirmation.</text>',
    ]
    return _svg("\n".join(body), "Optional confirmed retest variant — separate from main entry")


def main() -> None:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "idea12_figure17_synthetic.svg").write_text(render_fixture_a(data), encoding="utf-8")
    (OUT / "idea12_figure18_synthetic.svg").write_text(render_fixture_b(data), encoding="utf-8")
    (OUT / "idea12_confirmed_retest_synthetic.svg").write_text(
        render_retest(data), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
