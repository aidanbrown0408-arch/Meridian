"""Minimal inline-SVG chart rendering for George's HTML dashboard.

No charting library dependency: the dashboard is a single static HTML file,
so every chart is a hand-built `<svg>` string. Two shapes cover everything
the spec's report layout asks for (equity curve + benchmark overlay,
drawdown, per-trader P/L bars): a multi-series line chart and a
positive/negative bar chart.
"""

from __future__ import annotations

NAVY = "#0F1C33"
GOLD = "#D4AF6A"
PARCHMENT = "#E8E3D3"
GAIN = "#7FBF7F"
LOSS = "#C97A7A"
GRID = "#2A3B5C"

SERIES_COLORS = [GOLD, GAIN, "#8FB4D9", "#C9A0DC", LOSS]


def _fmt(n: float) -> str:
    return f"{n:.2f}"


def line_chart(series: dict[str, list[float]], width: int = 640, height: int = 220,
               pad: int = 36, title: str = "") -> str:
    """Multi-series line chart. Every series is plotted on a shared y-scale so
    an overlay (e.g. a strategy vs. its SPY benchmark) is directly comparable.
    Series with fewer points than the longest are left-padded with their
    first value so lines still start together."""
    series = {name: list(values) for name, values in series.items() if values}
    if not series:
        return _empty_chart(width, height, title, "no data")

    n = max(len(v) for v in series.values())
    padded = {name: ([v[0]] * (n - len(v)) + v if len(v) < n else v)
             for name, v in series.items()}

    all_values = [x for v in padded.values() for x in v]
    lo, hi = min(all_values), max(all_values)
    if hi == lo:
        hi = lo + 1.0

    plot_w, plot_h = width - 2 * pad, height - 2 * pad

    def px(i: int) -> float:
        return pad + (i / (n - 1) * plot_w if n > 1 else 0.0)

    def py(v: float) -> float:
        return pad + plot_h - (v - lo) / (hi - lo) * plot_h

    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
            f'font-family="Georgia, serif">']
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="{NAVY}"/>')
    if title:
        parts.append(f'<text x="{pad}" y="18" fill="{PARCHMENT}" font-size="13" '
                     f'font-family="Playfair Display, Georgia, serif">{title}</text>')

    # Zero baseline (helps read drawdown / return charts at a glance).
    if lo < 0 < hi:
        zero_y = py(0.0)
        parts.append(f'<line x1="{pad}" y1="{zero_y:.1f}" x2="{width - pad}" y2="{zero_y:.1f}" '
                     f'stroke="{GRID}" stroke-width="1" stroke-dasharray="3,3"/>')

    for idx, (name, values) in enumerate(padded.items()):
        color = SERIES_COLORS[idx % len(SERIES_COLORS)]
        points = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(values))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" '
                     f'stroke-width="2"/>')

    legend_x = pad
    for idx, name in enumerate(padded):
        color = SERIES_COLORS[idx % len(SERIES_COLORS)]
        y = height - 8
        parts.append(f'<rect x="{legend_x}" y="{y - 8}" width="9" height="9" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 13}" y="{y}" fill="{PARCHMENT}" font-size="11">'
                     f'{name}</text>')
        legend_x += 13 + 8 * len(name) + 16

    parts.append("</svg>")
    return "".join(parts)


def bar_chart(labels: list[str], values: list[float], width: int = 640, height: int = 220,
             pad: int = 40, title: str = "", unit: str = "%") -> str:
    """Horizontal-zero bar chart, green above zero and red below -- built for
    per-trader return/Sharpe comparisons."""
    if not labels or not values:
        return _empty_chart(width, height, title, "no data")

    lo, hi = min(0.0, min(values)), max(0.0, max(values))
    if hi == lo:
        hi = lo + 1.0
    plot_w, plot_h = width - 2 * pad, height - 2 * pad
    n = len(values)
    slot = plot_w / n
    bar_w = slot * 0.6

    def py(v: float) -> float:
        return pad + plot_h - (v - lo) / (hi - lo) * plot_h

    zero_y = py(0.0)
    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
            f'font-family="Georgia, serif">']
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="{NAVY}"/>')
    if title:
        parts.append(f'<text x="{pad}" y="18" fill="{PARCHMENT}" font-size="13" '
                     f'font-family="Playfair Display, Georgia, serif">{title}</text>')
    parts.append(f'<line x1="{pad}" y1="{zero_y:.1f}" x2="{width - pad}" y2="{zero_y:.1f}" '
                f'stroke="{GRID}" stroke-width="1"/>')

    for i, (label, value) in enumerate(zip(labels, values)):
        x = pad + i * slot + (slot - bar_w) / 2
        y_val = py(value)
        y, h = (y_val, zero_y - y_val) if value >= 0 else (zero_y, y_val - zero_y)
        color = GAIN if value >= 0 else LOSS
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{max(h, 0):.1f}" '
                     f'fill="{color}"/>')
        label_y = zero_y + 14 if value < 0 else zero_y - 6
        text_y = y - 4 if value >= 0 else y + h + 12
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{text_y:.1f}" fill="{PARCHMENT}" '
                     f'font-size="10" text-anchor="middle">{_fmt(value)}{unit}</text>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{height - 6}" fill="{PARCHMENT}" '
                     f'font-size="10" text-anchor="middle">{label}</text>')

    parts.append("</svg>")
    return "".join(parts)


def _empty_chart(width: int, height: int, title: str, message: str) -> str:
    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
            f'font-family="Georgia, serif">',
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="{NAVY}"/>']
    if title:
        parts.append(f'<text x="20" y="18" fill="{PARCHMENT}" font-size="13" '
                     f'font-family="Playfair Display, Georgia, serif">{title}</text>')
    parts.append(f'<text x="{width / 2}" y="{height / 2}" fill="{GOLD}" font-size="13" '
                f'text-anchor="middle">{message}</text>')
    parts.append("</svg>")
    return "".join(parts)
