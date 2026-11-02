"""The feature card: what the teacher LLM and the students see for one trade signal.

A card describes a StochRSI turn on 15-minute crypto bars in relative terms only:
no symbol, no date, no price level. The outcome of the trade is never on the card.
The text layout matches the card the teacher was prompted with upstream, so the
text student reads the same words the LLM read.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

SESSIONS = [(0, 8, "Asia"), (8, 13, "Europe"), (13, 21, "US"), (21, 24, "late US")]

# Numeric inputs of the tabular students, in a fixed order.
FEATURES = [
    "is_long", "k", "k_prev", "atr_pct", "atr_rank_30d", "fee_pct_of_risk",
    "ema50_atr", "ema200_atr", "ema50_above_200", "vwap_atr", "range_pos_20",
    "move_1", "move_2", "move_3", "move_4", "move_sum_4", "rsi14", "volume_ratio_20",
    "hour_sin", "hour_cos",
]


@dataclass
class Card:
    """One labelled signal: card fields, its outcome, and the teacher's answer."""

    key: str  # unique id, e.g. "<bar_start_ts>-<direction>"
    ts: int  # signal bar start, unix seconds (used for the time split only)
    group: str  # where it came from, e.g. "ADA" or "BTC" (never shown to a model)
    direction: str  # LONG or SHORT
    k: float
    k_prev: float
    atr_pct: float
    atr_rank_30d: float | None
    fee_pct_of_risk: float
    ema50_atr: float | None
    ema200_atr: float | None
    ema50_above_200: bool | None
    vwap_atr: float | None
    range_pos_20: float
    moves_4: list[float]
    rsi14: float | None
    volume_ratio_20: float | None
    utc_hour: int
    # ground truth (never on the card): win / loss / timeout and R after costs
    outcome: str
    r_net: float
    # teacher answer: decision may be None if the teacher gave none (v2 prompts)
    teacher_decision: str | None = None
    teacher_p_win: float | None = None
    teacher_reason: str = ""
    teacher_seconds: float | None = None
    text: str = field(default="")

    def __post_init__(self) -> None:
        if not self.text:
            self.text = render(self)

    @property
    def session(self) -> str:
        return session_of(self.utc_hour)

    @property
    def teacher_take(self) -> bool:
        return self.teacher_decision == "TAKE"

    def to_dict(self) -> dict:
        return asdict(self)


def session_of(hour: int) -> str:
    return next(name for a, b, name in SESSIONS if a <= hour < b)


def _n(x, fmt: str = "{:+.2f}") -> str:
    return "n/a" if x is None else fmt.format(x)


def render(c: Card) -> str:
    """The card text, as the teacher saw it."""
    zone = "oversold" if c.direction == "LONG" else "overbought"
    trend = ("n/a" if c.ema50_above_200 is None
             else "EMA50 above EMA200" if c.ema50_above_200 else "EMA50 below EMA200")
    return "\n".join([
        f"Signal: {c.direction}, 15m bars. StochRSI K {c.k_prev:.1f} -> {c.k:.1f} "
        f"(turn out of {zone}).",
        f"Volatility: ATR(14) = {c.atr_pct:.2f}% of price; rank "
        f"{_n(c.atr_rank_30d, '{:.0f}/100')} over the last 30 days.",
        f"Cost: round-trip fee = {c.fee_pct_of_risk:.0f}% of the 1-ATR risk.",
        "Trade plan: stop 1 ATR, take-profit 2 ATR, max 20 bars.",
        f"Trend: price vs EMA50 {_n(c.ema50_atr)} ATR, vs EMA200 {_n(c.ema200_atr)} ATR; "
        f"{trend}.",
        f"Location: price vs daily VWAP {_n(c.vwap_atr)} ATR; position in last 20 bars' "
        f"range {c.range_pos_20:.0f}/100.",
        "Last 4 bar moves (ATR, oldest first): "
        + ", ".join(f"{m:+.2f}" for m in c.moves_4) + ".",
        f"RSI(14) {_n(c.rsi14, '{:.0f}')}; volume {_n(c.volume_ratio_20, '{:.1f}')}x its "
        "20-bar average.",
        f"Time: {c.utc_hour:02d}:00 UTC ({c.session} session).",
        "Answer: TAKE or DONT_TAKE, win probability 0-1, reason in at most 40 words.",
    ])


def feature_row(c: Card) -> list[float]:
    """The card as numbers for the tabular students; missing values are NaN."""
    def f(x):
        return math.nan if x is None else float(x)

    angle = 2 * math.pi * c.utc_hour / 24
    return [
        float(c.direction == "LONG"), c.k, c.k_prev, c.atr_pct, f(c.atr_rank_30d),
        c.fee_pct_of_risk, f(c.ema50_atr), f(c.ema200_atr), f(c.ema50_above_200),
        f(c.vwap_atr), c.range_pos_20, *c.moves_4, sum(c.moves_4), f(c.rsi14),
        f(c.volume_ratio_20), math.sin(angle), math.cos(angle),
    ]
