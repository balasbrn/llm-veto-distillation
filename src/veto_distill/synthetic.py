"""A synthetic dataset with the same shape as the real one, for tests, CI and the demo.

Everything here is made up: the card fields are drawn from rough ranges seen on real
cards, the outcomes carry a small planted signal, and the "teacher" is a hand-written
heuristic that reads the card the way an LLM might (trend, RSI, cost) with noise and
p_win rounded to 0.05. It lets the whole pipeline run without market data or an API key.
Results on it say nothing about markets or LLMs.
"""

from __future__ import annotations

import math
import random

from veto_distill.card import Card
from veto_distill.teacher import rule_decision

START_TS = 1_640_995_200  # 2022-01-01, only used to order cards in time
BAR_S = 900


def _card_fields(rng: random.Random) -> dict:
    direction = rng.choice(["LONG", "SHORT"])
    k = rng.uniform(0, 25)
    k_prev = max(0.0, k - rng.uniform(0.5, 15))
    if direction == "SHORT":
        k, k_prev = 100 - k, 100 - k_prev
    atr_pct = rng.lognormvariate(math.log(0.5), 0.35)
    ema50 = rng.gauss(0, 2.5)
    ema200 = ema50 + rng.gauss(0, 3)
    return {
        "direction": direction, "k": round(k, 1), "k_prev": round(k_prev, 1),
        "atr_pct": round(atr_pct, 3),
        "atr_rank_30d": None if rng.random() < 0.03 else round(rng.uniform(0, 100), 1),
        "fee_pct_of_risk": round(min(10.0, 8 / atr_pct), 2),
        "ema50_atr": round(ema50, 2), "ema200_atr": round(ema200, 2),
        "ema50_above_200": ema200 < ema50,
        "vwap_atr": round(rng.gauss(0, 2), 2), "range_pos_20": round(rng.uniform(0, 100), 1),
        "moves_4": [round(rng.gauss(0, 0.5), 2) for _ in range(4)],
        "rsi14": round(rng.uniform(20, 45) if direction == "LONG" else rng.uniform(55, 80), 1),
        "volume_ratio_20": round(rng.lognormvariate(0, 0.5), 2),
        "utc_hour": rng.randrange(24),
    }


def _outcome(f: dict, rng: random.Random) -> tuple[str, float]:
    """Win / loss / timeout with a small planted signal, and R net after costs."""
    s = 1 if f["direction"] == "LONG" else -1
    with_trend = s * f["ema50_atr"] > 0
    p_win = 0.30 + 0.04 * with_trend - 0.04 * (2 <= f["utc_hour"] < 8)
    p_win += 0.01 * max(-3.0, min(3.0, -s * f["vwap_atr"]))
    u = rng.random()
    cost = f["fee_pct_of_risk"] / 100 + 0.002
    if u < p_win:
        return "win", 2.0 - cost
    if u < p_win + 0.55:
        return "loss", -1.0 - cost
    return "timeout", rng.gauss(-0.1, 0.5) - cost


def heuristic_teacher(f: dict, rng: random.Random) -> tuple[float, str]:
    """A stand-in for the LLM: p_win from a few card cues, plus noise, in 0.05 steps."""
    s = 1 if f["direction"] == "LONG" else -1
    p = 0.33
    reasons = []
    if s * f["ema50_atr"] > 0:
        p += 0.05
        reasons.append("trend supports the direction")
    else:
        reasons.append("against the EMA50 trend")
    if (f["direction"] == "LONG" and f["rsi14"] < 30) or (f["direction"] == "SHORT"
                                                          and f["rsi14"] > 70):
        p += 0.04
        reasons.append("RSI stretched")
    if f["volume_ratio_20"] > 1.5:
        p += 0.03
        reasons.append("volume confirms")
    if f["fee_pct_of_risk"] > 8:
        p -= 0.03
        reasons.append("fee heavy")
    p += rng.gauss(0, 0.04)
    p = min(0.95, max(0.05, round(p * 20) / 20))
    return p, "; ".join(reasons) + "."


def make_dataset(n: int = 1200, seed: int = 7, groups: tuple[str, ...] = ("A", "B")) -> list[Card]:
    rng = random.Random(seed)
    cards = []
    for i in range(n):
        f = _card_fields(rng)
        outcome, r_net = _outcome(f, rng)
        p, reason = heuristic_teacher(f, rng)
        ts = START_TS + i * 7 * BAR_S + rng.randrange(7) * BAR_S
        cards.append(Card(
            key=f"syn-{i}", ts=ts, group=groups[i % len(groups)], outcome=outcome,
            r_net=round(r_net, 4), teacher_decision=rule_decision(p, f["fee_pct_of_risk"]),
            teacher_p_win=p, teacher_reason=reason,
            teacher_seconds=round(rng.lognormvariate(math.log(1.3), 0.4), 2), **f))
    return cards
