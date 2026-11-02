"""The teacher: an LLM that reads a card and estimates the chance of a win.

The real labels in this project came from ``gemini-2.5-flash`` (temperature 0, JSON
answers) run upstream. This module keeps the prompt, the answer parser and the decision
rule so the labelling step is visible and re-runnable; ``label_with_gemini`` needs the
optional ``google-genai`` package and your own Google Cloud project (read from the
``GOOGLE_CLOUD_PROJECT`` environment variable). No key or project is stored here.
"""

from __future__ import annotations

import json
import os
import re
import time

TP_R = 2.0  # take-profit in R: 2 ATR target over a 1 ATR stop

# Prompt "v2": the model only estimates p_win; a fixed rule turns it into a decision.
PROMPT_V2 = """\
You estimate outcomes of trade signals for an account in a prop-firm challenge.

You see one signal card: a StochRSI turn on 15-minute crypto bars, described only in
relative terms (no coin, date or price). The trade plan is fixed: entry at the signal
bar's close, stop 1 ATR, take-profit 2 ATR, exit after 20 bars otherwise.

Estimate p_win: the probability that price reaches the take-profit before the stop.
Give your honest, calibrated estimate to two decimals, using everything on the card;
different cards should get different estimates when the evidence differs. Do not decide
whether to trade; a fixed rule does that from your estimate and the fee.

Reply with JSON only: {"p_win": number from 0 to 1, "reason": "at most 40 words"}."""


def rule_decision(p_win: float, fee_pct_of_risk: float) -> str:
    """TAKE when the expected R after fees is positive: 2 p - (1 - p) - fee in R > 0."""
    return "TAKE" if TP_R * p_win - (1 - p_win) - fee_pct_of_risk / 100 > 0 else "DONT_TAKE"


def parse(text: str) -> dict | None:
    """``{"p_win": float, "reason": str}`` from a model reply, or None if unusable."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        p = float(obj.get("p_win"))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not 0 <= p <= 1:
        return None
    return {"p_win": p, "reason": str(obj.get("reason", ""))[:400]}


def label_with_gemini(card_text: str, fee_pct_of_risk: float,
                      model: str = "gemini-2.5-flash", client=None) -> dict:
    """One teacher call. Returns decision, p_win, reason and latency, or an error.
    A failed call counts as DONT_TAKE downstream (no answer = no trade)."""
    if client is None:
        from google import genai  # pip install -e '.[teacher]'

        client = genai.Client(vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"],
                              location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))
    from google.genai import types

    config = types.GenerateContentConfig(
        system_instruction=PROMPT_V2, temperature=0.0, response_mime_type="application/json",
        max_output_tokens=512, thinking_config=types.ThinkingConfig(thinking_budget=0))
    start = time.monotonic()
    try:
        resp = client.models.generate_content(model=model, contents=card_text, config=config)
    except Exception as exc:  # network, quota, safety: recorded, never raised
        return {"error": f"{type(exc).__name__}: {exc}"[:400]}
    ans = parse(getattr(resp, "text", "") or "")
    seconds = round(time.monotonic() - start, 2)
    if ans is None:
        return {"error": "unparsable answer", "seconds": seconds}
    return {**ans, "decision": rule_decision(ans["p_win"], fee_pct_of_risk), "seconds": seconds}
