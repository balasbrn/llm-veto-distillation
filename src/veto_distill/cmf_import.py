"""Import teacher-labelled cards from the upstream research repo's outputs.

Upstream (a private research repo) writes, per coin and period:

- ``cards.parquet``: one row per signal with the card fields, the outcome
  (win / loss / timeout) and R net after fees and funding;
- ``<pair>-<period>-<model>-<prompt>-think<N>.jsonl``: one line per teacher call with
  ``bar_start_ts``, ``direction``, ``decision``, ``p_win``, ``reason``, ``seconds``
  (or ``error`` when the call failed).

This module joins the two on (bar_start_ts, direction) into this repo's dataset format.
Only cards the teacher answered are kept; the counts of the rest are returned.
Neither file is committed here: they stay where the upstream pipeline wrote them.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from veto_distill.card import Card


def _clean(x):
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def read_cards(path: Path) -> list[dict]:
    import pyarrow.parquet as pq  # optional dependency, only for the import

    return pq.read_table(path).to_pylist()


def read_teacher(path: Path, stats: dict | None = None) -> dict[tuple[int, str], dict]:
    """Last valid answer per signal; failed calls are skipped and counted in `stats`."""
    out: dict[tuple[int, str], dict] = {}
    stats = {} if stats is None else stats
    lines = [x for x in path.read_text().splitlines() if x.strip()]
    stats["teacher_lines"] = len(lines)
    stats["teacher_errors"] = 0
    for line in lines:
        r = json.loads(line)
        stats["teacher_errors"] += "error" in r
        if "error" in r or r.get("decision") not in ("TAKE", "DONT_TAKE"):
            continue
        out[(int(r["bar_start_ts"]), r["direction"])] = r
    return out


def join(card_rows: list[dict], answers: dict[tuple[int, str], dict],
         group: str, stats: dict | None = None) -> tuple[list[Card], dict]:
    cards = []
    for row in card_rows:
        key = (int(row["bar_start_ts"]), row["direction"])
        a = answers.get(key)
        if a is None:
            continue
        cards.append(Card(
            key=f"{group}-{key[0]}-{key[1]}", ts=key[0], group=group, direction=key[1],
            k=float(row["k"]), k_prev=float(row["k_prev"]), atr_pct=float(row["atr_pct"]),
            atr_rank_30d=_clean(row.get("atr_rank_30d")),
            fee_pct_of_risk=float(row["fee_pct_of_risk"]),
            ema50_atr=_clean(row.get("ema50_atr")), ema200_atr=_clean(row.get("ema200_atr")),
            ema50_above_200=_clean(row.get("ema50_above_200")),
            vwap_atr=_clean(row.get("vwap_atr")), range_pos_20=float(row["range_pos_20"]),
            moves_4=[float(m) for m in row["moves_4"]], rsi14=_clean(row.get("rsi14")),
            volume_ratio_20=_clean(row.get("volume_ratio_20")),
            utc_hour=int(row["utc_hour"]), outcome=row["outcome"], r_net=float(row["r_net"]),
            teacher_decision=a["decision"],
            teacher_p_win=None if a.get("p_win") is None else float(a["p_win"]),
            teacher_reason=str(a.get("reason", "")),
            teacher_seconds=None if a.get("seconds") is None else float(a["seconds"]),
            # text left empty: re-rendered from the fields, K to one decimal, which is
            # the text the teacher saw (the stored text column rounds K to integers)
        ))
    matched = {(c.ts, c.direction) for c in cards}
    counts = {**(stats or {}), "cards": len(card_rows), "answers": len(answers),
              "joined": len(cards), "answers_without_card": len(set(answers) - matched)}
    return cards, counts


def import_files(cards_path: Path, teacher_path: Path, group: str) -> tuple[list[Card], dict]:
    stats: dict = {}
    answers = read_teacher(teacher_path, stats)
    return join(read_cards(cards_path), answers, group, stats)


def default_paths(data_root: Path, pair: str, period: str, run: str) -> tuple[Path, Path]:
    """Upstream layout: h16/cards/<pair>/<period>/cards.parquet and
    h16/teacher/<pair>-<period>-<run>.jsonl, e.g. run = gemini-2.5-flash-v2-think0."""
    return (data_root / "h16" / "cards" / pair / period / "cards.parquet",
            data_root / "h16" / "teacher" / f"{pair}-{period}-{run}.jsonl")
