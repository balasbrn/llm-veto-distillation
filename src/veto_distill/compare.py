"""Card-by-card comparison of one model's vetoes against the teacher, for one run or two.

Reads the ``per_card.jsonl`` that ``vd run`` writes. With two runs (same cards, e.g. the
full tree and a tree without some features) it counts, separately on the teacher's veto
cards and on its allow cards, how many cards the change fixed and how many it broke, with
an exact two-sided sign (McNemar) test on each pair. Agreement comes first; trading
results follow in their own table.
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact sign test on the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def vetoes(rows: list[dict], model: str) -> dict:
    veto = [r for r in rows if r["teacher_decision"] != "TAKE"]
    allow = [r for r in rows if r["teacher_decision"] == "TAKE"]
    return {"cards": len(rows), "teacher_veto": len(veto), "teacher_allow": len(allow),
            "missed_vetoes": sum(r["take"][model] for r in veto),
            "extra_vetoes": sum(not r["take"][model] for r in allow)}


def fixed_broke(a: list[dict], b: list[dict], model: str) -> dict:
    """On teacher-veto cards a fix is A TAKE -> B veto; on teacher-allow cards a fix is
    A veto -> B TAKE. A break is the reverse."""
    out = {}
    for name, want_take in (("teacher_veto", False), ("teacher_allow", True)):
        pairs = [(x["take"][model], y["take"][model]) for x, y in zip(a, b)
                 if (x["teacher_decision"] == "TAKE") == want_take]
        fixed = sum(xa != want_take and yb == want_take for xa, yb in pairs)
        broke = sum(xa == want_take and yb != want_take for xa, yb in pairs)
        out[name] = {"cards": len(pairs), "fixed": fixed, "broke": broke,
                     "mcnemar_p": mcnemar_exact(fixed, broke)}
    return out


def trading(rows: list[dict], take) -> tuple[int, float | None, float | None]:
    sel = [r for r in rows if take(r)]
    if not sel:
        return 0, None, None
    return (len(sel), sum(r["outcome"] == "win" for r in sel) / len(sel),
            sum(r["r_net"] for r in sel) / len(sel))


def _check_same_cards(a: list[dict], b: list[dict]) -> None:
    if [r["key"] for r in a] != [r["key"] for r in b]:
        raise ValueError("the two runs do not score the same cards in the same order")
    if [r["teacher_decision"] for r in a] != [r["teacher_decision"] for r in b]:
        raise ValueError("the two runs disagree on the teacher's decisions")


def to_markdown(path_a: Path, path_b: Path | None = None, model: str = "tree") -> str:
    a = load(path_a)
    runs = [("A", path_a, a)]
    if path_b is not None:
        b = load(path_b)
        _check_same_cards(a, b)
        runs.append(("B", path_b, b))
    lines = [f"# Vetoes of `{model}` vs the teacher", ""]
    lines += [f"- {n}: `{p}`" for n, p, _ in runs] + ["",
              "## Agreement", "",
              "| run | cards | teacher vetoes | teacher allows | missed vetoes (model TAKE, "
              "teacher DONT_TAKE) | extra vetoes (model DONT_TAKE, teacher TAKE) |",
              "|---|---|---|---|---|---|"]
    for n, _, rows in runs:
        v = vetoes(rows, model)
        lines.append(f"| {n} | {v['cards']} | {v['teacher_veto']} | {v['teacher_allow']} | "
                     f"{v['missed_vetoes']} of {v['teacher_veto']} | "
                     f"{v['extra_vetoes']} of {v['teacher_allow']} |")
    if path_b is not None:
        fb = fixed_broke(a, b, model)
        lines += ["", "## B against A, card by card", "",
                  "| cards | count | fixed by B | broken by B | exact sign test p (two-sided) |",
                  "|---|---|---|---|---|"]
        for name, label in (("teacher_veto", "teacher vetoes"), ("teacher_allow", "teacher allows")):
            x = fb[name]
            lines.append(f"| {label} | {x['cards']} | {x['fixed']} | {x['broke']} | "
                         f"{x['mcnemar_p']:.3f} |")
        lines += ["", "On teacher vetoes, fixed = A took the trade and B vetoes it; on teacher "
                      "allows, fixed = A vetoed and B takes it. Broken is the reverse."]
    lines += ["", "## Trading (separate from agreement)", "",
              "| run | decision maker | TAKE | win rate | mean R net |", "|---|---|---|---|---|"]

    def row(n, label, t):
        k, w, r = t
        ws = "–" if w is None else f"{w:.1%}"
        rs = "–" if r is None else f"{r:+.3f}"
        return f"| {n} | {label} | {k} | {ws} | {rs} |"

    lines.append(row("A", "take every card", trading(a, lambda r: True)))
    lines.append(row("A", "teacher", trading(a, lambda r: r["teacher_decision"] == "TAKE")))
    for n, _, rows in runs:
        lines.append(row(n, f"`{model}`", trading(rows, lambda r: r["take"][model])))
    return "\n".join(lines) + "\n"
