"""Scoring: how well each student copies the teacher, and what its decisions earn.

Two questions, kept apart:

1. **Distillation:** does the student make the teacher's decisions? (agreement, Cohen's
   kappa, p_win error and rank correlation, speed)
2. **Usefulness:** do the TAKE trades earn anything? (win rate and mean R net of the
   TAKE set vs taking every signal). A student can copy the teacher perfectly and still
   lose money, if the teacher does.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import cohen_kappa_score

from veto_distill import students as st
from veto_distill.card import Card
from veto_distill.data import time_split


def agreement(student_take: np.ndarray, teacher_take: np.ndarray) -> dict:
    agree = float(np.mean(student_take == teacher_take)) if len(teacher_take) else float("nan")
    one_class = len(set(student_take.tolist()) | set(teacher_take.tolist())) < 2
    kappa = 1.0 if one_class else float(cohen_kappa_score(teacher_take, student_take))
    return {"agreement": agree, "kappa": kappa}


def pwin_fit(student_p: np.ndarray, teacher_p: np.ndarray) -> dict:
    mask = ~np.isnan(teacher_p)
    if mask.sum() < 3:
        return {"pwin_mae": None, "pwin_spearman": None}
    s, t = student_p[mask], teacher_p[mask]
    rank = lambda x: np.argsort(np.argsort(x))  # noqa: E731
    rho = np.corrcoef(rank(s), rank(t))[0, 1] if s.std() > 0 and t.std() > 0 else float("nan")
    return {"pwin_mae": float(np.mean(np.abs(s - t))), "pwin_spearman": float(rho)}


def trading(take: np.ndarray, cards: list[Card]) -> dict:
    taken = [c for c, t in zip(cards, take) if t]
    if not taken:
        return {"n_take": 0, "take_share": 0.0, "win_rate": None, "r_net_mean": None,
                "r_net_total": 0.0, "r_net_se": None}
    r = np.array([c.r_net for c in taken])
    return {"n_take": len(taken), "take_share": len(taken) / len(cards),
            "win_rate": float(np.mean([c.outcome == "win" for c in taken])),
            "r_net_mean": float(r.mean()), "r_net_total": float(r.sum()),
            "r_net_se": float(r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else None}


def calibration(cards: list[Card], bins: int = 6) -> list[dict]:
    """Teacher p_win in equal-count bins vs the realised win rate."""
    cs = sorted((c for c in cards if c.teacher_p_win is not None), key=lambda c: c.teacher_p_win)
    out = []
    for b in range(bins):
        part = cs[b * len(cs) // bins:(b + 1) * len(cs) // bins]
        if part:
            out.append({"p_mean": float(np.mean([c.teacher_p_win for c in part])),
                        "win_rate": float(np.mean([c.outcome == "win" for c in part])),
                        "n": len(part)})
    return out


def run(cards: list[Card], train_share: float = 0.6) -> dict:
    """Fit every student on the earlier cards, score all of them on the later ones."""
    train, test, purged = time_split(cards, train_share)
    if not train or not test:
        raise ValueError(f"need cards in both parts; got {len(train)} train, {len(test)} test")
    t_take = st.teacher_take(test)
    t_p = np.array([np.nan if c.teacher_p_win is None else c.teacher_p_win for c in test])
    secs = [c.teacher_seconds for c in test if c.teacher_seconds is not None]
    all_take = np.ones(len(test), bool)
    rows = [{"name": "take_all", "kind": "reference", **agreement(all_take, t_take),
             **trading(all_take, test)},
            {"name": "teacher", "kind": "teacher", **agreement(t_take, t_take),
             **trading(t_take, test),
             "ms_per_card": 1000 * float(np.median(secs)) if secs else None}]
    takes = {"take_all": [True] * len(test), "teacher": t_take.tolist()}
    student_p = {}
    for kind, makers in (("student", st.STUDENTS), ("reference", st.REFERENCES)):
        for name, fn in makers:
            start = time.perf_counter()
            pred = fn(train, test)
            elapsed = time.perf_counter() - start
            row = {"name": name, "kind": kind, **agreement(pred.take, t_take),
                   **trading(pred.take, test),
                   "ms_per_card": 1000 * elapsed / len(test)}  # fit + predict, upper bound
            if pred.p_win is not None:
                row.update(pwin_fit(pred.p_win, t_p))
                student_p[name] = pred.p_win.tolist()
            rows.append(row)
            takes[name] = pred.take.tolist()
    by_group = {}
    for g in sorted({c.group for c in test}):
        idx = [i for i, c in enumerate(test) if c.group == g]
        sub = [test[i] for i in idx]
        by_group[g] = {"n": len(sub), "take_all": trading(np.ones(len(sub), bool), sub),
                       "teacher": trading(t_take[idx], sub)}
    teacher_gain = _gain(rows, "teacher")
    for r in rows:
        if r["kind"] == "student":
            g = _gain(rows, r["name"])
            r["share_of_teacher_gain"] = (g / teacher_gain if teacher_gain and teacher_gain > 0
                                          and g is not None else None)
    return {
        "n_cards": len(cards), "n_train": len(train), "n_test": len(test),
        "n_purged": purged,
        "split_by_group": {g: {"train": sum(c.group == g for c in train),
                               "test": sum(c.group == g for c in test)}
                           for g in sorted({c.group for c in cards})},
        "train_share": train_share,
        "test_from_ts": min(c.ts for c in test), "train_to_ts": max(c.ts for c in train),
        "teacher_take_share_train": float(st.teacher_take(train).mean()),
        "rows": rows, "by_group": by_group,
        "calibration": calibration(cards),
        "teacher_gain_r_per_trade": teacher_gain,
        # per test card, for the plots (not part of metrics.json)
        "series": {"ts": [c.ts for c in test], "r_net": [c.r_net for c in test],
                   "teacher_p_win": [c.teacher_p_win for c in test],
                   "take": takes, "student_p_win": student_p},
    }


def _gain(rows: list[dict], name: str) -> float | None:
    """Mean R net of a row's TAKE set minus taking every signal."""
    base = next(r for r in rows if r["name"] == "take_all")["r_net_mean"]
    row = next(r for r in rows if r["name"] == name)
    if row["r_net_mean"] is None or base is None:
        return None
    return row["r_net_mean"] - base
