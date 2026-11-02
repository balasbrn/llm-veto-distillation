"""The run report: metrics.json, REPORT.md and figures in one output folder."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from veto_distill import plots
from veto_distill.card import Card

NOTES = {
    "take_all": "take every signal (no veto)",
    "teacher": "the LLM's decisions",
    "majority": "student: always the teacher's most common answer",
    "logistic": "student: logistic regression on the card's numbers",
    "tree": "student: gradient-boosted tree (depth 3) on the card's numbers",
    "text": "student: char n-gram TF-IDF + logistic regression on the card text",
    "pwin_tree": "student: tree regressing the teacher's p_win, then the teacher's rule",
    "outcome_tree": "not a student: the same tree trained on real outcomes",
}


def best_student(res: dict) -> str:
    cands = [r for r in res["rows"] if r["kind"] == "student" and r["name"] != "majority"]
    return max(cands, key=lambda r: (r["kappa"], r["agreement"]))["name"]


def _f(x, fmt: str) -> str:
    return "–" if x is None else fmt.format(x)


def _date(ts: int) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


def _join_lines(res: dict) -> list[str]:
    jc = res.get("join_counts")
    if not jc:
        return []
    out = [f"**Import** (teacher run `{res.get('teacher_run')}`):", "",
           "| group | teacher lines | errors | answered signals | cards | joined |",
           "|---|---|---|---|---|---|"]
    out += [f"| {g} | {c.get('teacher_lines', '–')} | {c.get('teacher_errors', '–')} | "
            f"{c['answers']} | {c['cards']} | {c['joined']} |" for g, c in jc.items()]
    return out + [""]


def _cutoff_lines(res: dict) -> list[str]:
    cutoff = res.get("teacher_cutoff_ts")
    if cutoff is None:
        return []
    last = max(res["series"]["ts"])
    if last < cutoff:
        return [f"**Teacher cutoff.** Every card predates the teacher's training cutoff "
                f"({_date(cutoff)}), so the teacher may have seen these months during "
                "training. Read these numbers as a method demonstration, not as clean "
                "out-of-sample evidence.", ""]
    return [f"Teacher cutoff {_date(cutoff)}; some cards are later than it.", ""]


def to_markdown(res: dict, title: str, source: str, examples: list[Card],
                figures: list[str]) -> str:
    best = best_student(res)
    rows = {r["name"]: r for r in res["rows"]}
    t, b, a = rows["teacher"], rows[best], rows["take_all"]
    lines = [
        f"# {title}", "",
        f"Source: {source}", "",
        *([f"Dataset fingerprint (sha256): `{res['dataset_sha256']}`", ""]
          if res.get("dataset_sha256") else []),
        f"Cards: {res['n_cards']} teacher-labelled; students train on the earliest "
        f"{res['train_share']:.0%} of each group ({res['n_train']} cards, up to "
        f"{_date(res['train_to_ts'])}) and are scored on the rest ({res['n_test']} cards). "
        f"{res['n_purged']} training cards within one day of the split were purged, so no "
        f"training outcome (20 bars = 5 hours) overlaps the test period. "
        f"The teacher said TAKE on {res['teacher_take_share_train']:.0%} of training cards.", "",
        "| group | train | test |", "|---|---|---|",
        *[f"| {g} | {v['train']} | {v['test']} |" for g, v in res["split_by_group"].items()],
        "",
        f"**Small data.** With {res['n_test']} test cards, a mean R per trade carries a "
        "standard error of roughly ±0.05–0.15 R (shown as ± below); differences smaller "
        "than that are noise.", "",
        *_join_lines(res),
        *_cutoff_lines(res),
        "## Summary", "",
        f"- **Distillation:** the best student (`{best}`) agrees with the teacher on "
        f"{b['agreement']:.0%} of test cards (Cohen's κ {b['kappa']:.2f}); "
        f"always giving the teacher's most common answer agrees on "
        f"{rows['majority']['agreement']:.0%}.",
        f"- **Speed:** teacher median {_f(t.get('ms_per_card'), '{:,.0f}')} ms per card "
        f"(one API call); `{best}` {b['ms_per_card']:.2f} ms per card including training.",
        f"- **Usefulness:** taking every test signal earns {_f(a['r_net_mean'], '{:+.3f}')} R "
        f"per trade; the teacher's TAKE set {_f(t['r_net_mean'], '{:+.3f}')} R; `{best}` "
        f"{_f(b['r_net_mean'], '{:+.3f}')} R. "
        + ("The teacher beats taking everything, so there is something to distil."
           if (t["r_net_mean"] or -9) > (a["r_net_mean"] or -9)
           else "The teacher does not beat taking everything, so even a perfect copy "
                "inherits no edge."), "",
        "## All decision makers (test cards)", "",
        "| model | what it is | agreement | κ | TAKE | win rate | R net / trade | total R | ms / card |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in res["rows"]:
        agree = "–" if "agreement" not in r else f"{r['agreement']:.1%}"
        kappa = "–" if "kappa" not in r else f"{r['kappa']:.2f}"
        lines.append(
            f"| `{r['name']}` | {NOTES.get(r['name'], '')} | {agree} | {kappa} | "
            f"{r['n_take']} ({r['take_share']:.0%}) | {_f(r['win_rate'], '{:.1%}')} | "
            f"{_f(r['r_net_mean'], '{:+.3f}')} ± {_f(r['r_net_se'], '{:.3f}')} | {r['r_net_total']:+.1f} | "
            f"{_f(r.get('ms_per_card'), '{:,.2f}')} |")
    pw = rows.get("pwin_tree", {})
    if pw.get("pwin_mae") is not None:
        lines += ["", f"`pwin_tree` reproduces the teacher's p_win with a mean absolute error "
                      f"of {pw['pwin_mae']:.3f} and rank correlation {pw['pwin_spearman']:.2f}."]
    lines += ["", "Teacher ms / card is the API latency the teacher run logged; student ms / "
              "card is fit plus predict time on this machine divided by test cards, so it is "
              "an upper bound on inference time. R net ± is one standard error of the mean.", "",
              "## Per group (test cards)", "",
              "| group | cards | take all: R net / trade | teacher: TAKE | teacher: R net / trade |",
              "|---|---|---|---|---|"]
    for g, v in res["by_group"].items():
        lines.append(f"| {g} | {v['n']} | {_f(v['take_all']['r_net_mean'], '{:+.3f}')} | "
                     f"{v['teacher']['n_take']} | {_f(v['teacher']['r_net_mean'], '{:+.3f}')} |")
    lines += ["", "## Is the teacher calibrated?", "",
              "| teacher p_win (bin mean) | realised win rate | cards |", "|---|---|---|"]
    lines += [f"| {c['p_mean']:.3f} | {c['win_rate']:.1%} | {c['n']} |" for c in res["calibration"]]
    lines += ["", "## Figures", ""] + [f"![{f}](figures/{f})" for f in figures]
    lines += ["", "## Example cards and teacher answers", ""]
    for c in examples:
        p = "n/a" if c.teacher_p_win is None else f"{c.teacher_p_win:.2f}"
        lines += ["```", c.text, "```",
                  f"Teacher: **{c.teacher_decision}**, p_win {p}. _{c.teacher_reason}_  ",
                  f"Outcome (never shown to any model): {c.outcome}, R net {c.r_net:+.2f}.", ""]
    return "\n".join(lines) + "\n"


def write(res: dict, out: Path, title: str, source: str, examples: list[Card]) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    figures = plots.all_figures(res, out / "figures", best_student(res))
    metrics = {k: v for k, v in res.items() if k != "series"}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    path = out / "REPORT.md"
    path.write_text(to_markdown(res, title, source, examples, figures))
    return path
