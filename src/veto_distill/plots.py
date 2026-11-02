"""Static figures for the report (PNG, light background, for a README)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
# colour follows the entity, never its rank
COLORS = {"teacher": "#2a78d6", "student": "#eb6834", "reference": "#8a8984",
          "outcome_tree": "#1baf7a"}


def _style(ax, title: str, ylabel: str = "", xlabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _fig(w: float = 7, h: float = 3.6):
    fig, ax = plt.subplots(figsize=(w, h), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _color(row: dict) -> str:
    return COLORS.get(row["name"], COLORS[row["kind"]])


def agreement_chart(res: dict, path: Path) -> None:
    rows = [r for r in res["rows"] if r["kind"] != "teacher"]
    fig, ax = _fig()
    x = np.arange(len(rows))
    ax.bar(x, [100 * r["agreement"] for r in rows], width=0.6, color=[_color(r) for r in rows])
    for xi, r in zip(x, rows):
        ax.text(xi, 100 * r["agreement"] + 1.5, f"{100 * r['agreement']:.0f}%\nκ {r['kappa']:.2f}",
                ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set_xticks(x, [r["name"] for r in rows])
    ax.set_ylim(0, 115)
    _style(ax, "How often each model makes the teacher's decision (test cards)",
           "agreement with teacher, %")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def r_net_chart(res: dict, path: Path) -> None:
    rows = [r for r in res["rows"] if r["r_net_mean"] is not None]
    fig, ax = _fig()
    x = np.arange(len(rows))
    vals = [r["r_net_mean"] for r in rows]
    ax.bar(x, vals, width=0.6, color=[_color(r) for r in rows])
    ax.axhline(0, color=INK_2, linewidth=0.8)
    for xi, r, v in zip(x, rows, vals):
        ax.text(xi, v + (0.01 if v >= 0 else -0.01), f"{v:+.3f}\n{r['n_take']} trades",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=8, color=INK_2)
    lo, hi = min(vals + [0]), max(vals + [0])
    pad = 0.35 * (hi - lo or 0.1)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xticks(x, [r["name"] for r in rows], rotation=0)
    _style(ax, "Mean R per TAKE trade after costs (test cards; 0 = break-even)",
           "R net per trade")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def cumulative_chart(res: dict, path: Path, student: str) -> None:
    s = res["series"]
    r = np.array(s["r_net"])
    fig, ax = _fig()
    ends = []
    for name, color in (("take_all", COLORS["reference"]), ("teacher", COLORS["teacher"]),
                        (student, COLORS["student"]), ("outcome_tree", COLORS["outcome_tree"])):
        take = np.array(s["take"][name])
        curve = np.cumsum(np.where(take, r, 0.0))
        ax.plot(np.arange(len(r)), curve, color=color, linewidth=2, label=name)
        ends.append((curve[-1], name))
    ends.sort()
    lo, hi = ax.get_ylim()
    gap = 0.06 * (hi - lo)  # keep end labels apart
    placed = []
    for y, name in ends:
        yy = max(y, placed[-1] + gap) if placed else y
        placed.append(yy)
        ax.text(len(r) * 1.01, yy, f"{name} {y:+.0f}R", fontsize=8, color=INK_2, va="center")
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_xlim(0, len(r) * 1.18)
    _style(ax, "Cumulative R of the TAKE trades, test cards in time order",
           "cumulative R net", "test card (time order)")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def calibration_chart(res: dict, path: Path) -> None:
    cal = res["calibration"]
    fig, ax = _fig(5, 4)
    if cal:
        xs = [b["p_mean"] for b in cal]
        ys = [b["win_rate"] for b in cal]
        hi = max(xs + ys) + 0.08
        ax.plot([0, hi], [0, hi], color=GRID, linewidth=1.5, label="perfect calibration")
        ax.plot(xs, ys, color=COLORS["teacher"], linewidth=2, marker="o", markersize=7,
                label="teacher")
        for b in cal:
            ax.annotate(f"n={b['n']}", (b["p_mean"], b["win_rate"]), xytext=(5, -10),
                        textcoords="offset points", fontsize=7, color=INK_2)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax, "Teacher p_win vs realised win rate", "realised win rate",
           "teacher p_win (bin mean)")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def pwin_scatter(res: dict, path: Path, name: str = "pwin_tree") -> bool:
    s = res["series"]
    if name not in s["student_p_win"]:
        return False
    t = np.array([np.nan if p is None else p for p in s["teacher_p_win"]], dtype=float)
    p = np.array(s["student_p_win"][name])
    mask = ~np.isnan(t)
    if mask.sum() == 0:
        return False
    fig, ax = _fig(5, 4)
    jitter = np.random.default_rng(0).uniform(-0.006, 0.006, mask.sum())
    ax.scatter(t[mask] + jitter, p[mask], s=12, color=COLORS["student"], alpha=0.45,
               edgecolors="none")
    lo, hi = min(t[mask].min(), p[mask].min()) - 0.02, max(t[mask].max(), p[mask].max()) + 0.02
    ax.plot([lo, hi], [lo, hi], color=INK_2, linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    _style(ax, f"p_win: teacher vs student ({name})", "student p_win", "teacher p_win")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return True


def all_figures(res: dict, out: Path, best_student: str) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    names = ["agreement.png", "r_net.png", "cumulative_r.png", "calibration.png"]
    agreement_chart(res, out / names[0])
    r_net_chart(res, out / names[1])
    cumulative_chart(res, out / names[2], best_student)
    calibration_chart(res, out / names[3])
    if pwin_scatter(res, out / "pwin_scatter.png"):
        names.append("pwin_scatter.png")
    return names
