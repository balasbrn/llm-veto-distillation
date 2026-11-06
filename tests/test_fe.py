import json

import numpy as np
import pytest

from veto_distill import cli, compare, data, evaluate, students
from veto_distill.card import FEATURES
from veto_distill.synthetic import make_dataset


def _run(tmp_path, name, *extra, n="600"):
    d = tmp_path / "d.jsonl"
    if not d.exists():
        data.save(make_dataset(int(n)), d)
    out = tmp_path / name
    assert cli.main(["run", "--data", str(d), "--out", str(out), *extra]) == 0
    return out


def test_per_card_file_keeps_trading_apart_from_agreement(tmp_path):
    out = _run(tmp_path, "ref")
    rows = [json.loads(x) for x in (out / "per_card.jsonl").read_text().splitlines()]
    m = json.loads((out / "metrics.json").read_text())
    assert len(rows) == m["n_test"]
    r = rows[0]
    assert set(r) == {"key", "group", "ts", "teacher_decision", "teacher_p_win", "take",
                      "p_win", "outcome", "r_net"}
    assert {"tree", "teacher", "take_all", "outcome_tree"} <= set(r["take"])
    assert "pwin_tree" in r["p_win"]
    assert set(m["versions"]) == {"python", "numpy", "scikit-learn"}


def test_drop_features_changes_only_the_tabular_inputs(tmp_path):
    cards = make_dataset(300)
    students.set_dropped(("atr_pct", "atr_rank_30d", "fee_pct_of_risk"))
    try:
        assert students.matrix(cards).shape == (300, len(FEATURES) - 3)
        assert "atr_pct" not in students.used_features()
    finally:
        students.set_dropped(())
    assert students.matrix(cards).shape == (300, len(FEATURES))
    with pytest.raises(ValueError):
        students.set_dropped(("no_such_feature",))
    res = evaluate.run(cards, drop=("atr_pct",))
    assert res["dropped_features"] == ["atr_pct"]
    assert students.matrix(cards).shape[1] == len(FEATURES)  # restored after the run


def test_same_run_twice_is_identical(tmp_path):
    a = _run(tmp_path, "a")
    b = _run(tmp_path, "b")
    assert (a / "per_card.jsonl").read_text() == (b / "per_card.jsonl").read_text()


def test_date_split_purges_one_day_before_the_cutoff():
    cards = make_dataset(400)
    cut = sorted(c.ts for c in cards)[250]
    train, test, purged = data.date_split(cards, cut)
    assert all(c.ts + data.PURGE_S < cut for c in train)
    assert all(c.ts >= cut for c in test)
    assert len(train) + len(test) + purged == len(cards) and purged > 0


def test_cutoff_flag_uses_one_shared_date(tmp_path):
    out = _run(tmp_path, "cut", "--cutoff", "2022-01-25")
    m = json.loads((out / "metrics.json").read_text())
    assert m["split"] == "date" and m["test_from_ts"] >= m["cutoff_ts"]
    assert m["train_to_ts"] + 86400 < m["cutoff_ts"]
    assert "one shared date split" in (out / "REPORT.md").read_text()


def test_compare_counts_fixed_and_broken_cards(tmp_path):
    def row(k, teacher, take):
        return {"key": k, "group": "A", "ts": k, "teacher_decision": teacher,
                "teacher_p_win": 0.4, "take": {"tree": take}, "p_win": {},
                "outcome": "loss", "r_net": -1.0}

    a = [row(1, "DONT_TAKE", True), row(2, "DONT_TAKE", False), row(3, "TAKE", False),
         row(4, "TAKE", True)]
    b = [row(1, "DONT_TAKE", False), row(2, "DONT_TAKE", True), row(3, "TAKE", True),
         row(4, "TAKE", True)]
    v = compare.vetoes(a, "tree")
    assert (v["missed_vetoes"], v["extra_vetoes"]) == (1, 1)
    fb = compare.fixed_broke(a, b, "tree")
    assert (fb["teacher_veto"]["fixed"], fb["teacher_veto"]["broke"]) == (1, 1)
    assert (fb["teacher_allow"]["fixed"], fb["teacher_allow"]["broke"]) == (1, 0)
    with pytest.raises(ValueError):
        compare._check_same_cards(a, b[::-1])


def test_mcnemar_exact():
    assert compare.mcnemar_exact(0, 0) == 1.0
    assert compare.mcnemar_exact(5, 5) == 1.0
    assert abs(compare.mcnemar_exact(0, 6) - 2 / 64) < 1e-12
    assert compare.mcnemar_exact(10, 0) < 0.01


def test_compare_cli_on_two_runs(tmp_path):
    a = _run(tmp_path, "full")
    b = _run(tmp_path, "fe1", "--drop-features", "atr_pct,atr_rank_30d,fee_pct_of_risk")
    out = tmp_path / "cmp.md"
    assert cli.main(["compare", str(a / "per_card.jsonl"), str(b / "per_card.jsonl"),
                     "--out", str(out)]) == 0
    text = out.read_text()
    assert "fixed by B" in text and text.index("## Agreement") < text.index("## Trading")
