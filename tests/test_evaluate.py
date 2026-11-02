import json

import numpy as np

from veto_distill import cli, evaluate
from veto_distill.synthetic import make_dataset


def test_agreement_and_kappa():
    t = np.array([True, True, False, False])
    assert evaluate.agreement(t, t) == {"agreement": 1.0, "kappa": 1.0}
    assert evaluate.agreement(~t, t)["kappa"] == -1.0
    assert evaluate.agreement(np.ones(4, bool), t)["kappa"] == 0.0


def test_trading_counts_only_taken_cards():
    cards = make_dataset(4)
    take = np.array([True, False, True, False])
    out = evaluate.trading(take, cards)
    assert out["n_take"] == 2
    assert out["r_net_total"] == cards[0].r_net + cards[2].r_net
    assert evaluate.trading(np.zeros(4, bool), cards)["r_net_mean"] is None


def test_run_scores_every_model_and_the_teacher_agrees_with_itself():
    res = evaluate.run(make_dataset(400))
    names = [r["name"] for r in res["rows"]]
    assert names == ["take_all", "teacher", "majority", "logistic", "tree", "text",
                     "pwin_tree", "outcome_tree"]
    teacher = next(r for r in res["rows"] if r["name"] == "teacher")
    assert teacher["agreement"] == 1.0
    assert res["n_train"] + res["n_test"] + res["n_purged"] == 400
    assert res["n_purged"] > 0
    for r in res["rows"]:
        assert len(res["series"]["take"][r["name"]]) == res["n_test"]


def test_students_learn_the_synthetic_teacher_better_than_chance():
    res = evaluate.run(make_dataset(1200))
    best = max(r["kappa"] for r in res["rows"] if r["kind"] == "student")
    assert best > 0.2


def test_demo_cli_writes_report_metrics_and_figures(tmp_path):
    assert cli.main(["demo", "--n", "300", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "REPORT.md").read_text().startswith("# Distillation run")
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert "series" not in m and m["n_cards"] == 300
    assert (tmp_path / "figures" / "cumulative_r.png").stat().st_size > 1000


def test_run_cli_on_saved_data(tmp_path):
    data = tmp_path / "d.jsonl"
    assert cli.main(["demo", "--n", "200", "--out", str(tmp_path / "a"),
                     "--save-data", str(data)]) == 0
    assert cli.main(["run", "--data", str(data), "--out", str(tmp_path / "b"),
                     "--teacher-cutoff", "2025-01-31"]) == 0
    text = (tmp_path / "b" / "REPORT.md").read_text()
    assert "method demonstration" in text and "purged" in text


def test_paired_diff_is_zero_for_take_all_and_detects_a_planted_edge():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 1, 2000)
    assert evaluate.paired_diff(np.ones(2000, bool), r)["diff_vs_take_all"] == 0
    good = r > 0.5  # a filter that picks winners must be far beyond 2 SE
    out = evaluate.paired_diff(good, r, boot=300)
    assert out["diff_z"] > 10
    noise = rng.random(2000) < 0.5  # a random filter must not be
    assert abs(evaluate.paired_diff(noise, r, boot=300)["diff_z"]) < 3


def test_report_says_no_measurable_difference_on_noise(tmp_path):
    assert cli.main(["demo", "--n", "400", "--out", str(tmp_path)]) == 0
    text = (tmp_path / "REPORT.md").read_text()
    assert "Cohen's κ" in text.split("## All decision makers")[0]
    assert "paired" in text and "Calibration" in text
    assert "beats taking everything" not in text
