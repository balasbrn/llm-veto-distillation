import json
import math

import pyarrow as pa
import pyarrow.parquet as pq

from veto_distill import cmf_import


def _row(ts, direction="LONG", **kw):
    row = {"direction": direction, "bar_start_ts": ts, "year": 2022, "outcome": "loss",
           "r_net": -1.1, "k": 12.34, "k_prev": 3.0, "atr_pct": 0.5, "atr_rank_30d": None,
           "fee_pct_of_risk": 8.0, "ema50_atr": 1.0, "ema200_atr": float("nan"),
           "ema50_above_200": None, "vwap_atr": -0.5, "range_pos_20": 40.0,
           "moves_4": [0.1, -0.2, 0.3, 0.0], "move_sum_4": 0.2, "rsi14": 30.0,
           "volume_ratio_20": 1.2, "utc_hour": 3, "session": "Asia", "text": "old text"}
    row.update(kw)
    return row


def test_join_keeps_answered_cards_and_skips_errors(tmp_path):
    cards = tmp_path / "cards.parquet"
    pq.write_table(pa.Table.from_pylist([_row(100), _row(200), _row(300, "SHORT", k=88.0,
                                                                     k_prev=97.0)]), cards)
    teacher = tmp_path / "t.jsonl"
    lines = [
        {"bar_start_ts": 100, "direction": "LONG", "decision": "DONT_TAKE", "p_win": 0.2,
         "reason": "first", "seconds": 1.0},
        {"bar_start_ts": 100, "direction": "LONG", "decision": "TAKE", "p_win": 0.5,
         "reason": "last wins", "seconds": 2.0},
        {"bar_start_ts": 200, "direction": "LONG", "error": "429"},
        {"bar_start_ts": 300, "direction": "SHORT", "decision": "TAKE", "p_win": 0.4,
         "reason": "ok", "seconds": 1.5},
        {"bar_start_ts": 999, "direction": "LONG", "decision": "TAKE", "p_win": 0.4},
    ]
    teacher.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    out, counts = cmf_import.import_files(cards, teacher, "ADA")
    assert counts == {"cards": 3, "answers": 3, "joined": 2, "answers_without_card": 1}
    first = out[0]
    assert first.key == "ADA-100-LONG" and first.teacher_decision == "TAKE"
    assert first.teacher_reason == "last wins" and first.teacher_seconds == 2.0
    assert first.ema200_atr is None  # NaN cleaned
    assert "K 3.0 -> 12.3" in first.text  # re-rendered, one decimal
    assert not any(isinstance(v, float) and math.isnan(v) for v in first.to_dict().values())


def test_default_paths(tmp_path):
    c, t = cmf_import.default_paths(tmp_path, "ADAUSD", "dev", "gemini-2.5-flash-v2-think0")
    assert c == tmp_path / "h16/cards/ADAUSD/dev/cards.parquet"
    assert t.name == "ADAUSD-dev-gemini-2.5-flash-v2-think0.jsonl"


def test_import_cli_writes_dataset_and_fingerprint(tmp_path):
    from veto_distill import cli

    run = "gemini-2.5-flash-v2-think0"
    cards, teacher = cmf_import.default_paths(tmp_path, "ADAUSD", "dev", run)
    cards.parent.mkdir(parents=True)
    teacher.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([_row(100)]), cards)
    teacher.write_text(json.dumps({"bar_start_ts": 100, "direction": "LONG",
                                   "decision": "TAKE", "p_win": 0.4}) + "\n")
    out = tmp_path / "ds.jsonl"
    assert cli.main(["import", "--data-root", str(tmp_path), "--source", "ADA:ADAUSD:dev",
                     "--out", str(out)]) == 0
    m = json.loads(out.with_suffix(".manifest.json").read_text())
    assert m["cards"] == 1 and len(m["dataset_sha256"]) == 64 and len(m["inputs"]) == 2
