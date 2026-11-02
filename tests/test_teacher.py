from veto_distill.teacher import parse, rule_decision


def test_parse_accepts_json_inside_text():
    assert parse('sure: {"p_win": 0.42, "reason": "trend"}') == {"p_win": 0.42, "reason": "trend"}


def test_parse_rejects_bad_answers():
    assert parse("no json") is None
    assert parse('{"p_win": 1.5}') is None
    assert parse('{"p_win": "high"}') is None
    assert parse('{"p_win": 0.3') is None


def test_rule_decision_breaks_even_at_one_third_plus_fee():
    # 2p - (1 - p) - fee > 0  <=>  p > (1 + fee) / 3
    assert rule_decision(0.34, 0) == "TAKE"
    assert rule_decision(0.33, 0) == "DONT_TAKE"
    assert rule_decision(0.366, 10) == "DONT_TAKE"
    assert rule_decision(0.367, 10) == "TAKE"
