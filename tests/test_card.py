from veto_distill.card import FEATURES, feature_row, session_of
from veto_distill.synthetic import make_dataset


def test_feature_row_matches_feature_names():
    c = make_dataset(5)[0]
    assert len(feature_row(c)) == len(FEATURES)


def test_card_text_does_not_depend_on_the_outcome():
    from dataclasses import replace

    for c in make_dataset(50):
        other = replace(c, outcome="win", r_net=2.0, text="")
        assert other.text == c.text
        assert "202" not in c.text  # no year, no date


def test_k_is_shown_to_one_decimal():
    c = make_dataset(1)[0]
    assert f"{c.k_prev:.1f} -> {c.k:.1f}" in c.text


def test_sessions():
    assert [session_of(h) for h in (0, 8, 13, 21, 23)] == ["Asia", "Europe", "US", "late US",
                                                           "late US"]
