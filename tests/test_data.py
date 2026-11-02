import pytest

from veto_distill import data
from veto_distill.synthetic import make_dataset


def test_save_and_load_round_trip(tmp_path):
    cards = make_dataset(20)
    path = tmp_path / "d.jsonl"
    data.save(cards, path)
    back = data.load(path)
    assert [c.to_dict() for c in back] == [c.to_dict() for c in sorted(cards, key=lambda c: (c.ts, c.key))]


def test_time_split_is_per_group_and_never_looks_ahead():
    cards = make_dataset(300, groups=("A", "B", "C"))
    train, test = data.time_split(cards, 0.6)
    assert len(train) + len(test) == len(cards)
    for g in "ABC":
        tr = [c.ts for c in train if c.group == g]
        te = [c.ts for c in test if c.group == g]
        assert tr and te
        assert max(tr) <= min(te)


def test_time_split_rejects_bad_share():
    with pytest.raises(ValueError):
        data.time_split(make_dataset(10), 1.0)


def test_public_sample_hides_dates_and_names_but_keeps_order_and_labels():
    cards = make_dataset(1000, groups=("ADA", "BTC"))
    sample = data.public_sample(cards, 500)
    assert 490 <= len(sample) <= 500
    assert {c.group for c in sample} == {"G1", "G2"}
    assert max(c.ts for c in sample) < 1_000_000  # bar index, not a date
    assert all("ADA" not in c.key and "BTC" not in c.key for c in sample)
    originals = {(c.k, c.k_prev, c.r_net) for c in cards}
    assert all((c.k, c.k_prev, c.r_net) in originals for c in sample)
    for g in ("G1", "G2"):  # still splittable in time order
        assert [c.ts for c in sample if c.group == g] == sorted(c.ts for c in sample if c.group == g)
