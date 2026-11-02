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
    train, test, purged = data.time_split(cards, 0.6)
    assert len(train) + len(test) + purged == len(cards)
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
    assert min(c.ts for c in sample) == 0  # offsets, not dates
    assert max(c.ts for c in sample) < 1_000_000_000
    assert all("ADA" not in c.key and "BTC" not in c.key for c in sample)
    originals = {(c.k, c.k_prev, c.r_net) for c in cards}
    assert all((c.k, c.k_prev, c.r_net) in originals for c in sample)
    for g in ("G1", "G2"):  # still splittable in time order
        assert [c.ts for c in sample if c.group == g] == sorted(c.ts for c in sample if c.group == g)


def test_purge_drops_training_cards_whose_outcome_reaches_the_test_period():
    from dataclasses import replace

    base = make_dataset(1)[0]
    hour = 3600
    # one group, a card every hour for 10 days
    cards = [replace(base, key=f"c{i}", ts=i * hour, group="A") for i in range(240)]
    train, test, purged = data.time_split(cards, 0.5)
    first_test = min(c.ts for c in test)
    assert first_test == 120 * hour
    assert max(c.ts for c in train) + data.PURGE_S < first_test
    assert purged == 24  # one day of hourly cards
    assert all(c.ts + data.HORIZON_S < first_test for c in train)


def test_purge_must_cover_the_outcome_horizon():
    with pytest.raises(ValueError):
        data.time_split(make_dataset(10), 0.6, purge_s=3600)
