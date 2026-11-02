"""Datasets of teacher-labelled cards: JSONL on disk, split by time."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from veto_distill.card import Card


def save(cards: list[Card], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for c in cards:
            f.write(json.dumps(c.to_dict()) + "\n")


def load(path: Path) -> list[Card]:
    cards = [Card(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
    return sorted(cards, key=lambda c: (c.ts, c.key))


HORIZON_S = 20 * 900  # an outcome runs 20 bars of 15 minutes past its card
PURGE_S = 86_400  # purge a full day at the split boundary (covers the 5-hour horizon)


def time_split(cards: list[Card], train_share: float = 0.6,
               purge_s: int = PURGE_S) -> tuple[list[Card], list[Card], int]:
    """Earliest `train_share` of each group (by time) to train, the rest to test, with a
    purge: a training card whose outcome window could reach the test period leaks test-time
    prices into training, so every training card with ``ts + purge_s >= first test ts`` of
    its group is dropped. Returns (train, test, number of purged cards).

    Splitting per group keeps both groups in both parts even when their periods differ.
    """
    if not 0 < train_share < 1:
        raise ValueError(f"train_share must be between 0 and 1, not {train_share}")
    if purge_s < HORIZON_S:
        raise ValueError(f"purge_s must cover the {HORIZON_S} s outcome horizon, not {purge_s}")
    train, test, purged = [], [], 0
    for g in sorted({c.group for c in cards}):
        part = sorted((c for c in cards if c.group == g), key=lambda c: (c.ts, c.key))
        cut = int(len(part) * train_share)
        if cut == len(part):
            train += part
            continue
        first_test = part[cut].ts
        kept = [c for c in part[:cut] if c.ts + purge_s < first_test]
        purged += cut - len(kept)
        train += kept
        test += part[cut:]
    key = lambda c: (c.ts, c.key)  # noqa: E731
    return sorted(train, key=key), sorted(test, key=key), purged


def sha256(path: Path) -> str:
    """Fingerprint of a file, so a result can be tied to the exact snapshot it came from."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def public_sample(cards: list[Card], n: int = 500) -> list[Card]:
    """An anonymous sample: n cards spread evenly over time in each group, timestamps
    shifted to seconds since the group's first card (order and gaps kept so the purge still
    works, calendar dates gone), keys and group names replaced by neutral ones. Card fields,
    teacher answers and outcomes stay. Publishing it needs the owner's go-ahead."""
    groups = sorted({c.group for c in cards})
    names = {g: f"G{i + 1}" for i, g in enumerate(groups)}
    out = []
    for g in groups:
        part = sorted((c for c in cards if c.group == g), key=lambda c: (c.ts, c.key))
        k = min(len(part), n * len(part) // len(cards) or 1)
        picks = [part[i * len(part) // k] for i in range(k)]
        start = part[0].ts
        for j, c in enumerate(picks):
            out.append(replace(c, key=f"{names[g]}-{j:04d}", group=names[g], ts=c.ts - start,
                               text=""))
    return sorted(out, key=lambda c: (c.ts, c.key))
