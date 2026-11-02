"""Datasets of teacher-labelled cards: JSONL on disk, split by time."""

from __future__ import annotations

import hashlib
import json
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


def time_split(cards: list[Card], train_share: float = 0.6) -> tuple[list[Card], list[Card]]:
    """Earliest `train_share` of each group (by time) to train, the rest to test.

    Splitting per group keeps both groups in both parts even when their periods differ,
    and every test card is later than every train card of its own group, so a student
    never learns from a card that comes after one it is scored on.
    """
    if not 0 < train_share < 1:
        raise ValueError(f"train_share must be between 0 and 1, not {train_share}")
    train, test = [], []
    for g in sorted({c.group for c in cards}):
        part = sorted((c for c in cards if c.group == g), key=lambda c: (c.ts, c.key))
        cut = int(len(part) * train_share)
        train += part[:cut]
        test += part[cut:]
    key = lambda c: (c.ts, c.key)  # noqa: E731
    return sorted(train, key=key), sorted(test, key=key)


def sha256(path: Path) -> str:
    """Fingerprint of a file, so a result can be tied to the exact snapshot it came from."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
