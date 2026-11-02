"""Students: small, fast models trained to copy the teacher's decisions.

Classifier students weight both answers equally (``class_weight="balanced"``) so a
lopsided teacher (e.g. TAKE on 87% of cards) does not collapse them into "always
TAKE"; that is what the ``majority`` student shows. Each student is fitted on the training cards and returns, for the test cards, a TAKE
flag and (when it has one) an estimated p_win. Students copy the teacher; the
``outcome_tree`` reference instead learns from the real outcomes, to show what the
same model gets from the ground truth rather than from the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from veto_distill.card import Card, feature_row
from veto_distill.teacher import TP_R

SEED = 13


@dataclass
class Prediction:
    take: np.ndarray  # bool per test card
    p_win: np.ndarray | None = None  # estimated p_win per test card, if the model has one


def matrix(cards: list[Card]) -> np.ndarray:
    return np.array([feature_row(c) for c in cards], dtype=float)


def teacher_take(cards: list[Card]) -> np.ndarray:
    return np.array([c.teacher_take for c in cards])


def _fees(cards: list[Card]) -> np.ndarray:
    return np.array([c.fee_pct_of_risk for c in cards]) / 100


def _rule(p: np.ndarray, cards: list[Card]) -> np.ndarray:
    """The same decision rule the teacher's p_win went through: expected R net > 0."""
    return TP_R * p - (1 - p) - _fees(cards) > 0


def majority(train: list[Card], test: list[Card]) -> Prediction:
    """Always gives the teacher's most common decision on the training cards."""
    take = teacher_take(train).mean() >= 0.5
    return Prediction(np.full(len(test), take))


def _classifier(model, train_x, test_x, train: list[Card], test: list[Card]) -> Prediction:
    y = teacher_take(train)
    if len(set(y)) < 2:  # the teacher gave one answer only: nothing to learn
        return majority(train, test)
    model.fit(train_x, y)
    p_take = model.predict_proba(test_x)[:, 1]
    return Prediction(p_take >= 0.5)


def logistic(train: list[Card], test: list[Card]) -> Prediction:
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"))
    return _classifier(model, matrix(train), matrix(test), train, test)


def tree(train: list[Card], test: list[Card]) -> Prediction:
    model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                           class_weight="balanced", random_state=SEED)
    return _classifier(model, matrix(train), matrix(test), train, test)


def text(train: list[Card], test: list[Card]) -> Prediction:
    """Reads the same card text the LLM read (character n-grams), no hand-made features."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    if len({c.teacher_take for c in train}) < 2:
        return majority(train, test)
    x_train = vec.fit_transform([c.text for c in train])
    model = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    return _classifier(model, x_train, vec.transform([c.text for c in test]), train, test)


def pwin_tree(train: list[Card], test: list[Card]) -> Prediction:
    """Regresses the teacher's p_win, then applies the teacher's own decision rule."""
    known = [c for c in train if c.teacher_p_win is not None]
    if len(known) < 10:
        return majority(train, test)
    model = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200,
                                          random_state=SEED)
    model.fit(matrix(known), [c.teacher_p_win for c in known])
    p = np.clip(model.predict(matrix(test)), 0, 1)
    return Prediction(_rule(p, test), p)


def outcome_tree(train: list[Card], test: list[Card]) -> Prediction:
    """Not a student: the same tree trained on real outcomes (win or not), no teacher."""
    y = np.array([c.outcome == "win" for c in train])
    if len(set(y)) < 2:
        return Prediction(np.zeros(len(test), dtype=bool))
    model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                           random_state=SEED)
    model.fit(matrix(train), y)
    p = model.predict_proba(matrix(test))[:, 1]
    return Prediction(_rule(p, test), p)


# name, function, whether it copies the teacher (a student) or not (a reference)
STUDENTS: list[tuple[str, Callable[[list[Card], list[Card]], Prediction]]] = [
    ("majority", majority),
    ("logistic", logistic),
    ("tree", tree),
    ("text", text),
    ("pwin_tree", pwin_tree),
]
REFERENCES: list[tuple[str, Callable[[list[Card], list[Card]], Prediction]]] = [
    ("outcome_tree", outcome_tree),
]
