"""Reproduce the SepWav pickle/unpickle AttributeError on a tiny Iris slice."""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from scikit_longitudinal.data_preparation.separate_waves import SepWav
from scikit_longitudinal.estimators.ensemble.longitudinal_voting.longitudinal_voting import (
    LongitudinalEnsemblingStrategy,
)


def main() -> None:
    iris = load_iris(as_frame=True)
    X = iris.data.iloc[:30].copy()
    y = iris.target.iloc[:30].copy()

    feature_list_names = list(X.columns)
    features_group = [[0, 1], [2, 3]]

    sepwav = SepWav(
        estimator=DecisionTreeClassifier(random_state=0, max_depth=2),
        features_group=features_group,
        feature_list_names=feature_list_names,
        voting=LongitudinalEnsemblingStrategy.MAJORITY_VOTING,
    )
    sepwav.fit(X.values, y.values)

    print("[smoke] fitted SepWav, predict on 3 rows:", sepwav.predict(X.values[:3]))
    print("[smoke] _extract_wave.__name__:", sepwav._extract_wave.__name__)
    print(
        "[smoke] clf_ensemble.extract_wave.__func__.__name__:",
        sepwav.clf_ensemble.extract_wave.__func__.__name__,
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pkl"
        with path.open("wb") as fh:
            pickle.dump(sepwav, fh)
        print(f"[smoke] dumped to {path}")

        with path.open("rb") as fh:
            try:
                loaded = pickle.load(fh)
                print("[smoke] LOAD OK (bug not reproduced):", loaded)
            except AttributeError as exc:
                print("[smoke] REPRODUCED AttributeError on pickle.load:", exc)
                raise


if __name__ == "__main__":
    main()
