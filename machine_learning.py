from sklearn.dummy import DummyClassifier
import numpy as np


class MachineLearning:

    @staticmethod
    def get_dummy_classifier(**kwargs) -> DummyClassifier:
        return DummyClassifier(**kwargs)

    @staticmethod
    def fit_dummy_classifier(dummy: DummyClassifier, feature: np.ndarray, target: np.ndarray, **kwargs):
        return dummy.fit(X=feature, y=target, **kwargs)

    @staticmethod
    def get_dummy_score(dummy: DummyClassifier, feature: np.ndarray, target: np.ndarray) -> float:
        return dummy.score(X=feature, y=target)