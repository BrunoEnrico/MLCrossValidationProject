from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, KFold, StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from typing import Any
import numpy as np
from data_processing import DataProcessing


class MachineLearning:

    @staticmethod
    def get_dummy_classifier(**kwargs) -> DummyClassifier:
        """
        Returns instance of dummy classifier.
        :param kwargs: Arguments for the dummy_classifier.
        :return: Dummy classifier instance.
        """
        return DummyClassifier(**kwargs)

    @staticmethod
    def fit_dummy_classifier(dummy_classifier: DummyClassifier, feature_train: np.ndarray,
                             target_train: np.ndarray) -> Any:
        """
        Fits a dummy classifier with given features and target values.
        :param dummy_classifier: Dummy classifier instance.
        :param feature_train: Feature training array.
        :param target_train: Target training array.
        :return: Fitted dummy classifier instance.
        """
        return dummy_classifier.fit(feature_train, target_train)


    @staticmethod
    def get_dummy_score(dummy_fit: DummyClassifier, feature_test: np.ndarray, target_test: np.ndarray) -> float:
        """
        Returns score of dummy classifier.
        :param dummy_fit: Dummy classifier instance.
        :param feature_test: Feature test array.
        :param target_test: Target test array.
        :return: Score of dummy classifier fit.
        """
        return dummy_fit.score(feature_test, target_test)

    @staticmethod
    def get_decision_tree_classifier(**kwargs) -> DecisionTreeClassifier:
        """
        Returns instance of decision tree classifier.
        :param kwargs: Arguments for the decision_tree_classifier.
        :return: Instance of decision tree classifier.
        """
        return DecisionTreeClassifier(**kwargs)

    @staticmethod
    def get_decision_tree_fit(decision_tree: DecisionTreeClassifier,feature: np.ndarray, target: np.ndarray, **kwargs) -> Any:
        """
        Gets fit from decision tree classifier.
        :param decision_tree: Decision tree classifier instance.
        :param feature: Feature array.
        :param target: Target array.
        :param kwargs: Arguments for the decision_tree_classifier.
        :return: Fitted decision tree classifier.
        """
        return decision_tree.fit(feature, target, **kwargs)

    @staticmethod
    def get_decision_tree_predict(decision_tree: DecisionTreeClassifier, feature: np.ndarray, check_input = None) -> Any:
        """
        Gets predict from decision tree classifier.
        :param decision_tree: Decision tree classifier instance.
        :param feature: Feature test array.
        :param check_input: Bypass several input checking
        :return: Predict array.
        """

        return decision_tree.predict(DataProcessing.convert_np64_to_np32(feature), check_input)

    @staticmethod
    def get_decision_tree_score(decision_tree: DecisionTreeClassifier, predicted_values: np.ndarray, target_test: np.ndarray,
                                sample_weight = None) -> float:
        """
        Returns score of decision tree classifier.
        :param decision_tree: Decision tree classifier instance.
        :param predicted_values: Predicted_values array.
        :param target_test: Target array.
        :param sample_weight: Sample weights
        :return: Float score of decision tree classifier.
        """

        return decision_tree.score(predicted_values, target_test, sample_weight)

    @staticmethod
    def get_cross_validate(model: BaseEstimator, feature: np.ndarray, target: np.ndarray, cv: Any, **kwargs):
        return cross_validate(model, feature, target, cv=cv, **kwargs)

    @staticmethod
    def get_k_fold(**kwargs) -> KFold:
        return KFold(**kwargs)

    @staticmethod
    def get_stratified_k_fold(**kwargs) -> StratifiedKFold:
        return StratifiedKFold(**kwargs)

    @staticmethod
    def get_groups_k_fold(**kwargs) -> GroupKFold:
        return GroupKFold(**kwargs)

    @staticmethod
    def get_standard_scaler(**kwargs) -> StandardScaler:
        return StandardScaler(**kwargs)

    @staticmethod
    def fit_scaler(scaler: StandardScaler, feature: np.ndarray) -> Any:
        return scaler.fit(feature)

    @staticmethod
    def transform_scaler(scaler: StandardScaler, feature: np.ndarray) -> Any:
        return scaler.transform(feature)

    @staticmethod
    def get_svc(**kwargs) -> SVC:
        return SVC(**kwargs)

    @staticmethod
    def fit_svc(svc: Any, feature: np.ndarray, target: np.ndarray, sample_weight: Any = None) -> Any:
        return svc.fit(feature, target, sample_weight=sample_weight)

    @staticmethod
    def predict_svc(svc: SVC, feature: np.ndarray) -> Any:
        return svc.predict(DataProcessing.convert_np64_to_np32(feature))

    @staticmethod
    def get_svc_score(svc: SVC, predicted_values: np.ndarray, target_test: np.ndarray) -> float:
        return svc.score(predicted_values, target_test)

    @staticmethod
    def get_pipeline(args: list) -> Pipeline:
        return Pipeline(args)