import pandas as pd
import numpy as np
from data_processing import DataProcessing
from machine_learning import MachineLearning


class Main:

    def __init__(self):
        self.seed = 5

    def process(self):

        uri = "machine-learning-carros.csv"

        data = pd.read_csv(uri).drop(columns=["Unnamed: 0"], axis=1)

        feature = data.drop(columns=["vendido"], axis=1).values
        target = data["vendido"].values

        np.random.seed(self.seed)

        feature_train, feature_test, target_train, target_test = DataProcessing.get_train_test_split(feature, target,
                                                                                                     test_size = 0.25,
                                                                                                     stratify = target)
        dummy_classifier = MachineLearning.get_dummy_classifier()
        dummy_fit = MachineLearning.fit_dummy_classifier(dummy_classifier, feature_train, target_train)
        dummy_score = MachineLearning.get_dummy_score(dummy_fit, feature_test, target_test)
        # print(f"The Dummy model's accuracy is {round(dummy_score * 100, 2)}")

        decision_tree_classifier = MachineLearning.get_decision_tree_classifier(max_depth=2)
        decision_tree_fit = MachineLearning.get_decision_tree_fit(decision_tree_classifier, feature_train, target_train)
        decision_tree_predict = MachineLearning.get_decision_tree_predict(decision_tree_fit, feature_test)
        decision_tree_score = MachineLearning.get_decision_tree_score(decision_tree_fit, feature_test, target_test)
        #print(f"The Decision Tree Score was {decision_tree_score * 100}%")

        k_fold = MachineLearning.get_k_fold(n_splits=10, shuffle=True, random_state=self.seed)
        cross_validate_results = MachineLearning.get_cross_validate(decision_tree_classifier, feature, target, cv=k_fold,
                                                                    return_train_score=False)
        #print(cross_validate_results["test_score"].mean())

        stratified_k_fold = MachineLearning.get_k_fold(n_splits=10, shuffle=True, random_state=self.seed)
        cross_validate_results = MachineLearning.get_cross_validate(decision_tree_classifier, feature, target, cv=stratified_k_fold,
                                                                    return_train_score=False)

        result = np.random.randint(-2, 2, size=10000)
        data["modelo"] = result
        data["modelo"] = data["idade_do_modelo"] + abs(data["modelo"].min() + 1)

        cv = MachineLearning.get_groups_k_fold(n_splits=10)
        group_k_fold_model = MachineLearning.get_decision_tree_classifier(max_depth=2)
        results = MachineLearning.get_cross_validate(group_k_fold_model, feature, target, cv=cv, groups = data.modelo, return_train_score=False)
        #print(results["test_score"].mean())

        scaler = MachineLearning.get_standard_scaler()
        scaler_fit = MachineLearning.fit_scaler(scaler, feature_train)
        feature_train_scaled = scaler_fit.transform(feature_train)
        feature_test_scaled = scaler_fit.transform(feature_test)
        svc = MachineLearning.get_svc()
        svc_fit = MachineLearning.fit_svc(svc, feature_train_scaled, target_train)
        MachineLearning.predict_svc(svc_fit, feature_train_scaled)
        # print(MachineLearning.get_svc_score(svc_fit, feature_train_scaled, target_train))

        results = MachineLearning.get_cross_validate(svc_fit, feature, target, cv=cv, groups=data.modelo,
                                                     return_train_score=False)

        cv = MachineLearning.get_groups_k_fold(n_splits=10)
        svc = MachineLearning.get_svc()
        scaler = MachineLearning.get_standard_scaler()
        pipeline = MachineLearning.get_pipeline([('scaler', scaler), ('transformer', svc)])
        MachineLearning.get_cross_validate(pipeline, feature, target, cv=cv, groups=data.modelo,
                                           return_train_score=False)








if __name__ == '__main__':
    main = Main()
    main.process()