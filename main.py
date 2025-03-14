import pandas as pd
from sklearn.dummy import DummyClassifier

from data_processing import DataProcessing
from machine_learning import MachineLearning

class Main:

    def __init__(self):
        pass

    @staticmethod
    def process():

        dp = DataProcessing()
        uri = "machine-learning-carros.csv"

        dados = pd.read_csv(uri).drop(columns=["Unnamed: 0"], axis=1)

        feature, target = dados.drop(columns=["vendido"], axis=1).values, dados["vendido"].values
        feature_train, feature_test, target_train, target_test = DataProcessing.get_train_test_split(feature, target,
                                                                                                     test_size = 0.25,
                                                                                                     stratify = target)
        dummy_classifier = MachineLearning.get_dummy_classifier()
        dummy_fit = MachineLearning.fit_dummy_classifier(dummy_classifier, feature_train, target_train)
        dummy_score = MachineLearning.get_dummy_score(dummy_fit, feature_test, target_test)
        print(f"The Dummy model's accuracy is {round(dummy_score * 100, 2)}")






if __name__ == '__main__':
    Main.process()