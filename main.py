import pandas as pd
from data_processing import DataProcessing as dp


class Main:

    def __init__(self):
        pass

    @staticmethod
    def process():

        uri = "machine-learning-carros.csv"

        dados = pd.read_csv(uri).drop(columns=["Unnamed: 0"], axis=1)

        feature, target = dados.drop(columns=["vendido"], axis=1).values, dados["vendido"].values
        feature_train, feature_test, target_train, target_test = dp.get_train_test_split(feature, target,
                                                                                         test_size = 0.25,
                                                                                         stratify = target)



if __name__ == '__main__':
    Main.process()