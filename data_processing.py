from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class DataProcessing:

    @staticmethod
    def sort_data(data: pd.DataFrame, column: str, ascending: bool) -> pd.DataFrame:
        """
        Sorts data based on column name.
        :param data: Data to be sorted.
        :param column: Column name to be sorted.
        :param ascending: Boolean value indicating whether sorting is ascending or descending.
        :return: Data sorted.
        """
        return data.sort_values(by=column, ascending=ascending)

    @staticmethod
    def get_train_test_split(feature: np.ndarray, target: np.ndarray, **kwargs) -> tuple:
        """
        Gets train and test split from a given dataframe.
        :param feature: Feature column.
        :param target: Target column.
        :param kwargs: Arguments for the train_test_split.
        :return: Tuple of train and test splits.
        """
        return train_test_split(feature, target, **kwargs)

    @staticmethod
    def convert_np64_to_np32(np64: np.ndarray) -> np.ndarray:
        """
        Converts numpy array to numpy array.
        :param np64: np64 array.
        :return: np32 array.
        """
        return np64.astype(np.float32)
