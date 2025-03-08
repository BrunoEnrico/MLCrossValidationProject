from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

class DataProcessing:

    @staticmethod
    def get_train_test_split(feature: np.ndarray, target: np.ndarray, **kwargs) -> tuple:
        return train_test_split(feature, target, **kwargs)