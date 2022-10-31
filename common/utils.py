import base64
import numpy as np
from collections import defaultdict
import pickle
from sklearn.base import BaseEstimator, TransformerMixin


def get_threshold(X, mse):
    num = max(0.01 * len(X), 2)

    th = 0.0001
    while (sum(mse > th) > num):
        th += 0.0001
    return th


def serialize_array(arr):
    temp_str = ""
    for element in arr:
        temp_str += str(element)
        temp_str += '|'

    return base64.b64encode(bytes(temp_str, "utf-8"))


def deserialize_string(b64_str: str):
    arr = base64.b64decode(b64_str).decode("utf-8", "ignore")

    return [float(element) for element in arr.split('|') if element != '']


class MinMaxScaler(TransformerMixin, BaseEstimator):
    def __init__(self):
        self._x_min = None
        self._x_max = None
        self._fitted = False

    def fit(self, X, y=None):
        self._x_min = np.min(X, 0)
        self._x_max = np.max(X, 0)
        self._fitted = True

        return self

    def transform(self, X):
        x_std = (X - self._x_min) / (self._x_max - self._x_min + 1e-5)
        x_std = np.nan_to_num(x_std, 0.5)
        return x_std

    def dump(self):
        return pickle.dumps(self)

    @staticmethod
    def load(pickle_str):
        return pickle.loads(pickle_str)

    def __add__(self, other: 'MinMaxScaler'):
        if not self._fitted:
            return other
        if not other._fitted:
            return self
        new_scaler = MinMaxScaler()
        new_scaler._x_max = np.max([self._x_max, other._x_max], axis=0)
        new_scaler._x_min = np.min([self._x_min, other._x_min], axis=0)
        new_scaler._fitted = self._fitted or other._fitted
        return new_scaler

    @staticmethod
    def from_min_max(x_min, x_max):
        scaler = MinMaxScaler()
        scaler._x_min = x_min
        scaler._x_max = x_max
        scaler._fitted = True
        return scaler


def scale_data(X, X_min, X_max):
    X_std = (X - X_min) / (X_max - X_min + 1e-5)
    X_std = np.nan_to_num(X_std, 0.5)
    return X_std


client_malware_map = defaultdict(str)
client_malware_map.update(
    # {
    #     1: 'CTU-Malware-Capture-Botnet-67-1',
    #     2: 'CTU-Malware-Capture-Botnet-219-2',
    #     3: 'CTU-Malware-Capture-Botnet-327-2',
    #     4: 'CTU-Malware-Capture-Botnet-346-1'
    # }
        {
        1: 'CTU-Malware-Capture-Botnet-67-1',
        2: 'CTU-Malware-Capture-Botnet-219-2',
        3: 'CTU-Malware-Capture-Botnet-230-1',
        4: 'CTU-Malware-Capture-Botnet-227-2',
        5: 'CTU-Malware-Capture-Botnet-346-1'
    }
)


def serialize(obj: object) -> bytes:
    return pickle.dumps(obj)


def deserialize(serialized_object: bytes) -> object:
    return pickle.loads(serialized_object)


def pprint_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    # source: https://gist.github.com/zachguo/10296432
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    # Begin CHANGES
    fst_empty_cell = (columnwidth - 3) // 2 * " " + "t/p" + (columnwidth - 3) // 2 * " "

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")
    # End CHANGES

    for label in labels:
        print(f"{label:{columnwidth}}", end=" ")

    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print(f"    {label1:{columnwidth}}", end=" ")
        for j in range(len(labels)):
            cell = f"{cm[i, j]:{columnwidth}.1f}"
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()