import base64
import numpy as np


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


def scale_data(X, X_min, X_max):
    X_std = (X - X_min) / (X_max - X_min + 1e-5)
    X_std = np.nan_to_num(X_std, 0.5)
    return X_std
