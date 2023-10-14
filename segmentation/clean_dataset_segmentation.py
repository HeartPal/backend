"""
Author: Ani Aggarwal
Github: www.github.com/AniAggarwal
"""
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def load_data(dataset_path):
    with open(dataset_path, "r") as f:
        data = json.load(f)

    # convert lists back to np arrays
    X = np.array(data["mfccs"])
    y = np.array(data["labels"])
    # shapes: X (13015, 500, 13), y (13015,)

    print(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}")
    return X, y


def unison_shuffled_copies(a, b):
    # from https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def process_data(X, y):
    # convert to 1 for abnormal and 0 for normal
    y[y == -1] = 0

    # shuffle data
    print("Shuffling data")
    X, y = unison_shuffled_copies(X, y)
    print(f"Data shuffled. X shape: {X.shape}, y shape: {y.shape}")

    # split data
    return train_test_split(X, y, test_size=0.15)


if __name__ == "__main__":
    json_path = Path("../datasets/segmentation/data.json")
    np_path = Path("../datasets/segmentation/")

    print("Loading data from", str(json_path))
    X, y = load_data(json_path)
    X_train, X_test, y_train, y_test = process_data(X, y)

    print(f"Data shapes: X_train {X_train.shape}; y_train {y_train.shape}; X_test {X_test.shape}; y_test {y_test.shape}")

    np.save(np_path / "X_train", X_train)
    np.save(np_path / "y_train", y_train)
    np.save(np_path / "X_test", X_test)
    np.save(np_path / "y_test", y_test)

    print("Data saved to", str(np_path))
