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
    X = np.array(data["audio"])
    y = np.array(data["labels"])

    print(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}")
    return X, y


def unison_shuffled_copies(a, b):
    # from https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def process_data(X, y):
    # add dimension to make it uniform with other dataset
    X = np.expand_dims(X, -1)

    # convert to 1 for abnormal and 0 for normal
    y[y == -1] = 0

    # shuffle data
    print("Shuffling data")
    X, y = unison_shuffled_copies(X, y)
    print(f"Data shuffled. X shape: {X.shape}, y shape: {y.shape}")

    # balance data
    # actual counts: abnormal: 3158, normal: 9857
    num_abnormal, num_normal = np.count_nonzero(y == 1), np.count_nonzero(y == 0)
    print(
        f"Balancing data. Before count abnormal: {num_abnormal}, normal: {num_normal}"
    )

    if num_abnormal > num_normal:
        # select few from abnormals
        X = np.concatenate([X[y == 0][:num_normal], X[y == 0]])
        y = np.concatenate([y[y == 0][:num_normal], y[y == 0]])
    else:
        # select few from normals
        X = np.concatenate([X[y == 0][:num_abnormal], X[y == 1]])
        y = np.concatenate([y[y == 0][:num_abnormal], y[y == 1]])

    print(
        f"After count abnormal: {np.count_nonzero(y == 1)}, normal: {np.count_nonzero(y == 0)}"
    )

    # split data
    return train_test_split(X, y, test_size=0.15)


if __name__ == "__main__":
    json_path = Path("./datasets/classification-heart-sounds-physionet/numpy-data/data.json")
    np_path = Path("./datasets/classification-heart-sounds-physionet/numpy-data/")

    print("Loading data from", str(json_path))
    X, y = load_data(json_path)
    X_train, X_test, y_train, y_test = process_data(X, y)

    np.save(np_path / "X_train", X_train)
    np.save(np_path / "y_train", y_train)
    np.save(np_path / "X_test", X_test)
    np.save(np_path / "y_test", y_test)

    print("Data saved to", str(np_path))
