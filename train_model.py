"""
Author: Ani Aggarwal
Github: www.github.com/AniAggarwal
"""
from pathlib import Path
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras


def load_data_np():
    df_train = pd.read_csv("./datasets/ECG-kaggle/mitbih_train.csv", header=None)
    df_train = df_train.sample(frac=1)
    df_test = pd.read_csv("./datasets/ECG-kaggle/mitbih_test.csv", header=None)
    df_test = df_test.sample(frac=1)

    X_train = np.array(df_train[list(range(187))].values)
    X_train = np.expand_dims(X_train, -1)
    y_train = np.array(df_train[187].values).astype(np.int8)
    y_train[y_train != 0] = 1  # converting to be just normal (0) and abnormal (1)

    X_test = np.array(df_test[list(range(187))].values)
    X_test = np.expand_dims(X_test, -1)
    y_test = np.array(df_test[187].values).astype(np.int8)
    y_test[y_test != 0] = 1
    print(
        f"X_train {X_train.shape}, y_train {y_train.shape}, X_test {X_test.shape}, y_test {y_test.shape}"
    )

    return X_train, y_train, X_test, y_test


def create_model(X_shape, lr):
    model = keras.Sequential(
        [
            keras.layers.Bidirectional(
                keras.layers.LSTM(
                    128, input_shape=(X_shape[1], X_shape[2]), return_sequences=True,
                )
            ),
            keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
            keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=False)),
            keras.layers.Dense(32),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dropout(0.7),
            keras.layers.Dense(16),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.build(X_shape)

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    model.summary()

    return model


def plot_history(history):
    fig, axes = plt.subplots(nrows=2, ncols=1)
    axes[0].title.set_text("Loss")
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="test")
    axes[0].legend()
    # plot accuracy during training
    axes[1].title.set_text("Accuracy")
    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="test")
    axes[1].legend()


if __name__ == "__main__":
    mirrored_strategy = tf.distribute.MirroredStrategy()

    EPOCHS = 100
    BATCH_SIZE = 1500
    LEARNING_RATE = 1e-2

    LOAD_MODEL = "none"  # "full", "weights", or "none"

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    data_path = Path("./datasets/ECG-kaggle/mitbih_train.csv")
    model_path = Path("./model_saves/") / (
        time_str
        + f"_epochs_{EPOCHS}-batch_size_{BATCH_SIZE}-lr_{LEARNING_RATE}"
    )
    model_name = (
        model_path
        / "epoch_{epoch:02d}-val_acc_{val_accuracy:.2f}-val_loss_{val_loss:.2f}.hdf5"
    )
    full_model_path = model_path / "full_save"
    tensorboard_log_path = Path(f"./model_saves/logs/{time_str}/")

    model_path.mkdir(parents=True, exist_ok=False)
    full_model_path.mkdir(parents=True, exist_ok=True)
    tensorboard_log_path.mkdir(parents=True, exist_ok=False)

    # file location for saved model weights
    # best so far: 0.9889 at 2021-06-26_17-23-26_epochs_100-batch_size_2000-lr_0.01
    # 0.9878 at 2021-06-26_16-39-09_epochs_50-batch_size_2000-lr_0.01
    # 0.9577 at 4746751_epochs_20-batch_size_5000-lr_0.001
    model_weights_save = Path(
        "./model_saves/4748001_epochs_50-batch_size_5000-lr_0.01/epoch_08-val_acc_0.83-val_loss_0.97.hdf5"
    )
    model_full_save = Path(
        "./model_saves/2021-06-28_19-22-14_epochs_100-batch_size_2000-lr_0.01/full_save"
    )

    # load in data
    X_train, y_train, X_test, y_test = load_data_np()

    # load/create model
    if LOAD_MODEL == "full":
        model = keras.models.load_model(model_full_save)
    else:
        model = create_model(X_train.shape, LEARNING_RATE)
        if LOAD_MODEL == "weights":
            model.load_weights(model_weights_save)

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=model_name,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
    )

    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        min_delta=0.005,
        patience=20,
        verbose=1,
        restore_best_weights=True,
    )

    lr_plateau_callback = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", mode="min", min_delta=0.005, patience=5, verbose=1,
    )

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=tensorboard_log_path, histogram_freq=1, update_freq="epoch", )

    # lr_schedule = keras.callbacks.LearningRateScheduler(
    #     lambda epoch: LEARNING_RATE * 10 ** (epoch / 20)
    # )

    callbacks_list = [
        checkpoint_callback,
        early_stopping_callback,
        lr_plateau_callback,
        tensorboard_callback
    ]

    try:
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks_list,
        )
        plot_history(history)

    finally:
        print("Saving full model to", str(full_model_path))
        keras.models.save_model(
            model, full_model_path,
        )

        plt.show()
