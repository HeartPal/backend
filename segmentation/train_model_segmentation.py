"""
Author: Ani Aggarwal
Github: www.github.com/AniAggarwal
"""
from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras


def load_data_np(data_path):
    X_train = np.load(data_path / "X_train.npy")
    y_train = np.load(data_path / "y_train.npy")
    X_test = np.load(data_path / "X_test.npy")
    y_test = np.load(data_path / "y_test.npy")

    print(
        f"X_train {X_train.shape}, y_train {y_train.shape}, X_test {X_test.shape}, y_test {y_test.shape}"
    )

    return X_train, y_train, X_test, y_test


def create_model(X_shape, lr):
    model = keras.Sequential(
        [
            # keras.layers.Bidirectional(
            #     keras.layers.LSTM(
            #         32, input_shape=(X_shape[1], X_shape[2]), return_sequences=False,
            #     )
            # ),
            keras.layers.LSTM(
                    32, input_shape=(X_shape[1], X_shape[2]), return_sequences=False,
                ),
            # keras.layers.SimpleRNN(32, input_shape=(X_shape[1], X_shape[2])),
            # keras.layers.Conv1D(8, 3, activation="relu", input_shape=(X_shape[1], X_shape[2])),
            # keras.layers.Flatten(),
            # keras.layers.BatchNormalization(),
            # keras.layers.ReLU(),
            # keras.layers.Dropout(0.4),
            keras.layers.Dense(100, activation="sigmoid"),
        ]
    )
    model.build(X_shape)

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=keras.losses.BinaryCrossentropy(from_logits=False), metrics=["accuracy"])

    model.summary()
    for layer in model.layers:
        print(layer.input_shape)

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
    plt.show()


if __name__ == "__main__":
    mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

    EPOCHS = 100
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-2

    LOAD_MODEL = "none"  # "full", "weights", or "none"

    np_path = Path("../datasets/segmentation")
    model_path = Path("../model_saves/segmentation") / (
        str(time.time()).split(".")[0][3:]
        + f"_epochs_{EPOCHS}-batch_size_{BATCH_SIZE}-lr_{LEARNING_RATE}"
    )
    model_name = (
        model_path
        / "epoch_{epoch:02d}-val_acc_{val_accuracy:.2f}-val_loss_{val_loss:.2f}.hdf5"
    )
    full_model_path = model_path / "full_save"

    model_path.mkdir(parents=True, exist_ok=False)
    full_model_path.mkdir(parents=True, exist_ok=True)

    # file location for saved model weights
    # best so far:
    model_weights_save = Path("../model_saves/segmentation/3531223_epochs_30-batch_size_2-lr_1e-06/epoch_17-val_acc_0.82-val_loss_0.42.hdf5")
    model_full_save = Path(
        "../model_saves/segmentation/4728838_epochs_1000-batch_size_2-lr_1e-06/full_save"
    )

    # load in data
    X_train, y_train, X_test, y_test = load_data_np(np_path)

    # load/create model
    if LOAD_MODEL == "full":
        model = keras.models.load_model(model_full_save)
    else:
        model = create_model(X_train.shape, LEARNING_RATE)
        if LOAD_MODEL == "weights":
            model.load_weights(model_weights_save)

    print()
    print(X_train[0].shape)
    print(y_train[0])
    print(model.predict_on_batch(np.expand_dims(X_train[0], 0)))


    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=model_name,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
    )

    lr_schedule = keras.callbacks.LearningRateScheduler(
        lambda epoch: LEARNING_RATE * 10 ** (epoch / 20)
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint_callback, lr_schedule],
    )

    # lrs = LEARNING_RATE * (
    #     10 ** np.arange(EPOCHS) / 20
    # )  # using the lr_shedule function
    # # general plotting code
    # plt.semilogx(lrs, history.history["loss"])
    # # numbers need to be modified to make a good graph
    # # params: xmin, xmax, ymin, ymax
    # # plt.axis([LEARNING_RATE, LEARNING_RATE*, 0, 300])
    # plt.show()

    print("Saving full model to", str(full_model_path))
    keras.models.save_model(
        model, full_model_path,
    )

    plot_history(history)
