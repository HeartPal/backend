"""
Author: Ani Aggarwal
Github: www.github.com/AniAggarwal
"""
from pathlib import Path
import datetime

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


def create_transfer_model(X_shape, lr):
    # base_model = keras.models.load_model(transfer_fulL_save)
    # base_model.summary()
    # for layer in base_model.layers:
    #     print(f"Input shape {layer.input_shape}")
    #
    # model = keras.Sequential()
    #
    # model.add(
    #     keras.layers.Bidirectional(
    #         keras.layers.LSTM(
    #             128, input_shape=(X_shape[1], X_shape[2]), return_sequences=True
    #         )
    #     )
    # )
    #
    # for layer in base_model.layers[1:]:
    #     model.add(layer)
    #
    # for idx, _ in enumerate(model.layers[1:3]):
    #     model.layers[idx].trainable = False
    model = keras.models.load_model(transfer_fulL_save)

    # for idx, _ in enumerate(model.layers[1:3]):
    #     model.layers[idx].trainable = False

    model.build(X_shape)
    optimizer = keras.optimizers.Adam(learning_rate=lr, amsgrad=True)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

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
    keras.backend.clear_session()
    mirrored_strategy = tf.distribute.MirroredStrategy()

    EPOCHS = 500
    BATCH_SIZE = 1000
    LEARNING_RATE = 1e-1

    LOAD_MODEL = "full"  # "full", "weights", "none", or "transfer

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    data_path = Path("./datasets/classification-heart-sounds-physionet/numpy-data/")
    model_path = Path("./model_saves/transfer") / (
        time_str + f"_epochs_{EPOCHS}-batch_size_{BATCH_SIZE}-lr_{LEARNING_RATE}"
    )
    model_name = (
        model_path
        / "TRANSFER_epoch_{epoch:02d}-val_acc_{val_accuracy:.2f}-val_loss_{val_loss:.2f}.hdf5"
    )
    full_model_path = model_path / "full_save"
    tensorboard_log_path = Path(f"./model_saves/logs/transfer/{time_str}/")

    model_path.mkdir(parents=True, exist_ok=False)
    full_model_path.mkdir(parents=True, exist_ok=True)
    tensorboard_log_path.mkdir(parents=True, exist_ok=False)

    # file location for saved model weights
    # best so far:
    # 0.6086 at 2021-06-26_21-04-45_epochs_100-batch_size_1000-lr_0.01
    # 0.8 at 2021-06-27_05-16-49_epochs_500-batch_size_1000-lr_0.01/TRANSFER_epoch_88-val_acc_0.80-val_loss_0.45.hdf5
    # transfer weights save:
    transfer_fulL_save = Path("./model_saves/transfer/transfer-base-weights/full_save")
    model_weights_save = Path("./model_saves/transfer/2021-06-27_05-16-49_epochs_500-batch_size_1000-lr_0.01/TRANSFER_epoch_88-val_acc_0.80-val_loss_0.45.hdf5")
    model_full_save = Path(
        "./model_saves/transfer/2021-06-27_05-57-14_epochs_500-batch_size_1000-lr_0.01/full_save"
    )

    # load in data
    X_train, y_train, X_test, y_test = load_data_np(data_path)

    # load/create model
    if LOAD_MODEL == "full":
        model = keras.models.load_model(model_full_save)
    else:
        model = create_transfer_model(X_train.shape, LEARNING_RATE)
        if LOAD_MODEL == "weights":
            model.load_weights(model_weights_save)
    model.summary()

    # print(np.expand_dims(X_train[0], 0).shape)
    # print(model.predict_on_batch(np.expand_dims(X_train[0], 0)))

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
        monitor="val_loss", mode="min", min_delta=0.005, patience=15, verbose=1,
    )

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=tensorboard_log_path, histogram_freq=1, update_freq="epoch"
    )

    # lr_schedule = keras.callbacks.LearningRateScheduler(
    #     lambda epoch: LEARNING_RATE * 10 ** (epoch / 20)
    # )

    callbacks_list = [
        checkpoint_callback,
        # early_stopping_callback,
        lr_plateau_callback,
        tensorboard_callback,
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
