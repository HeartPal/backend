"""
Author: Ani Aggarwal
Github: www.github.com/AniAggarwal
"""
from pathlib import Path

import soundfile
import tensorflow.keras as keras
import numpy as np
import librosa

from create_dataset import process_audio_np, denoise_audio_np, create_dataset
from clean_dataset import load_data, process_data
from train_model import load_data_np


class Inference:
    def __init__(self, model_path=Path("./BiLSTM_ECG"), input_sample_rate=125, sample_rate=125):
        self.model = keras.models.load_model(model_path)
        self.input_sample_rate = input_sample_rate
        self.sample_rate = sample_rate
        self.data_len = 187
        self.sample_len_sec = self.data_len / self.sample_rate

        self.model.summary()

    def predict(self, byte_list, threshold=0.05):
        if len(byte_list) / self.sample_rate < self.sample_len_sec:
            return "unsure"

        x = self.preprocess(byte_list)

        pred = self.model.predict_on_batch(x)
        avg_pred = np.mean(pred)

        if 0.5 - threshold <= avg_pred <= 0.5 + threshold:
            return "unsure"
        if avg_pred > 0.5:
            return "abnormal"
        return "normal"

    def evaluate(self, X_test, y_test):
        metrics = [
            keras.metrics.Accuracy(),
            keras.metrics.BinaryAccuracy(),
            keras.metrics.AUC(name="AUC-ROC", curve="ROC"),
            keras.metrics.AUC(name="AUC-PR", curve="PR"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.TruePositives(name="true_positive"),
            keras.metrics.FalsePositives(name="false_positive"),
            keras.metrics.FalseNegatives(name="false_negative"),
            keras.metrics.TrueNegatives(name="true_negative"),
        ]
        self.model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=metrics
        )

        result = self.model.evaluate(X_test, y_test)
        result_dict = [
            "loss",
            "accuracy",
            "binary_accuracy",
            "AUC-ROC",
            "AUC-PR",
            "precision",
            "recall",
            "true_positive",
            "false_positive",
            "false_negative",
            "true_negative",
        ]
        result_dict = {metric: result[idx] for idx, metric in enumerate(result_dict)}

        f1 = (
            2
            * result_dict["precision"]
            * result_dict["recall"]
            / (result_dict["precision"] + result_dict["recall"])
        )
        result_dict["f1"] = f1

        sensitivity = result_dict["true_positive"] / (
            result_dict["true_positive"] + result_dict["false_negative"]
        )
        specificity = result_dict["true_negative"] / (
            result_dict["true_negative"] + result_dict["false_positive"]
        )

        result_dict["sensitivity"] = sensitivity
        result_dict["specificity"] = specificity

        return result_dict

    def preprocess(self, byte_list):
        audio = np.array(byte_list)
        audio = librosa.resample(audio, self.input_sample_rate, self.sample_rate)
        audio_segments = process_audio_np(audio, self.sample_len_sec)
        audio_segments = np.array(audio_segments)
        if len(audio_segments.shape) == 2:
            audio_segments = np.expand_dims(audio_segments, -1)
        return audio_segments

    def get_wav(self, byte_list, wav_path=Path("./tmp/tmp_recording.wav")):
        audio = np.array(byte_list)
        audio = librosa.resample(audio, self.input_sample_rate, self.sample_rate)
        soundfile.write(wav_path, audio, 1000)
        return wav_path

    def delete_wav(self, wav_path=Path("./tmp/tmp_recording.wav")):
        wav_path = Path(wav_path)
        wav_path.unlink(missing_ok=True)


if __name__ == "__main__":
    # model_full_save = Path(
    #     "./model_saves/transfer/2021-06-27_05-35-15_epochs_100-batch_size_1000-lr_0.01/full_save"
    # )
    model_full_save = Path("./BiLSTM_ECG")
    test_audio_path = Path(
        "./datasets/classification-heart-sounds-physionet/training-a/a0004.wav"
    )
    # test_set_path = Path(
    #     "./datasets/classification-heart-sounds-physionet/validation"
    # )
    test_output_path = Path(
        "./datasets/classification-heart-sounds-physionet/numpy-data/data-test.json"
    )

    inference = Inference(
        model_full_save, 125, 125
    )  # for testing, dataset has 2kHz sample rate

    X_train, y_train, X_test, y_test = load_data_np()

    # create_test_set(test_set_path, test_output_path)
    # X_test, y_test = load_data(test_output_path)
    # X_test = np.expand_dims(X_test, -1)  # add dimension to make it uniform with model input
    y_test[y_test == -1] = 0  # convert to 1 for abnormal and 0 for normal

    print("Model eval", inference.evaluate(X_test, y_test))

    audio_np, sr = librosa.load(test_audio_path, sr=None)
    wav = inference.get_wav(audio_np.tolist())
    print(wav)

    inference.delete_wav(wav)

    print(inference.predict(audio_np))
