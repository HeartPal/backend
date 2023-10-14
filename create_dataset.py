"""
Author: Ani Aggarwal
Github: www.github.com/AniAggarwal
"""
import json

import librosa
import librosa.display

from scipy.signal import butter, lfilter
import numpy as np

from pathlib import Path
import csv

DOWN_SR = 125
BPF_LOW = int(DOWN_SR * 20 / 1000)
BPF_HIGH = int(DOWN_SR * 400 / 1000)
BPF_ORDER = 3

AUDIO_LEN = 187 / DOWN_SR  # in seconds
OVERLAP_LEN = AUDIO_LEN / 4  # in seconds


def create_dataset(dataset_dir, output_path):
    data = {"audio": [], "labels": []}

    for dataset_path in dataset_dir.iterdir():
        if "training-" in dataset_path.name:
            print(f"Processing training set at {dataset_path}.")
            with open(dataset_path / "REFERENCE.csv", "r") as f:
                csv_reader = csv.reader(f)
                for file_name, label in csv_reader:
                    file_path = str(dataset_path / file_name) + ".wav"

                    # print(f"Processing {file_name} at location {file_path}.")

                    audio = process_audio_file(file_path, AUDIO_LEN)

                    data["audio"] += audio
                    data["labels"] += [int(label)] * len(audio)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    print("Saved output to", output_path)

def create_test_set(test_set_dir, output_path):
    data = {"audio": [], "labels": []}

    with open(test_set_dir / "REFERENCE.csv", "r") as f:
        csv_reader = csv.reader(f)
        for file_name, label in csv_reader:
            file_path = str(test_set_dir / file_name) + ".wav"

            audio = process_audio_file(file_path, AUDIO_LEN)

            data["audio"] += audio
            data["labels"] += [int(label)] * len(audio)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    print("Saved output to", output_path)


def process_audio_file(audio_path, sample_len):
    audio, _ = librosa.load(audio_path, sr=DOWN_SR)  # load + down sample
    return process_audio_np(audio, sample_len)


def process_audio_np(audio_np, sample_len, filter=False):
    audio = denoise_audio_np(audio_np, filter)

    num_segments = int((audio.size / DOWN_SR) // sample_len)
    samples_per_segment = sample_len * DOWN_SR

    audio_segments = []
    for current_segment in range(num_segments):
        start = int((samples_per_segment * current_segment) - (OVERLAP_LEN * samples_per_segment))
        end = int(start + samples_per_segment)

        if start < 0:
            end -= start
            start = 0
        end = min(end, len(audio))

        audio_segments.append(audio[start:end].tolist())
    return audio_segments


def denoise_audio_np(audio, bpf_low=BPF_LOW, bpf_high=BPF_HIGH, bpf_order=BPF_ORDER, filter=False):
    if filter:
        audio = butter_bandpass_filter(audio, bpf_low, bpf_high, DOWN_SR, bpf_order)
    return (audio - audio.min()) / (audio.max() - audio.min())  # normalize between 0 and 1


# bandpass from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


if __name__ == "__main__":
    datasets_path = Path("./datasets/classification-heart-sounds-physionet/")
    data_output_path = Path("./datasets/classification-heart-sounds-physionet/numpy-data/data.json")

    create_dataset(datasets_path, data_output_path)

    # to unjsonify numpy arrays: https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
