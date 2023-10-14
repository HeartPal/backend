"""
Author: Ani Aggarwal
Github: www.github.com/AniAggarwal
"""
import json

import librosa
import librosa.display

from scipy.signal import butter, lfilter
import pandas as pd
import math

from pathlib import Path

DOWN_SR = 1000
BPF_LOW = 20
BPF_HIGH = 400
BPF_ORDER = 3
DATA_SR = 44100

AUDIO_LEN = 1  # in seconds
AUDIO_OVERLAP = 0.5
num_samples = AUDIO_LEN * DOWN_SR


def create_dataset(dataset_dir, output_path):
    df = pd.read_csv(dataset_dir / "set_a_timing.csv")

    hop_len = int(0.01 * DOWN_SR)

    data = {"mfccs": [], "labels": []}
    class_to_int = {"S1": 0, "S2": 1}

    for filename, group in df.groupby(["fname"]):
        print(f"processing file {filename}")
        file_path = dataset_dir / filename
        mfccs = process_audio_file(file_path, AUDIO_LEN)

        labels = [None] * len(mfccs) * len(mfccs[0])

        prev_class_idx = None
        for row_tuple in group[["sound", "location"]].itertuples():
            new_loc = int((int(row_tuple[2]) * DOWN_SR / DATA_SR) / hop_len)
            if new_loc < len(labels) - 1:
                if prev_class_idx is not None:
                    labels[prev_class_idx + 1 : new_loc + 1] = [
                        class_to_int[row_tuple[1]]
                    ] * (new_loc - prev_class_idx)
                else:
                    labels[0 : new_loc + 1] = [class_to_int[row_tuple[1]]] * (
                        new_loc + 1
                    )

                prev_class_idx = new_loc

        if labels[-1] is None:
            labels[prev_class_idx + 1 :] = [int(not labels[prev_class_idx])] * (
                len(labels) - prev_class_idx - 1
            )

        split_labels = []
        for i in range(len(mfccs)):
            split_labels.append(labels[i * len(mfccs[0]) : (i + 1) * len(mfccs[0])])

        data["mfccs"] += mfccs
        data["labels"] += split_labels

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


def process_audio_file(audio_path, sample_len):
    audio, _ = librosa.load(audio_path, sr=DOWN_SR)  # load + down sample
    return process_audio_np(audio, sample_len)


def process_audio_np(audio_np, sample_len):
    audio = denoise_audio_np(audio_np)

    n_fft = int(0.025 * DOWN_SR)
    hop_len = int(0.01 * DOWN_SR)

    num_segments = int((audio.size / DOWN_SR) // sample_len)
    samples_per_segment = sample_len * DOWN_SR
    expected_mfcc_per_segment = math.ceil(samples_per_segment / hop_len)

    mfccs = []
    for current_segment in range(num_segments):
        start = samples_per_segment * current_segment - (AUDIO_OVERLAP * DOWN_SR)
        end = start + samples_per_segment + (AUDIO_OVERLAP * DOWN_SR)
        print(start, end)

        # find mfcc
        mfcc = librosa.feature.mfcc(
            audio[start:end], DOWN_SR, n_fft=n_fft, hop_length=hop_len, n_mfcc=13
        )
        mfcc = mfcc.T

        if len(mfcc) == expected_mfcc_per_segment:
            mfccs.append(mfcc.tolist())

    return mfccs


def denoise_audio_np(audio_np, bpf_low=20, bpf_high=400, bpf_order=3):
    audio = butter_bandpass_filter(audio_np, bpf_low, bpf_high, DOWN_SR, bpf_order)
    return 2 * (audio - audio.min()) / (audio.max() - audio.min()) - 1.0  # normalize


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
    datasets_path = Path("../datasets/segmentation/heartbeat-sounds-kaggle/")
    data_output_path = Path("../datasets/segmentation/data.json")

    create_dataset(datasets_path, data_output_path)
