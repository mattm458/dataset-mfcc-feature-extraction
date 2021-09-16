# Extracts MFCCs from the Mozilla Common Voice corpus
import os

# Make NumPy single-threaded
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
from multiprocessing import Pool

import numpy as np
import pandas as pd
from librosa.feature import mfcc
from pydub import AudioSegment
from tqdm import tqdm

SPLIT = "train"
NUM_PROC = 16

SAMPLE_RATE = 22050
N_MFCC = 20

CV_PATH = "/data/cv-corpus-7.0-2021-07-21/en"
CV_JSON = f"{CV_PATH}/json"


def process(row):
    _, row = row

    if pd.isna(row.sentence):
        return

    file = f"{CV_PATH}/clips/{row.path}"

    if not os.path.exists(file):
        return

    audio = AudioSegment.from_mp3(file).set_frame_rate(SAMPLE_RATE)
    arr = audio.get_array_of_samples()
    audio_data = np.array(arr).astype(np.float32) / np.iinfo(arr.typecode).max

    output_data = row.to_dict()
    output_data["mfccs"] = mfcc(audio_data, sr=SAMPLE_RATE, n_mfcc=N_MFCC).T.ravel().tolist()

    with open(
        f"{CV_JSON}/mfcc_1/{SPLIT}/{row.path.replace('.mp3', '.json')}", "w"
    ) as outfile:
        json.dump(output_data, outfile)


if __name__ == "__main__":
    if not os.path.exists(CV_JSON):
        os.mkdir(CV_JSON)

    df = pd.read_csv(f"{CV_PATH}/{SPLIT}.tsv", delimiter="\t")

    with Pool(NUM_PROC) as pool:
        for _ in tqdm(pool.imap(process, df.iterrows()), total=len(df)):
            pass
