# Extracts MFCCs from the the LJSpeech corpus
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

NUM_PROC = 16

SAMPLE_RATE = 22050
N_MFCC = 20
MFCC_DIR = "mfcc_1"

LJ_PATH = "/data/LJSpeech-1.1"
LJ_JSON = f"{LJ_PATH}/json"


def process(row):
    _, row = row

    file = f"{LJ_PATH}/wavs/{row.id}.wav"

    if not os.path.exists(file):
        return

    audio = AudioSegment.from_wav(file).set_frame_rate(SAMPLE_RATE)
    arr = audio.get_array_of_samples()
    audio_data = np.array(arr).astype(np.float32) / np.iinfo(arr.typecode).max

    output_data = row.to_dict()
    output_data["mfccs"] = (
        mfcc(audio_data, sr=SAMPLE_RATE, n_mfcc=N_MFCC).T.ravel().tolist()
    )

    with open(f"{LJ_JSON}/{MFCC_DIR}/{row.id}.json", "w") as outfile:
        json.dump(output_data, outfile)


if __name__ == "__main__":
    if not os.path.exists(LJ_JSON):
        os.mkdir(LJ_JSON)
    if not os.path.exists(f"{LJ_JSON}/{MFCC_DIR}"):
        os.mkdir(f"{LJ_JSON}/{MFCC_DIR}")

    df = pd.read_csv(
        f"{LJ_PATH}/metadata.csv",
        delimiter="|",
        header=None,
        names=["id", "raw", "normalized"],
    )

    with Pool(NUM_PROC) as pool:
        for _ in tqdm(pool.imap(process, df.iterrows()), total=len(df)):
            pass
