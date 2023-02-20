import pandas as pd
import numpy as np

def random_sampling(df: pd.DataFrame, sampling_size: int):
    buff = {}
    features = df["Emotion"].unique()
    for row in df.to_numpy().tolist():
        [_, feat] = row
        if not buff.get(feat):
            buff[feat] = []
        if len(buff[feat]) >= sampling_size:
            continue         
        buff[feat].append(row)

    samples = []
    for feat in features:
        for row in buff[feat]:
            samples.append(row)
    return np.random.permutation(samples), features

data = pd.read_csv("data/train_data.txt", header=0, sep=";", names=["Sentence", "Emotion"])

samples, features = random_sampling(data, 2)

print(samples)