from importlib.resources import path
import os
import numpy as np
import pandas as pd
import sonopy
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.utils import make_chunks
from scipy.io import wavfile
from yaml import load, Loader

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

CONFIG = {}

with open('config.yml', 'r') as config_file:
    CONFIG = load(config_file, Loader=Loader)
    config_file.close()

Dirs = CONFIG['Dirs']
NNparams = CONFIG['NeuralNet']
Optiparams = CONFIG['Optimizer']

currdir = os.getcwd()
data_dir = os.path.join(currdir, Dirs["data_dir"])
posDir = os.path.join(data_dir, Dirs["positives"])
negDir = os.path.join(data_dir, Dirs["negatives"])
templates = os.path.join(data_dir, Dirs["backgrounds"])
model_data_dir = os.path.join(currdir, Dirs["nn"], Dirs["model_data_dir"])
training_path = os.path.join(model_data_dir, "training_data.csv")
testing_path = os.path.join(model_data_dir, "testing_data.csv")

def createData():
    paths = []
    labels = []
    for audio in os.listdir(posDir):
        paths.append(os.path.join(posDir, audio))
        labels.append(1)
    for audio in os.listdir(negDir):
        paths.append(os.path.join(negDir, audio))
        labels.append(0)
    data = pd.DataFrame({
        'path': paths,
        'label': labels
    })
    return data

def createDir(dirname):
    dirpath = os.path.join(dirname)
    if not os.path.exists(dirpath):
        try:
            os.mkdir(dirpath)
        except Exception as e:
            print(str(e))
    return dirpath

def createSamples(data):
    datalen = len(data)
    sampleDir = createDir(os.path.join(data_dir, "samples"))
    chunksDir = createDir(os.path.join(data_dir, "chunks"))
    threshold = 125
    df = pd.DataFrame()
    sampleCount = 1
    chunkCount = 1
    for audiopath in os.listdir(templates):
        tempPath = os.path.join(templates, audiopath)
        background = AudioSegment.from_wav(tempPath)
        segments = []
        nextsegment = 100
        tempSample = background
        while(True):
            randData = data.iloc[np.random.randint(datalen)]
            path = randData["path"]
            label = randData["label"]
            sample = AudioSegment.from_wav(path)
            duration = sample.duration_seconds
            if nextsegment + duration < 10000:
                try:
                    tempSample = tempSample.overlay(sample, position=nextsegment)
                    if label==1:
                        segments.append((nextsegment+threshold, nextsegment+(sample.duration_seconds*1000)-1-threshold))
                    nextsegment = nextsegment + (sample.duration_seconds*1000) + 199
                except:
                    pass
            else:
                break
        tempSample.export(os.path.join(sampleDir, f"sample{sampleCount}.wav"), format="wav")
        chunks = make_chunks(tempSample, chunk_length=500)
        chunks = [x for i, x in enumerate(chunks)]
        chunk_range = []
        start = 0
        for i in range(len(chunks)-1):
            sample = chunks[i] + chunks[i+1]
            chunk_range.append((start, start+999))
            curr_chunk_path = os.path.join(chunksDir, f"input{chunkCount}.wav")
            sample.export(curr_chunk_path, format="wav")
            label = 0
            for segment in segments:
                if(start<=segment[0] and start+999>=segment[1]):
                    label = label or 1
                else:
                    label = label or 0
            row = pd.Series({
                "path": curr_chunk_path,
                "label": label,
            })
            df = df.append(row, ignore_index=True)
            start = start+500
            chunkCount = chunkCount+1
        sampleCount = sampleCount+1
        df.to_csv("input.csv", index=False)
    return df

def split_data(inputs, frac):
    rng = np.random.RandomState()
    training_data = inputs.sample(frac=frac, random_state=rng)
    testing_data = inputs.loc[~inputs.index.isin(training_data.index)]
    createDir(model_data_dir)
    training_data.to_csv(training_path, index=False)
    testing_data.to_csv(testing_path, index=False)
    return training_data, testing_data

if __name__ == "__main__":
    data = createData()
    inputs = createSamples(data)
    train, test = split_data(inputs, frac=0.7)
    print("train:\t", train.shape)
    print("test:\t", test.shape)