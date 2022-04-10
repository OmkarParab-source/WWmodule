import os
import pandas as pd
from numpy.random import RandomState
from yaml import load, dump, Loader, Dumper

file = open("config.yml", 'r')
config = load(file, Loader=Loader)

Data = config["Data"]
Datainfo = {
    "dir": os.path.join(os.getcwd(), Data["dir"]),
    "positives": os.path.join(os.getcwd(), Data["dir"], Data["positives"]),
    "negatives": os.path.join(os.getcwd(), Data["dir"], Data["negatives"]),
    "backgrounds": os.path.join(os.getcwd(), Data["dir"], Data["backgrounds"]),
    "csv_files": os.path.join(os.getcwd(), Data["nn"], "csv_files")
}

Seed = config["Seed"]

NNparams = config["NeuralNet"]
if __name__ == "__main__":
    df = pd.read_csv('input.csv')
    rng = RandomState()

    train = df.sample(frac=0.7, random_state=rng)
    test = df.loc[~df.index.isin(train.index)]

    train.to_csv(os.path.join(Datainfo["csv_files"], "train.csv"), index=False)
    test.to_csv(os.path.join(Datainfo["csv_files"], "test.csv"), index=False)

    print("train:\t",train.shape)
    print("test:\t", test.shape)

