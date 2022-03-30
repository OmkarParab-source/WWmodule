import os
from yaml import load, dump, Loader, Dumper

file = open("config.yml", 'r')
config = load(file, Loader=Loader)

# Data
Data = config["DATA"]
dir = os.path.join(os.getcwd(), Data["dir"])
positives = os.path.join(dir, Data["positives"])
negatives = os.path.join(dir, Data["negatives"])
backgrounds = os.path.join(dir, Data["backgrounds"])
Datainfo = {
    "dir" : dir,
    "positives": positives,
    "negatives": negatives,
    "backgrounds": backgrounds,
}

if __name__ == "__main__":
    print(Datainfo)