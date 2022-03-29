from yaml import load, dump, Loader, Dumper

file = open("config.yml", 'r')
config = load(file, Loader=Loader)

Datainfo = config["DATA"]