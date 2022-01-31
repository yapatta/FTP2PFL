import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys

plt.xlabel("Elapsed time [sec]")
plt.ylabel("Accuracy [%]")

TESTNAME = sys.argv[1]
TESTSEC = sys.argv[2]
nodes = [4, 6, 8, 10]
colors = ["b", "g", "r", "c"]

for node, color in zip(nodes, colors):
    filename = "./{}/{}-{}/good.csv".format(TESTNAME, node, TESTSEC)
    df = pd.read_csv(filename, header=None, names=('time', 'accuracy', 'loss'))
    df["time"] = pd.to_datetime(df['time'])
    ini = df["time"][0]
    df = df.drop(index=0)
    df["accuracy"] = pd.to_numeric(df["accuracy"])
    df["loss"] = pd.to_numeric(df["loss"])
    df["time"] = (df["time"] - ini).dt.total_seconds()
    nodestr = "node: {}".format(node)
    plt.plot(df["time"], df["accuracy"], color=color, label=nodestr)

plt.grid()
plt.legend()

resultfile = "./accuracy/{}-{}-accuracy.pdf".format(TESTNAME, TESTSEC)
plt.savefig(resultfile)
