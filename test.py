from os import pipe
from random import shuffle
import math
import subprocess


def rmse(y, yhat):
    sum_result = 0
    for i in range(len(y)):
        sum_result += (float(y[i])-float(yhat[i]))*(float(y[i])-float(yhat[i]))
    return math.sqrt(sum_result/len(y))


datasetFile = open("./dataset/train.csv", "r")

datasetLines = datasetFile.read().split("\n")
datasetFile.close()
shuffle(datasetLines[1:len(datasetLines)])


def comma_separate_line(dataset): return [line.split(",") for line in dataset]


trainSet = comma_separate_line(datasetLines[0:17011])
testSet = comma_separate_line(datasetLines[17011:len(datasetLines)])

input = ""


input += "\n".join(["\t".join(line[0:81]) for line in trainSet])+"\n"
input += "\n".join([line[81] for line in trainSet])+"\n"
input += "\n".join(["\t".join(line[0:81]) for line in testSet])+"\n"
input += "\n".join([line[81] for line in testSet])

p = subprocess.Popen(["python", "./solution.py"], stdout=subprocess.PIPE,
                     stdin=subprocess.PIPE, stderr=subprocess.PIPE)

solutionOutput = p.communicate(input=input.encode())[0].decode()

predictions = solutionOutput.split("\n")[0:-1]

print(predictions)

# print(rmse(predictions,[line[81] for line in testSet]))
