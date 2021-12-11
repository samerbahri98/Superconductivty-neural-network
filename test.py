from os import pipe
from random import shuffle
import math
import subprocess

def rmse (y,yhat):
    sum_result = 0
    for i in range(len(y)):
        sum_result+=(float(y[i])-float(yhat[i]))*(float(y[i])-float(yhat[i]))
    return math.sqrt(sum_result/len(y))

datasetFile = open("./dataset/train.csv","r")

datasetLines=datasetFile.read().split("\n")
datasetFile.close()
shuffle(datasetLines)

commaSeparateLine = lambda dataset : [line.split(",") for line in dataset]

trainSet = commaSeparateLine(datasetLines[0:17011])
testSet = commaSeparateLine(datasetLines[17011:len(datasetLines)])

input =""


input += "\n".join(["\t".join(line[0:81]) for line in trainSet])+"\n"
input += "\n".join([line[81] for line in trainSet])+"\n"
input += "\n".join(["\t".join(line[0:81]) for line in testSet])+"\n"
input += "\n".join([line[81] for line in testSet])

p = subprocess.Popen(["python","./solution.py"],stdout=subprocess.PIPE,stdin=subprocess.PIPE,stderr=subprocess.PIPE)

solutionOutput = p.communicate(input=input.encode())[0].decode()

predictions = solutionOutput.split("\n")[0:-1]

print(rmse(predictions,[line[81] for line in testSet]))

# print(predictions,[line[81] for line in testSet])


# print (input)