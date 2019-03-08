# Naïve Bayes algorithm for Binary features and Binary class labels
# instead of computing P(x|c)P(c) compute log(P(x|c)+log(P(c))
# When making a prediction use the MAP technique.

import pandas as pd
import numpy as np
import math as m
import glob
import time

doBig = False

if doBig:
    files = glob.glob("./big/*")
else:
    files = glob.glob("./small/*")

for file in files:
    train = pd.read_csv(file)

    # separate training from test
    test = train.sample(frac=0.3)
    train.drop(test.index)

    # more preprocessing
    posSet = train[train['Class'] == 1]
    negSet = train[train['Class'] == -1]

    # start timer
    startTrain = time.time()

    # find P(CPlus) P(CMinus)
    # P(Xk | Ci) how likely Xk is given Ci
    xkGivenCPlus = []
    xkGivenCMin = []
    for col in train:
        if (col == 'Class'):
            break
        xkGivenCPlus.append(posSet[col].mean())
        xkGivenCMin.append(negSet[col].mean())

    cPlus = posSet.shape[0] / train.shape[0]
    cMinus = 1 - cPlus

    endTrain = time.time()

    # convert to log10 form
    xkGivenCPlus = [m.log(x, 10) if x != 0 else 0 for x in xkGivenCPlus]
    xkGivenCMin = [m.log(x, 10) if x != 0 else 0 for x in xkGivenCMin]

    cPlus = m.log(cPlus, 10) if cPlus != 0 else 0
    cMinus = m.log(cMinus, 10) if cMinus != 0 else 0

    startTest = time.time()

    # testing
    xkGivenCPlusLength = len(xkGivenCPlus) - 1
    correct = 0
    for index, row in test.iterrows():
        cPlusChance = cPlus
        cMinChance = cMinus
        for i in range(xkGivenCPlusLength):
            if row[i] == 1:
                cPlusChance += xkGivenCPlus[i]
                cMinChance += xkGivenCMin[i]
            else:
                cPlusChance += 1 - xkGivenCPlus[i]
                cMinChance += 1 - xkGivenCMin[i]

        # check if correct
        if ((cPlusChance > cMinChance) and row['Class'] == 1) or ((cMinChance >= cPlusChance) and row['Class'] == -1):
            correct += 1
    endTest = time.time()
    print(file + " Accuracy: " + str(correct / test.shape[0]) + " Training: " + str(
        endTrain - startTrain) + " Testing: " + str(endTest - startTest))
    # print("Test factor: "+str(test.size)+ " Train factor: " + str(train.size)+" attributes: "+str(len(test.columns)-1))
