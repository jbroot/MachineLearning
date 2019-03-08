import pandas as pd
import numpy as np
import sklearn
import sklearn.metrics
import matplotlib.pyplot as plt

original = pd.read_csv("diabetes.csv")

doRollingAvg = True
#else will just take the average from each kfold
numberToRoll = 400
saveGraphs = False

# shuffles rows
df = original.sample(frac=1)

# split into inputs and class
results = df['class']
df = df.drop(['class'], axis=1)
results = results.replace(-1, 0)

# clean to ndArrays
df = df.values
results = results.values[np.newaxis].T

learningRates = [0.5, 0.1, 0.01, 0.001, 0.0001]
numInputs = df.shape[1]
kFold = 5
epochs = 4000
testSize = round((df.shape[0] / kFold))

accuracyByN = []
for learningRate in learningRates:
    avgAccuracy = []
    avgKAcc = np.zeros((epochs, 1))
    # use k-fold cross-validation (3-10)
    for i in range(kFold):
        accuracy = []
        weights = np.random.rand(numInputs + 1, 1) * 0.1 - .05
        # test range
        minTest = i * testSize
        maxTest = minTest + testSize

        # divide training and test
        xTrain = np.concatenate([df[:minTest], df[maxTest - 1:]])
        xTest = df[minTest + 1:maxTest]
        yTrain = np.concatenate([results[:minTest], results[maxTest - 1:]])
        yTest = results[minTest + 1:maxTest]

        # normalization
        xTrainMean = xTrain.mean(axis=0)
        xTrainVar = xTrain.var(axis=0)
        xTest = (xTest - xTrainMean) / xTrainVar
        xTrain = (xTrain - xTrainMean) / xTrainVar

        # add bias input
        bias = np.full(xTrain[:, :1].shape, -1)
        xTrain = np.append(xTrain, bias, axis=1)

        bias = np.full(xTest[:, :1].shape, -1)
        xTest = np.append(xTest, bias, axis=1)

        # perceptron algorithm
        for T in range(epochs):
            # train
            fireTrain = np.dot(xTrain, weights)
            fireTrain = np.where(fireTrain >= 0, 1, 0)
            weights -= learningRate * np.dot(xTrain.T, fireTrain - yTrain)

            # test
            fire = np.dot(xTest, weights)
            fire = np.where(fire >= 0, 1, 0)
            acc = sklearn.metrics.accuracy_score(fire, yTest)

            if doRollingAvg:
                accuracy.append(acc)
                if len(accuracy) >= numberToRoll:
                    avgKAcc[T] += (sum(accuracy[-numberToRoll:]) / numberToRoll)
                else:
                    avgKAcc[T] += (sum(accuracy) / len(accuracy))
            else:
                avgKAcc[T] += acc
    accuracyByN.append(avgKAcc / kFold)

# Graph accuracy score as a function of the number of training epochs.
# 5 graphs for different learning rates
for i in range(len(learningRates)):
    plt.figure(i)
    plt.plot(accuracyByN[i])
    plt.title("Learning rate: " + str(learningRates[i]))
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.show()
    if saveGraphs:
        plt.savefig('Learning_rate_' + str(learningRates[i]) + '.png')
