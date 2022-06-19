import pandas as pd

import utils


class perceptron:
    def __init__(self, ):
        pass

    def _init_weight(self, X, y):
        self.weights = pd.DataFrame([0.0 for i in range(len(X.columns) + 1)])
        self.X['threshold'] = -1
        self.y = y

    def train(self, learning_rate=0.1, epoch=10, activation=utils.sigmoid):
        for epo in range(epoch):
            predicted = self.X.dot(self.weights.to_numpy()).apply(activation, axis=1)
            error = self.y.subtract(predicted, axis=0)
            for i in range(len(self.X)):
                for j in range(self.weights):
                    self.weights.iloc[j, 0] += learning_rate * error.iloc[i, 0] * self.X.iloc[i, j]

    def predict(self, test, activation=utils.sigmoid):
        predict = test.dot(self.weights.to_numpy()).apply(activation, axis=1)
        return predict
