import math


def sgn(x):
    if (x >= 0):
        return 1
    else:
        return 0


def sigmoid(x):
    return 1 / (1 + math.exp(-x))
