import numpy as np
import pandas as pd
"""
Entropy calculates the purity of data. 
E = sum(p_k * log(p_k)) where 
p_k : the proportion of kth kind data in the data set
sum sums over the total kinds of data
"""
def entropy(x):
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()
    property_count = {}
    for i in range(len(x)):
        property_count[x[i]] = 0

    for i in range(len(x)):
        property_count[x[i]] += 1
    enp = .0
    for k, v in property_count.items():
        enp -= v/len(x)*np.log2(v/len(x))
    return enp


"""
infoGain computes the gain from certain property
x is the given property
Suppose a dataset D has k properties {D_1, D_2 ... D_k}
infoGain is computed by Ent(D) - sum(|D^v|/|D| Ent(D^k))
"""

def infoGain(x, y):
    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        x = x.to_numpy()
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y = y.to_numpy()

    enp = .0
    # set(x) calculates the different kinds in the given propoerty
    for prop in set(x):
        prop_idx = np.where(x == prop)
        prop_x = x[prop_idx]
        prop_y = y[prop_idx]
        enp += len(prop_x) / len(x) * entropy(prop_y)
    return entropy(y) - enp


def gainRatio(x, y):
    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        x = x.to_numpy()
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y = y.to_numpy()

    iv = .0
    for prop in set(x):
        prop_idx = np.where(x == prop)
        prop_x = x[prop_idx]
        iv -= len(prop_x) / len(x) * np.log2(len(prop_x) / len(x))

    return infoGain(x, y) / iv
