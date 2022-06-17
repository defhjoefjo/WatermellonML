from .. import utils
import pandas as pd
import numpy as np

class DecisionTree:
    class Node:
        def __init__(self, type, child = {}):
            self.child = child
            self.type = type

    def __init__(self, type = "ID3"):
        self.type = type

    def _build_tree(self, node, df, y_col):
        y = df[:, y_col]
        x = df[:, df.columns != y_col]
        cur_node = self.Node()
        cur_node.child = {}
        bestGain = 0
        bestprop = None
        gainFunc = utils.infoGain
        for props in x.columns.values.tolist():
            curGain = gainFunc(x[:,props], y)
            if (curGain > bestGain):
                bestGain = curGain
                bestprop = props

        for v in pd.unique(x[bestprop]):
            gb = df[df[bestprop] == v].groupby(y_col)
            if len(df.columns) > 2:
                if len(gb) == 1:
                    return self.Node(None, df[df[bestprop]==v].mode())
                else:
                    cur_node.type = '%s, %s' % (bestprop, v)
                    child_node = self._build_tree(cur_node, df.drop(bestprop, axis = 1), y_col)
                    cur_node.child.append(child_node)
            else:
                return self.Node(None, df[df[bestprop]==v].mode())

    def train(self, df):
        y_col = df.columns.values.tolist()[-1]
        root = self.Node("root", df, y_col)
        return root








