import utils
import pandas as pd


class DecisionTree:
    class Node:
        def __init__(self, type=""):
            self.child = list()
            self.type = type

    def __init__(self, type="ID3"):
        self.type = type

    def _build_tree(self, node, df, y_col):
        y = df[y_col]
        x = df.loc[:, df.columns != y_col]
        bestGain = 0
        bestprop = None
        gainFunc = utils.infoGain
        for props in x.columns.values.tolist():
            curGain = gainFunc(x.loc[:, props], y)
            if (curGain > bestGain):
                bestGain = curGain
                bestprop = props

        for v in pd.unique(x[bestprop]):
            gb = df[df[bestprop] == v].groupby(y_col)
            cur_node = self.Node('%s - %s' % (bestprop, v))
            node.child.append(cur_node)
            if len(df.columns) > 2:
                if len(gb) == 1:
                    cur_node.child.append(self.Node(df[df[bestprop] == v][y_col].mode()[0]))
                else:
                    self._build_tree(cur_node, df.drop(bestprop, axis=1), y_col)
            else:
                cur_node.child.append(self.Node(df[df[bestprop] == v][y_col].mode()[0]))

    def train(self, df):
        y_col = df.columns.values.tolist()[-1]
        root = self.Node("root")
        self._build_tree(root, df, y_col)
        return root


t = DecisionTree()
df = pd.read_csv("watermellon.txt")
df = df.drop("编号", axis = 1)
root = t.train(df)
for c in root.child:
    print(c.type)
