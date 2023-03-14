#!/usr/bin/env python
from Tree import Tree
from statistics import mean

#TODO add annotation
class Forest:
    def __init__(self, df, *, target, n_trees=10) -> None:
        self.df = df
        self.regression = self._Class_or_regression(df, target)
        self.bootstraps = self._Bootstrap(df, n_trees)
        self.Forest = [Tree(df_sample, target=target) for df_sample in self.bootstraps]


    def _Class_or_regression(self, df, target):
        if type(df.select(target).row(0)[0]) is str:
            return False
        else:
            return True

    def _Bootstrap(self, df, n_trees) -> list:
        # TODO add bootstrap error estimation
        return [df.sample(n=df.shape[0], with_replacement=True) for i in range(n_trees)]


    def Predict(self, df_inp):
        if self.regression is True:
            return self._Reg_Predict(df_inp)
        else:
            return self._Class_predict(df_inp)


    def _Class_predict(self, df_inp):
        #TODO add auto regression or classification
        all_pred = [T.Predict(df_inp) for T in self.Forest]
        mode = max(set(all_pred), key=all_pred.count)
        return mode

    def _Reg_Predict(self, df_inp):
        all_pred = [T.Predict(df_inp) for T in self.Forest]
        return mean(all_pred)



