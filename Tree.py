#!/usr/bin/env python
from Node import Node
import polars as pl

class Tree:
    def __init__(self, df) -> None:
        self.df = df
        pass

    def _Split(self) -> None:

        pass

    def _Entropy(self):
        
        return

    def _Calc_pi(self, df):
        return

    def _Variance(self, df1):
        return df1.var()[0, 0]

    def _Possible_splits_col(self, df, column, split_criteria=1):
        n_splits = df.shape[0]
        # look up all possible data points to split over
        split_value = [df.select(column)[i][0, 0] for i in range(n_splits)]
        # create all splitted dataframes.
        splitted = [df.filter(pl.col(column) <= val) for val in split_value]

        splitted_comp = [df.filter(pl.col(column) > val) for val in split_value]
        var1 = list(map(self._Variance, splitted))
        var12 = [x if x is not None else 0 for x in var1]
        weight1 = [split.shape[0]/n_splits for split in splitted]
        var2 = list(map(self._Variance, splitted_comp))
        var22 = [x if x is not None else 0 for x in var2]
        weight2 = [split.shape[0]/n_splits for split in splitted_comp]
        zipped = zip(weight1, var12, weight2, var22)
        weighted_var = [w1*v1 + w2*v2 for w1, v1, w2, v2 in zipped]
        min_var = min(weighted_var)
        return


