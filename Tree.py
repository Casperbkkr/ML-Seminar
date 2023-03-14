#!/usr/bin/env python
from Node import Node
import polars as pl
import math as mt
class Tree:
    def __init__(self, df, target) -> None:
        self.df = df
        self.target = target


    def _Split(self) -> None:

        pass

    def _Entropy(self, df, target):
        # count occurrence of different categories in df and divide by total size to get pi of category
        a = df.select(pl.col(target).value_counts() / df.shape[0])
        b = a.to_dicts()

        # extract pi for categories
        P = [b[i][target]["counts"] for i in range(len(b))]

        # define function to calculate entropy
        plogp = lambda p: p*mt.log(p)
        entropy = (-1) * sum([plogp(pi) for pi in P])
        return entropy

    def _Average_entropy(self, df1, df2, target):
        n_parent = df1.shape[0] + df2.shape[0]
        weight1 = df1.shape[0]/n_parent
        weight2 = df2.shape[0]/n_parent
        avg_ent = weight1*self._Entropy(df1, target) + weight2*self._Entropy(df2, target)
        return avg_ent

    def _Calc_pi(self, df):
        return

    def _Variance(self, df):
        target = self.target
        return df.select(target).var()[0, 0]

    def _Average_variance(self, df1, df2, n_splits):
        # calculate variance of the splitted dataset
        var1 = list(map(self._Variance, df1))
        var12 = [x if x is not None else 0 for x in var1]
        weight1 = [split.shape[0] / n_splits for split in df1]


    def _Possible_splits_col(self, df, column, split_criteria=1):
        n_splits = df.shape[0]
        # look up all possible data points to split over
        split_value = [df.select(column)[i][0, 0] for i in range(n_splits)]

        # create all splitted dataframes.
        splitted = [df.filter(pl.col(column) <= val) for val in split_value]
        splitted_comp = [df.filter(pl.col(column) > val) for val in split_value]

        # calculate variance of the splitted dataset
        var1 = list(map(self._Variance, splitted))
        var12 = [x if x is not None else 0 for x in var1]
        weight1 = [split.shape[0]/n_splits for split in splitted]

        # calculate variance of its complement
        var2 = list(map(self._Variance, splitted_comp))
        var22 = [x if x is not None else 0 for x in var2]
        weight2 = [split.shape[0]/n_splits for split in splitted_comp]

        # zip together for processing
        zipped = zip(weight1, var12, weight2, var22)

        # calculate the weighted variance
        weighted_var = [w1*v1 + w2*v2 for w1, v1, w2, v2 in zipped]
        # find minimum weighted variance
        min_var = min(weighted_var)
        index = weighted_var.index(min_var)

        return min_var, splitted[index], splitted_comp[index]


