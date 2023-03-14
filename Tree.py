#!/usr/bin/env python
from Node import Node
import polars as pl
import math as mt

class Tree:
    def __init__(self, df, *, target, depth=0, root=None, max_depth=3) -> None:
        self.df = df
        self.target = target
        self.Root = root
        self.Depth = depth
        self.thresh = 0
        self.split_criteria = self._Average_entropy
        self.is_leaf = self._Leaf(df, target)
        if self.is_leaf is True or self.Depth > max_depth:
            self.branch_left = None
            self.branch_right = None
            self.val = self.df.select(pl.col(self.target).mode())[0, 0]
            a=1
        else:
            z, df_l, df_r, thr = self._Split(df, target, self.split_criteria)
            self.branch_left = Tree(df_l, target=self.target, depth=self.Depth + 1, root=self, max_depth=max_depth)
            self.branch_right = Tree(df_r, target=self.target, depth=self.Depth + 1, root=self, max_depth=max_depth)
            self.val = thr

    def _Leaf(self, df, target) -> bool:
        # check if a df is pure
        a = df.groupby(target, maintain_order=True).agg(pl.count())
        if a.shape[0] == 1:
            return True
        else:
            return False


    def _Split(self, df, target, split_criteria) -> None:
        print("made a split")
        return self._Best_split(df, target, split_criteria)

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


    def _Average_entropy(self, df1, df2, n_parent, target):
        weight1 = df1.shape[0]/n_parent
        weight2 = df2.shape[0]/n_parent
        avg_ent = weight1*self._Entropy(df1, target) + weight2*self._Entropy(df2, target)
        return avg_ent


    def _Variance(self, df):
        target = self.target
        return df.select(target).var()[0, 0]


    def _Average_variance(self, df1, df2, n_parent, target):

        # calculate variance of the split dataset
        var1 = self._Variance(df1)
        if var1 == None: Var1 = 0
        weight1 = df1.shape[0]/n_parent

        # calculate variance of its complement
        var2 = self._Variance(df2)
        if var2 == None: var2 = 0
        weight2 = df2.shape[0]/n_parent

        return weight1 * var1 + weight2 * var2


    def _Possible_splits_row(self, df, column, target, split_criteria):
        n_splits = df.shape[0]
        # look up all possible data points to split over
        split_value = [df.select(column)[i][0, 0] for i in range(n_splits)]

        # create all splitted dataframes.
        split = [df.filter(pl.col(column) <= val) for val in split_value]
        split_comp = [df.filter(pl.col(column) > val) for val in split_value]

        split_f = lambda i: split_criteria(split[i], split_comp[i], n_splits, target)
        weighted_splits = [split_f(i) for i in range(len(split))]

        # find minimum weighted variance
        min_split = min(weighted_splits)
        index = weighted_splits.index(min_split)

        return min_split, split[index], split_comp[index], split_value[index]


    def _Best_split(self, df, target, split_criteria):
        n_attr = df.shape[1] - 1
        attr_keys = list(df.schema.keys())[0:-1]

        f_split = lambda col: self._Possible_splits_row(df, col, target, split_criteria)
        possible_splits = [f_split(key) for key in attr_keys]
        split_val = [possible_splits[i][0] for i in range(len(possible_splits))]
        min_split_val = min(split_val)
        min_split_ind = split_val.index(min_split_val)

        return possible_splits[2]

