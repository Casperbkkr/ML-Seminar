#!/usr/bin/env python
from Node import Node
import polars as pl
import math as mt

class Tree:  # TODO add auto classification or regression
    def __init__(self, df, *, target, depth=0, root=None, max_depth=3, id=1) -> None:
        self.id = id
        self.df = df
        self.target = target
        self.regression = self._Class_or_regression(df, target)
        self.Root = root
        self.Depth = depth
        self.thresh = 0
        if self.regression is True:
            self.split_criteria = self._Average_variance
        else:
            self.split_criteria = self._Average_entropy

        self.is_leaf = self._Leaf(df, target)
        if self.is_leaf is True or self.Depth >= max_depth:
            self.is_leaf = True
            self.val = self.df.select(pl.col(self.target).mode())[0, 0]
            self.attr = None

            #print(self.id)
            #print("Leaf node made with value: ", self.val)

            self.branch_left = None
            self.branch_right = None

        else:
            z, df_l, df_r, thr, attr = self._Split(df, target, self.split_criteria)
            self.val = thr
            self.attr = attr
            #print(self.id)
            #print("Split node made on attribute and value: ", self.attr, ",", self.val)
            idl = str(self.id) + "1"
            idr = str(self.id) + "2"
            self.branch_left = Tree(df_l, target=self.target, depth=self.Depth + 1, root=self, max_depth=max_depth, id=idl)
            self.branch_right = Tree(df_r, target=self.target, depth=self.Depth + 1, root=self, max_depth=max_depth, id=idr)


    def _Class_or_regression(self, df, target):
        if type(df.select(target).row(0)[0]) is str:
            return False
        else:
            return True

    def Predict(self, df_inp):
        if self.is_leaf is True:
            return self.val  # return prediction
        else:
            a = df_inp.select(self.attr).row(0)[0]
            if a <= self.val:
                return self.branch_left.Predict(df_inp)
            else:
                return self.branch_right.Predict(df_inp)


    def _Leaf(self, df, target) -> bool:
        # check if a df is pure
        a = df.groupby(target, maintain_order=True).agg(pl.count())
        if a.shape[0] == 1:
            return True
        else:
            return False


    def _Split(self, df, target, split_criteria) -> None:
        return self._Best_split(df, target, split_criteria)


    def _Entropy(self, df, target) -> float:
        # count occurrence of different categories in df and divide by total size to get pi of category
        a = df.select(pl.col(target).value_counts() / df.shape[0])
        b = a.to_dicts()

        # extract pi for categories
        P = [b[i][target]["counts"] for i in range(len(b))]

        # define function to calculate entropy
        plogp = lambda p: p*mt.log(p)
        entropy = (-1) * sum([plogp(pi) for pi in P])
        return entropy


    def _Average_entropy(self, df1, df2, n_parent, target) -> float:
        weight1 = df1.shape[0]/n_parent
        weight2 = df2.shape[0]/n_parent
        avg_ent = weight1*self._Entropy(df1, target) + weight2*self._Entropy(df2, target)
        return avg_ent


    def _Variance(self, df) -> float:
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
        #n_attr = df.shape[1] - 1
        attr_keys = list(df.schema.keys())[0:-1]

        f_split = lambda col: self._Possible_splits_row(df, col, target, split_criteria)
        possible_splits = [f_split(key) for key in attr_keys]
        split_val = [possible_splits[i][0] for i in range(len(possible_splits))]
        min_split_val = min(split_val)
        min_split_ind = split_val.index(min_split_val)

        return *possible_splits[min_split_ind], attr_keys[min_split_ind]

    def Training_Error(self, df_er):


        return


