#!/usr/bin/env python
import pandas as pd
import polars as pl
from Tree import Tree
df = pl.read_csv('iris.csv')
df.drop_in_place('Id')
T1 = Tree(df, 'Species')
c = T1._Possible_splits_row(T1.df, 'SepalWidthCm', 'Species', T1._Average_entropy)

T2 = Tree(df, 'PetalLengthCm')
d = T2._Possible_splits_row(T2.df, 'SepalLengthCm', 'PetalLengthCm', T2._Average_variance)
x=1

a = T2._Split(T2.df, "Species", T2._Average_entropy)