#!/usr/bin/env python
import pandas as pd
import polars as pl
from Tree import Tree
df = pl.read_csv('iris.csv')

T = Tree(df, 'Species')

a = T._Entropy(T.df, 'Species')
b = T._Average_entropy(T.df, T.df, 'Species')
c = T._Possible_splits_col(T.df, 'SepalLengthCm')
print(b)

x=1