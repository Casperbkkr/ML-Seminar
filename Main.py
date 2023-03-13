#!/usr/bin/env python
import pandas as pd
import polars as pl
from Tree import Tree
df = pl.read_csv('iris.csv')

T = Tree(df)

a = T._Possible_splits_col(T.df, "SepalLengthCm", 1)



x=1