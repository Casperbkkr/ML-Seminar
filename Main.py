#!/usr/bin/env python
import pandas as pd
import polars as pl
from Tree import Tree
from Forest import Forest
df = pl.read_csv('Iris.csv')
df.drop_in_place('Id')
#F = Forest(df, n_trees=5, target="Species")
df_inp = df[58]
#out = F.Predict(df_inp)

df2 = pl.read_csv("exams.csv")
F2 = Forest(df2, n_trees=3, target="writing score")

out2 = F2.Predict(df2[50])
print(out2)
print(df2.row(50))
x=1