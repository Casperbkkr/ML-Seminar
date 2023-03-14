#!/usr/bin/env python
import pandas as pd
import polars as pl
from Tree import Tree
df = pl.read_csv('exams.csv')
#df.drop_in_place('Id')
T1 = Tree(df, target='parental level of education')


x=1