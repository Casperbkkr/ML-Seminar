#!/usr/bin/env python
import matplotlib.pyplot as plt
import polars as pl
from Tree import Tree
from Forest import Forest

df = pl.read_csv('exams.csv')
#df.drop_in_place('Id')


df_sh = df.sample(n=df.shape[0], with_replacement=False, shuffle=True)
#print(df_sh)
training_size = 30
test_size = df_sh.shape[0] - training_size
df_tr = df_sh[:training_size]
df_er = df_sh[training_size:]

T1 = Tree(df_tr, target="writing score", max_depth=2)
out = []
for depth in range(1, 10):

	T1 = Tree(df_tr, target="writing score", max_depth=depth)
	S = 0
	for i in range(df_er.shape[0]):
		pred = T1.Predict(df_er[i])
		if pred == df_er.row(i)[-1]:
			S += 1
		else:
			continue

	accuracy = S/df_er.shape[0]
	out.append(accuracy)


plt.plot(out)
plt.show()


x=1