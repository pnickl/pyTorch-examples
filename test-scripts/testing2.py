import matplotlib as plt
from itertools import combinations
from sklearn.datasets import make_classification,make_regression
import numpy as np
import matplotlib.pyplot as plt

input_dim = 5
data_points = 100
X, y = make_classification(data_points, input_dim, n_informative=3, random_state=101)
X = X.astype(np.float32)
y = y.astype(np.float32)

comb_list = list(combinations([v for v in range(5)],2))

fig, ax  = plt.subplots(5,2,figsize=(10,18))
axes = ax.ravel()
for i,c in enumerate(comb_list):
    j,k = c
    axes[i].scatter(X[:,j],X[:,k],c=y,edgecolor='k',s=200)
    axes[i].set_xlabel("X"+str(j),fontsize=15)
    axes[i].set_ylabel("X"+str(k),fontsize=15)
plt.show()