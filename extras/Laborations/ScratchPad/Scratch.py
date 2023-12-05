#%%
import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import matplotlib.pyplot as plt

n_samples = 1000

X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)

# Print untrained data
plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y,
            cmap=plt.cm.RdYlBu)


# %%
