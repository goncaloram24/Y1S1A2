import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import random
from sklearn.cluster import MeanShift
from MyImplementations.Mean_Shift import Mean_Shift




centers = random.randrange(2, 5)
print(centers)

X, y = make_blobs(n_samples=75, centers=centers, n_features=2)
'''
X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11],
              [8, 2],
              [10, 2],
              [9, 3],])
'''
colors = 10*["g", "r", "c", "b", "k", "y", "m"]


clf_obj = Mean_Shift()

clf_obj.fit(X)

plt.figure()

plt.scatter(clf_obj.centroids[:, 0], clf_obj.centroids[:, 1], marker='*', s=150, linewidths=5)

plt.scatter(X[:, 0], X[:, 1], marker='x', c='k', s=150)

plt.show()









