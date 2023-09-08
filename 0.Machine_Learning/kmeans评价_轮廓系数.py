import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt 
import random
from sklearn.metrics import pairwise_distances

n_sample=50
n_feature=5
n_cluster=3

X = np.random.rand(n_sample,n_feature)
Y = np.random.choice(range(n_cluster),size=n_sample) # random assign cluster label
score = silhouette_score(X, Y, metric='euclidean')
print("sklearn calculate silhouette:",score)
