import numpy as np
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from statistics import mode


def Mean_Shift(data, SBS, C, EP, CP, selected_products):
  ms = MeanShift(bandwidth=2)
  ms.fit(data)
  
  n_clusters = len(ms.cluster_centers_)

  EP_Length = len(EP)
  # list of lists
  arr = [[] for i in range(n_clusters)]
  for i,j in enumerate(ms.labels_):
    arr[j].append(i)
  
  cluster_nos_of_selected_products = [ms.labels_[i] for i in selected_products]

  # Run over the cluster from which majority of the products have been selected previously.
  cluster = max(set(cluster_nos_of_selected_products), key = cluster_nos_of_selected_products.count)

  EP_New, CP_New = [], []
  for i in arr[cluster]:
    if i<EP_Length:
      EP_New.append(i)
    else:
      CP_New.append(i)

  return EP_New, CP_New
