import numpy as np
from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt
from statistics import mode


def Affinity_Propagation(data, SBS, C, EP, CP, selected_products):
  ap = AffinityPropagation(preference=-200)
  ap.fit(data)
  
  n_clusters = len(ap.cluster_centers_)

  EP_Length = len(EP)
  # list of lists
  arr = [[] for i in range(n_clusters)]
  for i,j in enumerate(ap.labels_):
    arr[j].append(i)
  
  cluster_nos_of_selected_products = [ap.labels_[i] for i in selected_products]

  # Run over the cluster from which majority of the products have been selected previously.
  cluster = max(set(cluster_nos_of_selected_products), key = cluster_nos_of_selected_products.count)

  EP_New, CP_New = [], []
  for i in arr[cluster]:
    if i<EP_Length:
      EP_New.append(i)
    else:
      CP_New.append(i)

  return EP_New, CP_New, n_clusters
