import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from statistics import mode


"""***************Attempting to import relatively*****************UN-SUCCESSFULL"""
# from algorithms.SPG import SPG
# from algorithms.IG import IG
# from algorithms.UBP import UBP
# from algorithms.Apriori import Apriori



def calc_KMean_clusters(SBS):
  cost =[] 
  # calculatig for 1-10 clusters
  for i in range(1, 10): 
      KM = KMeans(n_clusters = i, max_iter = 200) 
      KM.fit(SBS) 
      cost.append(KM.inertia_) 
  # Plotting the cost function
  plt.plot(range(1, 10), cost, linewidth ='3') 
  plt.xlabel("Value of K") 
  plt.ylabel("Sqaured Error (Cost)") 
  plt.show()
  return True


def K_Means(data, SBS, C, EP, CP, selected_products, n_clusters = 5):
  kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
  
  EP_Length = len(EP)
  # list of lists
  arr = [[] for i in range(n_clusters)]
  arrEP = [[] for i in range(n_clusters)]
  for i,j in enumerate(kmeans.labels_):
    arr[j].append(i)
    if(i < EP_Length):
      arrEP[j].append(i)

  # Clusters = [[] for i in range(n_clusters)]
  # for i in range(n_clusters):
  #   Clusters[i] = data[arr[i],:]
  # Print the no. of products each cluster contains.
  # for i in range(n_clusters):
    # print("%d: %d"% (i,len(Clusters[i])), end='\n')

  cluster_nos_of_selected_products = [kmeans.labels_[i] for i in selected_products]

  # Run over the cluster from which majority of the products have been selected previously.
  cluster = max(set(cluster_nos_of_selected_products), key = cluster_nos_of_selected_products.count)
  # cluster = mode(cluster_nos_of_selected_products)
  # print('Selected cluster:', cluster)

  SBS_New = SBS[arr[cluster]]
  # SBS_New = np.append(SBS_New, SBS[EP,:], axis=0)

  EP_New = range(len(arrEP))
  CP_NEW = range(len(arrEP), len(SBS_New))

  return SBS_New, EP_New, CP_NEW
  # rows,cols = SBS_New.shape

  # Creating Sets of Customers, Candidate Products and Existing Products.
  # C = np.arange(0, cols)
  """****************************NEED TO FOCUS***********************"""
  # EP = np.arange(0, len(Clusters[cluster]))
  # CP = np.arange(len(Clusters[cluster]), rows)


  # SPGA
  # timeTaken = datetime.datetime.now()
  # CSPG_k = SPG(k, SBS_New, EP, CP)
  # timeTaken = datetime.datetime.now() - timeTaken 
  # print("Single Product Based Greedy Algorithm : \n", CSPG_k)
  # print("Time taken:", timeTaken.seconds)

  # IG
  # timeTaken = datetime.datetime.now()
  # CIG_k = IG(k, SBS_New, EP, CP)
  # timeTaken = datetime.datetime.now() - timeTaken
  # print("Incremental Based Greedy Algorithm : \n", CIG_k)
  # print("Time taken:", timeTaken.seconds)

  # # UBP
  # timeTaken = datetime.datetime.now()
  # CUBP_k = UBP(k, SBS_New, EP, CP)
  # timeTaken = datetime.datetime.now() - timeTaken
  # print("Upper Bound Pruning Algorithm : \n", CUBP_k)
  # print("Time taken:", timeTaken.seconds)

  # return True
