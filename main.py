from dataManager import createData, createSBS, readData, readSBS
from algorithms.IG import IG
from algorithms.SPG import SPG
from algorithms.Apriori import Apriori
from algorithms.UBP import UBP
from clustering import KMeans, AgglomerativeHierarchical, AffinityPropagation, MeanShift
from sampling import sampling
import numpy as np
import time


# ====================================FILE HANDLING======================================

# Change the fileLocation to your car.data folder
dataLocation = "./data_process/car.csv"
sbsLocation = "./data_process/car_SBS.csv"
datasetLocation = "./dataset/car/car.data"

try:
    data = readData(dataLocation) 
    SBS = readSBS(sbsLocation)
except:
    data = createData(datasetLocation) 
    SBS = createSBS(data, datasetLocation)


# =======================================GLOBALS=========================================

rows=len(SBS)
cols=len(SBS[0])

# Creating IDs of Customers, Candidate Products and Existing Products.
C = np.arange(0, cols)
EP = np.arange(0, (rows*30)//100)
CP = np.arange((rows*30)//100, rows)

# Number of products to select
k = 5


# =================================BEFORE CLUSTERING=====================================

print('BEFORE CLUSTERING:')
# IG
timeTaken = int(round(time.time() * 1000))
selectedProdsIG, productScore = IG(k, C, SBS, EP, CP)
timeTaken = int(round(time.time() * 1000)) - timeTaken
print("Incremental Based Greedy Algorithm : \n", selectedProdsIG)
print("Time taken in millis:", timeTaken)
# print("Product score:", productScore)

# SPGA
timeTaken = int(round(time.time() * 1000))
selectedProdsSPG, productScore = SPG(k, C, SBS, EP, CP)
timeTaken = int(round(time.time() * 1000)) - timeTaken 
print("Single Product Based Greedy Algorithm : \n", selectedProdsSPG)
print("Time taken in millis:", timeTaken)
# print("Product score:", productScore)


# =====================================SAMPLING==========================================

sampledEP, sampledCP = sampling(EP, CP)

bestSampledProds, productScore = SPG(k*2, C, SBS, sampledEP, sampledCP)
# print("Best sampled products:", bestSampledProds)


# =================================AFFINITY PROPAGATION==================================

timeTaken = int(round(time.time() * 1000))
EP_New, CP_New, n_clusters = AffinityPropagation.Affinity_Propagation(data, SBS, C, EP, CP, bestSampledProds)
timeTaken = int(round(time.time() * 1000)) - timeTaken
print('Affinity Propagetion time:', timeTaken)

print('AFTER AFFINITY PROPAGATION:')
# IG

selectedProdsIG, productScore = IG(k, C, SBS, EP_New, CP_New)
timeTaken = int(round(time.time() * 1000)) - timeTaken
print("Incremental Based Greedy Algorithm : \n", selectedProdsIG)
print("Time taken in millis:", timeTaken)
# print("Product score:", productScore)

# SPG
timeTaken = int(round(time.time() * 1000))
selectedProdsSPG, productScore = SPG(k, C, SBS, EP_New, CP_New)
timeTaken = int(round(time.time() * 1000)) - timeTaken 
print("Single Product Based Greedy Algorithm : \n", selectedProdsSPG)
print("Time taken in millis:", timeTaken)
# print("Product score:", productScore)


# =================================MEAN SHIFT CLUSTERING=================================
timeTaken = int(round(time.time() * 1000))
EP_New, CP_New = MeanShift.Mean_Shift(data, SBS, C, EP, CP, bestSampledProds)
timeTaken = int(round(time.time() * 1000)) - timeTaken
print('Mean Shift time:', timeTaken)

print('AFTER MEAN SHIFT CLUSTERING:')
# IG
timeTaken = int(round(time.time() * 1000))
selectedProdsIG, productScore = IG(k, C, SBS, EP_New, CP_New)
timeTaken = int(round(time.time() * 1000)) - timeTaken
print("Incremental Based Greedy Algorithm : \n", selectedProdsIG)
print("Time taken in millis:", timeTaken)
# print("Product score:", productScore)

# SPG
timeTaken = int(round(time.time() * 1000))
selectedProdsSPG, productScore = SPG(k, C, SBS, EP_New, CP_New)
timeTaken = int(round(time.time() * 1000)) - timeTaken 
print("Single Product Based Greedy Algorithm : \n", selectedProdsSPG)
print("Time taken in millis:", timeTaken)
# print("Product score:", productScore)


# =================================KMEANS CLUSTERING=====================================

timeTaken = int(round(time.time() * 1000))
EP_New, CP_New = KMeans.K_Means(data, SBS, C, EP, CP, bestSampledProds, n_clusters)
timeTaken = int(round(time.time() * 1000)) - timeTaken
print('Kmeans time:', timeTaken)

print('AFTER KMEANS CLUSTERING:')
# IG
timeTaken = int(round(time.time() * 1000))
selectedProdsIG, productScore = IG(k, C, SBS, EP_New, CP_New)
timeTaken = int(round(time.time() * 1000)) - timeTaken
print("Incremental Based Greedy Algorithm : \n", selectedProdsIG)
print("Time taken in millis:", timeTaken)
# print("Product score:", productScore)

# SPG
timeTaken = int(round(time.time() * 1000))
selectedProdsSPG, productScore = SPG(k, C, SBS, EP_New, CP_New)
timeTaken = int(round(time.time() * 1000)) - timeTaken 
print("Single Product Based Greedy Algorithm : \n", selectedProdsSPG)
print("Time taken in millis:", timeTaken)
# print("Product score:", productScore)


# ================================AGGLOMERATIVE CLUSTERING===============================

timeTaken = int(round(time.time() * 1000))
EP_New, CP_New = AgglomerativeHierarchical.Agglomerative_Clustering(data, SBS, C, EP, CP, bestSampledProds, n_clusters)
timeTaken = int(round(time.time() * 1000)) - timeTaken
print('agglomertaive  time:', timeTaken)

print('AFTER AGGLOMERATIVE CLUSTERING:')
# IG
timeTaken = int(round(time.time() * 1000))
selectedProdsIG, productScore = IG(k, C, SBS, EP_New, CP_New)
timeTaken = int(round(time.time() * 1000)) - timeTaken
print("Incremental Based Greedy Algorithm : \n", selectedProdsIG)
print("Time taken in millis:", timeTaken)
# print("Product score:", productScore)

# SPG
timeTaken = int(round(time.time() * 1000))
selectedProdsSPG, productScore = SPG(k, C, SBS, EP_New, CP_New)
timeTaken = int(round(time.time() * 1000)) - timeTaken 
print("Single Product Based Greedy Algorithm : \n", selectedProdsSPG)
print("Time taken in millis:", timeTaken)
# print("Product score:", productScore)
