from dataManager import createData, createSBS, readData, readSBS
from algorithms.IG import IG
from algorithms.SPG import SPG
from algorithms.Apriori import Apriori
from algorithms.UBP import UBP
from clustering import KMeans
from sampling import sampling
import numpy as np
import datetime

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

rows=len(SBS)
cols=len(SBS[0])

# Creating IDs of Customers, Candidate Products and Existing Products.
C = np.arange(0, cols)
EP = np.arange(0, (rows*30)//100)
CP = np.arange((rows*30)//100, rows)

k = 5

# IG
timeTaken = datetime.datetime.now()
selectedProdsIG, productScore = IG(k, C, SBS, EP, CP)
timeTaken = datetime.datetime.now() - timeTaken
print("Incremental Based Greedy Algorithm : \n", selectedProdsIG)
print("Time taken:", timeTaken.seconds)
print("Product score:", productScore)

# SPGA
timeTaken = datetime.datetime.now()
selectedProdsSPG, productScore = SPG(k, C, SBS, EP, CP)
timeTaken = datetime.datetime.now() - timeTaken 
print("Single Product Based Greedy Algorithm : \n", selectedProdsSPG)
print("Time taken:", timeTaken.seconds)
print("Product score:", productScore)

sampledEP, sampledCP = sampling(EP, CP)

print("Sampled EP, CP length:", len(sampledEP), len(sampledCP))
bestSampledProds, productScore = SPG(k*2, C, SBS, sampledEP, sampledCP)
print("Best sampled products:", bestSampledProds)

SBS_New, EP_New, CP_New = KMeans.K_Means(data, SBS, C, EP, CP, bestSampledProds, n_clusters = 10)

print('Length of selected cluster:', len(SBS_New))

print('After KMeans clustering:')
# IG after KMeans
timeTaken = datetime.datetime.now()
selectedProdsIG, productScore = IG(k, C, SBS_New, EP_New, CP_New)
timeTaken = datetime.datetime.now() - timeTaken
print("Incremental Based Greedy Algorithm : \n", selectedProdsIG)
print("Time taken:", timeTaken.seconds)
print("Product score:", productScore)

# SPGA after KMeans
timeTaken = datetime.datetime.now()
selectedProdsSPG, productScore = SPG(k, C, SBS_New, EP_New, CP_New)
timeTaken = datetime.datetime.now() - timeTaken 
print("Single Product Based Greedy Algorithm : \n", selectedProdsSPG)
print("Time taken:", timeTaken.seconds)
print("Product score:", productScore)
