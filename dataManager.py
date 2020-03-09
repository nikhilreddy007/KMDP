import pandas as pd
import numpy as np
import random

""" 
Our Data format:
    buyingCost   v-high, high, med, low
    maintCost    v-high, high, med, low
    doors        2, 3, 4, 5-more
    seats        2, 4, more
    lug_boot     small, med, big
    safety       low, med, high
"""

# Create data file and creates a npArray "data" (uint8 format).
def createData(fileLocation = "./dataset/car/car.data"):
    print("Reading dataset...")
    data = pd.read_csv(fileLocation, header=None)
    data = np.array(data)
    print("Formatting data...")
    data[data=='5more'] = 5
    data[data=='vhigh'] = 3
    data[data=='more'] = 5
    data[data=='high'] = 2
    data[data=='big'] = 2
    data[data=='med'] = 1
    data[data=='low'] = 0
    data[data=='small'] = 0
    # Classes defining if a user BUYs or NOT.
    data[data=='unacc'] = 0
    data[data=='acc'] = 1
    data[data=='good'] = 2
    data[data=='vgood'] = 3
    data = np.array(data, dtype=np.uint8)
    np.random.shuffle(data)

    fileName = fileLocation.split("/")[-1].split(".")[0]
    pd.DataFrame(data, dtype=np.uint8).to_csv("./data_process/"+fileName+".csv", header=None, index=None)
    return data

# Creates and saves a "Satisfaction Bit String table" from the "data" for 500 customers.
def createSBS(data, fileLocation="./dataset/car/car.data"):
    no_cust = 500
    print("creating Empty SBS table...")
    SBS = np.zeros((data.shape[0], no_cust), dtype=np.uint8)

    print("Filling SBS table...")
    for i in range(SBS.shape[0]):
        if data[i][-1] == 3: 	# 90% chance
            for j in range(SBS.shape[1]):
                SBS[i][j]= 0 if random.randint(0,100)%10==0 else 1
        elif data[i][-1] == 2:  # 75% chance
            for j in range(SBS.shape[1]):
                SBS[i][j]= 0 if random.randint(0,100)%4==0 else 1
        elif data[i][-1] == 1:  # 50% chance
            for j in range(SBS.shape[1]):
                SBS[i][j]= 0 if random.randint(0,100)%2==0 else 1
        elif data[i][-1] == 0:  # 10% chance
            for j in range(SBS.shape[1]):
                SBS[i][j]= 1 if random.randint(0,100)%10==0 else 0

    print("Saving to file...")
    fileName = fileLocation.split("/")[-1].split(".")[0]
    pd.DataFrame(SBS, dtype=np.uint8).to_csv("./data_process/"+fileName+"_SBS.csv", header=None, index=None)
    return SBS

# Reads data file and creates a npArray "data" (uint8 format).
def readData(fileLocation='./data_process/car.csv'):
    print("Reading data...")
    data = pd.read_csv(fileLocation, header=None)
    data = np.array(data, dtype=np.uint8)
    return data

# Reads the saved Satisfaction Bit String from saved file.
def readSBS(SBSLocation='./data_process/car_SBS.csv'):
    print("Reading SBS...")
    SBS = pd.read_csv(SBSLocation, header=None)
    SBS = np.array(SBS, dtype=np.uint8)
    return SBS
