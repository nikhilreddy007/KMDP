import numpy as np


def sampling(EP, CP):
    sampled_CP = np.random.choice(CP, (len(CP)*10)//100, replace=False)
    sampled_EP = np.random.choice(EP, (len(EP)*10)//100, replace=False)
    return sampled_EP, sampled_CP
