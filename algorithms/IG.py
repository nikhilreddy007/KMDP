import numpy as np
from algorithms.functions import N, N_Calc, P, E

def IG(k, C, SBS, EP, CP):
  S = np.array((),dtype=np.uint8)
  Nsbs = N_Calc(SBS, EP, CP)
  while(len(S) < k):
    max_E=0
    max_CP=0
    maxIndex = 0
    for cp in range(len(CP)):
      temp_S = np.append(S, CP[cp])
      e = E(temp_S, C, Nsbs, SBS)
      if (e > max_E):
        max_E = e
        max_CP = CP[cp]
        maxIndex = cp
    S = np.append(S, max_CP)
    # index = np.argwhere(CP==max_CP)
    CP = np.delete(CP, maxIndex)
  return S, max_E
