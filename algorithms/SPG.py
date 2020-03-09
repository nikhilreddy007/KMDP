import heapq

from algorithms.functions import N_Calc, N, P, E


def SPG(k, C, SBS, EP, CP):
  k = [(0, 0)]*k
  heapq.heapify(k)
  Nsbs = N_Calc(SBS, EP, CP)
  for cp in CP:
    S = set()
    S.add(cp)
    e = E(S, C, Nsbs, SBS)
    heapq.heappush(k, (e, cp))
    heapq.heappop(k)
  return [pair[1] for pair in k], sum([pair[0] for pair in k])
