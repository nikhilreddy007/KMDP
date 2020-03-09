def N_Calc(SBS, EP, CP):
  cols = len(SBS[0])
  Nsbs = [[0]*2]*cols
  for i in range(cols):
    EP_count=0
    CP_count=0
    for j in EP:
      if SBS[j][i]==1:
        EP_count+=1
    for j in CP:
      if SBS[j][i]==1:
        CP_count+=1
    Nsbs[i][0] = EP_count
    Nsbs[i][1] = CP_count
  return Nsbs

def N(cust, Nsbs):
  EP_count = Nsbs[cust][0]
  CP_count = Nsbs[cust][1]
  return EP_count+CP_count

def P(cp_i, c_j, Nsbs, SBS):
  if SBS[cp_i][c_j]==1:
    p = 1 / N(c_j, Nsbs)
  else:
    p = 0
  return p

def E(kCP_n, C, Nsbs, SBS):
  E = 0
  for cp_i in kCP_n:
    for c_j in C:
      E += P(cp_i, c_j, Nsbs, SBS)
  return E
