import numpy as np
from collections import defaultdict
from gurobipy import *



cf = 44
co = 2
cp = 45
cm = 37
pechpmax = 135
pechpmin = 0
pthchpsmax = 135
pthchpsmin = 0
pesmax = 100
pesmin = 0
z = 0
w = 0



pl = [197, 161, 188, 197, 183, 192, 205, 262, 224, 270, 271, 275, 240, 185, 199, 278, 264, 275, 279, 295, 289, 270, 231,156]
thl = [79, 62, 83, 79, 75, 70, 75, 62, 49, 37, 38, 44, 42, 32, 33, 55, 59, 65, 99, 86, 75, 75, 78, 72]
pres = [183, 163, 142, 176, 177, 199, 172, 208, 235, 201, 219, 265, 260, 265, 255, 240, 180, 150, 151, 161, 143, 146, 151, 168]
pmda = [142.4, 44.15, 0, 42.45, 0, 153.7, 0, 58.4, 0, 0, 168.9, 0, 54.8, 35, 0, 0, 0, 0, 0, 3.1, 87, 0, 0, 0]


n = 0
while n < 22:
    d = defaultdict(list)
    for i in np.linspace(0, 3, 4):
        if i == 0:
            d[i].append(0)
            d[i].append(0)
            d[i].append(0)
            d[i].append(0)
            continue
        else:
            d[i].append(pl[n])
            d[i].append(thl[n])
            d[i].append(pres[n])
            d[i].append(pmda[n])
            n += 1

    n -= 2
    d = dict(d)
    time, pl, thl, pres, pmda = multidict(d)

    m = Model("vpp")

    pechp = m.addVars(time, lb=0.0, ub=pechpmax, vtype=GRB.CONTINUOUS, name='pechp')
    pimb = m.addVars(time, lb=0.0, vtype=GRB.CONTINUOUS, name='pimb')
    pes_ch = m.addVars(time, lb=0.0, vtype=GRB.CONTINUOUS, name='pes_ch')
    pes_dis = m.addVars(time, lb=0.0, vtype=GRB.CONTINUOUS, name='pes_dis')
    se = m.addVars(time, lb=0.0, ub=pesmax, vtype=GRB.CONTINUOUS, name='se')
    sth = m.addVars(time, lb=0.0, ub=pthchpsmax, vtype=GRB.CONTINUOUS, name='sth')
    ths_ch = m.addVars(time, lb=0.0,  vtype=GRB.CONTINUOUS, name='ths_ch')
    ths_dis = m.addVars(time, lb=0.0, vtype=GRB.CONTINUOUS, name='ths_dis')



    m.setObjective(quicksum(cf * pechp[j] + co * pes_ch[j] + pes_dis[j] * co + pimb[j] * cp - pmda[j] * cm for j in time), GRB.MINIMIZE)

    m.addConstrs((pechp[j] + pes_dis[j] + pres[j] - pmda[j] - pes_ch[j] + pimb[j] >= pl[j] for j in time), 'e')
    m.addConstrs((0.8 * pechp[j] + ths_dis[j]-ths_ch[j] >= thl[j] for j in time), 'th')

    m.addConstr((se[0] == z), 'e_0')
    m.addConstr((sth[0] == w), 'th_0')
    m.addConstrs((sth[j] == sth[j-1]+ths_ch[j]-ths_dis[j] for j in time[1:]), 'th_s')
    m.addConstrs((se[j] == se[j-1]+pes_ch[j]-pes_dis[j] for j in time[1:]), 'e_s')

    m.optimize()
    print('='*50)
    # for var in m.getVars():
    #     #print(var.varName)
    #     if var.varName is se:
    #         #print(f"{se}: {round(se.X, 3)}")
    #         print(f"{var.varName}: {round(var.X, 3)}")
    # for v in se.values():
    #     if v.varName is se[1.0]:
    #         print(v)
        #print("{}: {}".format(v[1].varName, v[1].X))
    z = se[1.0].X
    w = sth[1.0].X

    pl = [197, 161, 188, 197, 183, 192, 205, 262, 224, 270, 271, 275, 240, 185, 199, 278, 264, 275, 279, 295, 289, 270,
          231, 156]
    thl = [79, 62, 83, 79, 75, 70, 75, 62, 49, 37, 38, 44, 42, 32, 33, 55, 59, 65, 99, 86, 75, 75, 78, 72]
    pres = [183, 163, 142, 176, 177, 199, 172, 208, 235, 201, 219, 265, 260, 265, 255, 240, 180, 150, 151, 161, 143,
            146, 151, 168]
    pmda = [142.4, 44.15, 0, 42.45, 0, 153.7, 0, 58.4, 0, 0, 168.9, 0, 54.8, 35, 0, 0, 0, 0, 0, 3.1, 87, 0, 0, 0]


