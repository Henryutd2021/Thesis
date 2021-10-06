import numpy as np
from collections import defaultdict
from gurobipy import *
import pandas as pd


od = []
cf = 44
co = 2
z = 0
w = 0
n = 0
PH = 3
CH = 1

df = pd.read_csv('input_data_24.csv')

while n < len(df)-(PH-1):
    d = defaultdict(list)
    for i in range(0, PH+1):
        if i == 0:
            for k in range(6):
                d[i].append(0)
                continue
        else:
            for r in ['pel', 'thl', 'pres', 'pmda', 'cm', 'cp']:
                d[i].append(df[r].tolist()[n])
            n += 1
    n -= (PH - CH)
    time, pl, thl, pres, pmda, cm, cp = multidict(d)

    m = Model("vpp")

    pechp = m.addVars(time, lb=0.0, ub=135, vtype=GRB.CONTINUOUS, name='pechp')
    pimb = m.addVars(time, lb=0.0, vtype=GRB.CONTINUOUS, name='pimb')
    pes_ch = m.addVars(time, lb=0.0, vtype=GRB.CONTINUOUS, name='pes_ch')
    pes_dis = m.addVars(time, lb=0.0, vtype=GRB.CONTINUOUS, name='pes_dis')
    se = m.addVars(time, lb=0.0, ub=100, vtype=GRB.CONTINUOUS, name='se')
    sth = m.addVars(time, lb=0.0, ub=135, vtype=GRB.CONTINUOUS, name='sth')
    ths_ch = m.addVars(time, lb=0.0,  vtype=GRB.CONTINUOUS, name='ths_ch')
    ths_dis = m.addVars(time, lb=0.0, vtype=GRB.CONTINUOUS, name='ths_dis')


    m.setObjective(quicksum(cf * pechp[j] + co * pes_ch[j] + pes_dis[j] * co + pimb[j] * cp[j] - pmda[j] * cm[j] for j in time[1:]), GRB.MINIMIZE)
    m.addConstrs((pechp[j] + pes_dis[j] + pres[j] - pmda[j] - pes_ch[j] + pimb[j] >= pl[j] for j in time[1:]), 'e')
    m.addConstrs((0.8 * pechp[j] + ths_dis[j]-ths_ch[j] >= thl[j] for j in time[1:]), 'th')
    m.addConstr((se[0] == z), 'e_0')
    m.addConstr((sth[0] == w), 'th_0')
    m.addConstrs((sth[j] == sth[j-1]+ths_ch[j]-ths_dis[j] for j in time[1:]), 'th_s')
    m.addConstrs((se[j] == se[j-1]+pes_ch[j]-pes_dis[j] for j in time[1:]), 'e_s')

    m.optimize()
    # for var in m.getVars():
    #     print(f'{var.varName}: {round(var.X, 3)}')
    print('='*50)

    z = se[1.0].X
    w = sth[1.0].X
    cost1 = cf * pechp[1.0].X + co * pes_ch[1.0].X + pes_dis[1.0].X * co + pimb[1.0].X * cp[1.0] - pmda[1.0] * cm[1.0]
    od.append([pechp[1.0].X, pes_ch[1.0].X, pes_dis[1.0].X, se[1.0].X, ths_ch[1.0].X, ths_dis[1.0].X, sth[1.0].X, pimb[1.0].X, cost1])

    # cost2 = cf * pechp[2.0].X + co * pes_ch[2.0].X + pes_dis[2.0].X * co + pimb[2.0].X * cp[2.0] - pmda[2.0] * cm[2.0]
    # od.append([pechp[2.0].X, pes_ch[2.0].X, pes_dis[2.0].X, se[2.0].X, ths_ch[2.0].X, ths_dis[2.0].X, sth[2.0].X, pimb[2.0].X,
    #      cost2])
    #
    # cost3 = cf * pechp[3.0].X + co * pes_ch[3.0].X + pes_dis[3.0].X * co + pimb[3.0].X * cp[3.0] - pmda[3.0] * cm[3.0]
    # od.append(
    #     [pechp[3.0].X, pes_ch[3.0].X, pes_dis[3.0].X, se[3.0].X, ths_ch[3.0].X, ths_dis[3.0].X, sth[3.0].X, pimb[3.0].X,
    #      cost3])
    #
    # cost4 = cf * pechp[4.0].X + co * pes_ch[4.0].X + pes_dis[4.0].X * co + pimb[4.0].X * cp[4.0] - pmda[4.0] * cm[4.0]
    # od.append(
    #     [pechp[4.0].X, pes_ch[4.0].X, pes_dis[4.0].X, se[4.0].X, ths_ch[4.0].X, ths_dis[4.0].X, sth[4.0].X, pimb[4.0].X,
    #      cost4])

df_od = pd.DataFrame(od,columns=['pechp', 'pes_ch', 'pes_dis', 'se', 'ths_ch', 'ths_dis', 'sth', 'pimb', 'cost'])
df_od.to_csv('output_data_24_3_1.csv', sep=',', index=False, header=True)