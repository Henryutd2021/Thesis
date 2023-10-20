from gurobipy import *
import pandas as pd
from collections import defaultdict


cf = 44
co = 2
cm = 42
df = pd.read_csv('input_data_dh_y.csv')
d = defaultdict(list)

for i in range(len(df)):
    for r in ['pel', 'thl', 'pres']:
        d[i].append(df[r].tolist()[i])

time, pel, pthl, pres = multidict(d)


m = Model("vpp")

pmda = m.addVars(time, lb=0.0, vtype=GRB.CONTINUOUS, name='pmda')
pechp = m.addVars(time, lb=0, ub=500, vtype=GRB.CONTINUOUS, name='pechp')
pes_ch = m.addVars(time, lb=0.0, vtype=GRB.CONTINUOUS, name='pes_ch')
pes_dis = m.addVars(time, lb=0.0, vtype=GRB.CONTINUOUS, name='pes_dis')
ths_ch = m.addVars(time, lb=0.0, vtype=GRB.CONTINUOUS, name='ths_ch')
ths_dis = m.addVars(time, lb=0.0, vtype=GRB.CONTINUOUS, name='ths_dis')
se = m.addVars(time, lb=0, ub=100, vtype=GRB.CONTINUOUS, name='se')
sth = m.addVars(time, lb=0, ub=135, vtype=GRB.CONTINUOUS, name='sth')


m.setObjective(quicksum(cf*pechp[j] + co*pes_ch[j] + co*pes_dis[j] - pmda[j]*cm for j in time[1:]), GRB.MINIMIZE)

m.addConstrs((pechp[j]+pes_dis[j]+pres[j]-pmda[j]-pes_ch[j] >= pel[j] for j in time[1:]), 'e')
m.addConstrs((0.8*pechp[j]+ths_dis[j]-ths_ch[j] >= pthl[j] for j in time[1:]), 'th')
m.addConstr((se[0] == 0), 'e_0')
m.addConstr((sth[0] == 0), 'th_0')
m.addConstrs((sth[j] == sth[j-1]+ths_ch[j]-ths_dis[j] for j in time[1:]), 'th_s')
m.addConstrs((se[j] == se[j-1]+pes_ch[j]-pes_dis[j] for j in time[1:]), 'e_s')


m.optimize()
od = [pmda[s].X for s in range(len(df))]
df.insert(loc=4, column='pmda', value=od)
df.to_csv('input_data_dh_y.csv')
# for var in m.getVars():
#     print(f'{var.varName}: {round(var.X, 3)}')

#m.write('model.lp')







