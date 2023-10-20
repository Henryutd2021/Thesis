from amplpy import AMPL, DataFrame
import pandas as pd
import time


ampl = AMPL()
ampl.reset()
ampl.setOption('solver', 'GUROBI')
ampl.setOption('solution_round', '3')

ampl.read('fmla_m3M.mod')
ampl.readData('fmla_sca8.dat')
start = time.perf_counter()
# Solve
ampl.solve()
end = time.perf_counter()
print(f'Running time: {end - start} Seconds')

writer = pd.ExcelWriter('fmla_sca8(buy 140).xlsx')
x = ampl.getVariables()

for i in x:
    #print(i)
    D = ampl.getVariable(str(i[0])).getValues().toPandas()
    D.to_excel(writer, sheet_name=str(i[0]))
writer.save()

# totalcost = ampl.getObjective('total_cost')
# print("Objective is:", totalcost.get().value())

