import csv 
import numpy as np
from plot import graph

ddqnr1 = []
varTitle = "DQN T3: Var100 vs Episode"
medTitle = "DQN T3: Med100 vs Episode"
rewardTitle = "DQN T3: Rewards vs Episode"
fileName = './data/dqn/dqnr3.csv'
algo = "DQN"
with open(fileName,'r') as csvfile: 
    reader = csv.reader(csvfile, delimiter=',', quotechar='|') 
    for row in reader:
        ddqnr1.append(np.array(row).astype(np.float))
y = [np.sum(row) for row in ddqnr1]

x = list(range(len(y)))
graph(rewardTitle, [(algo, y)])

y2 = [ np.var( y[i-100:i] ) for i in range(100, len(ddqnr1)) ]
x2 = list(range(100, len(y2)+100))
graph(varTitle, [(algo, y2)])

y3 = [ np.median( y[i-100:i] ) for i in range(100, len(ddqnr1)) ]
x3 = list(range(100, len(y3)+100))
graph(medTitle, [(algo, y3)])




# dqnr1 = []
# with open('./data/dqn/dqnr1.csv','r') as csvfile: 
#     reader = csv.reader(csvfile, delimiter=',', quotechar='|') 
#     for row in reader:
#         dqnr1.append(np.array(row).astype(np.float))
# y = [np.sum(row) for row in dqnr1]

# x = list(range(len(y)))
# graph("DQN T1: Rewards vs Episode", [("DQN", y)])

# y2 = [ np.var( y[i-100:i] ) for i in range(100, len(dqnr1)) ]
# x2 = list(range(100, len(y2)+100))
# graph("DQN T1: Var100 vs Episode", [("DQN", y2)])

# y3 = [ np.median( y[i-100:i] ) for i in range(100, len(dqnr1)) ]
# x3 = list(range(100, len(y3)+100))
# graph("DQN T1: Med100 vs Episode", [("DQN", y3)])