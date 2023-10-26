import os
import numpy as np
import sys
import matplotlib.pyplot as plt

lambd = 0.1
mu = 0.1

length = 1000000
state_limit = 10

# First method

race = np.zeros(shape=[length,2])
race[:,0] = np.random.exponential(scale=(1/mu),size=length)
race[:,1] = np.random.exponential(scale=(1/lambd),size=length)

holding_times = np.min(race,axis=1)# Values
bd = np.argmin(race,axis=1)
bd[bd==0] = -1

# states = [0]

# for i in range(length):
#     cur_state = states[-1]
#     change = bd[i]
#     if cur_state == 0 and change == -1:# We only take birth 
#         holding_times[i] = race[i,1]
#         # holding_times[i] = np.random.exponential(scale=(1/lambd))
#         change = 1
#     if cur_state == state_limit and change==1:
#         holding_times[i] = race[i,0]
#         # holding_times[i] = np.random.exponential(scale=(1/mu))
#         change = -1
#     states.append(cur_state + change)

tempHist = np.histogram(holding_times,bins=20, range=[0,5])

plt.bar(tempHist[1][1::]-0.2, tempHist[0]/length, width=0.05, label='Race of Exponential')

# Second method

holding_times2 = np.random.exponential(scale=(1/(mu + lambd)), size=length)

# states = [0]
# birth_or_death = np.random.choice([-1,1], length, p=[mu/(lambd+mu), lambd/(lambd+mu)])

# for i in range(length):
#     if states[-1]==0 and birth_or_death[i] == -1: 
#         temp1 = np.random.exponential(scale=(1/lambd))
#         temp2 = np.random.exponential(scale=(1/mu))
#         while temp1 < temp2:
#             temp1 = np.random.exponential(scale=(1/lambd))
#             temp2 = np.random.exponential(scale=(1/mu))
#         new_time = temp1
#         holding_times2[i] = new_time
#         # holding_times2[i] = np.random.exponential(scale=(1/lambd))
#         birth_or_death[i] = 1
#     if states[-1]==state_limit and birth_or_death[i] == 1: 
#         temp1 = np.random.exponential(scale=(1/lambd))
#         temp2 = np.random.exponential(scale=(1/mu))
#         while temp2 < temp1:
#             temp1 = np.random.exponential(scale=(1/lambd))
#             temp2 = np.random.exponential(scale=(1/mu))
#         new_time = temp2
#         holding_times2[i] = new_time
#         # holding_times2[i] = np.random.exponential(scale=(1/mu))
#         birth_or_death[i] = -1

#     states.append(states[-1] + birth_or_death[i])

tempHist2 = np.histogram(holding_times2, bins=20, range=[0,5])

plt.bar(tempHist2[1][1::]-0.15, tempHist2[0]/length, width=0.05, label='Embedded Markov')

plt.legend(loc="upper right")
# Third method

strips = np.concatenate((np.cumsum(race[:,0]), np.cumsum(race[:,1])))
strips = np.sort(strips)
holding_times3 = strips[1::] - strips[0:-1]
holding_times3 = holding_times3[0:length]

tempHist3 = np.histogram(holding_times3, bins=20, range=[0,5])

plt.bar(tempHist3[1][1::]-0.1, tempHist3[0]/length, width=0.05, label='Method3')

plt.legend(loc="upper right")


print(np.mean(holding_times))
print(np.mean(holding_times2))
print(np.mean(holding_times3))
