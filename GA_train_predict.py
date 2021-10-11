
import GA_LSTM_Single_Layer as project
import os
import sys
import keras
import numpy as np # linear algebra
from scipy.stats import randint
import pandas as pd
from math import sqrt
from numpy import concatenate
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

#read input and output
dataindex=pd.read_excel('/Users/ycs/Desktop/Task/data/label-27minutes.xlsx')
Label=dataindex.loc[ (dataindex['TimeStamp (mS)']>495369) & (dataindex['TimeStamp (mS)']< 2114095) , ['TimeStamp (mS)','S','D','H','R']]
Feature=pd.read_csv('/Users/ycs/Desktop/Task/data/featuredata10_clear.csv',header=None)

# In[3]:


feature = Feature.values
label = Label.values
#data normalization
scaler1 = MinMaxScaler(feature_range=(0, 1))
scaled1 = scaler1.fit_transform(feature)
scaler2 = MinMaxScaler(feature_range=(0, 1))
scaled2 = scaler2.fit_transform(label)


# In[4]:


#train
x_train=scaled1[0:983,[1,2,3,4,5,6,7,8,9,10]]
y_train=scaled2[0:983,[1,2,3,4]]#four different labels

#validation
x_val=scaled1[984:1214,[1,2,3,4,5,6,7,8,9,10]]
y_val=scaled2[984:1214,[1,2,3,4]]#four different labels




#reshape for the BiLSTM
train_X = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
val_X = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))

print(train_X.shape, val_X.shape)


DNA_SIZE = 4
POP_SIZE = 50
CROSS_RATE = 0.5
MUTATION_RATE = 0.01
N_GENERATIONS = 20


def get_fitness(x):
    return project.classify(train_X, y_train, val_X , y_val, num=x)


def select(pop, fitness):
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=fitness / fitness.sum())
    return pop[idx]


def crossover(parent, pop):
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)
        parent[cross_points] = pop[i_, cross_points]
    return parent


def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            if point == 1:
                child[point] = np.random.randint(4, 24)
            elif point == 2:
                child[point] = np.random.randint(15, 50)
            elif point == 3:
                child[point] = np.random.randint(2, 6)
            else:
                child[point] = np.random.randint(2, 30)
    return child


pop = np.zeros((POP_SIZE, DNA_SIZE), np.int32)  # initialize the pop DNA
pop[:, 0] = np.random.randint(4, 24, size=(POP_SIZE,))
pop[:, 1] = np.random.randint(15, 50, size=(POP_SIZE,))
pop[:, 2] = np.random.randint(2, 6, size=(POP_SIZE,))
pop[:, 3] = np.random.randint(2, 30, size=(POP_SIZE,))

for each_generation in range(N_GENERATIONS):
    fitness = np.zeros([POP_SIZE, ])
    for i in range(POP_SIZE):
        pop_list = list(pop[i])
        fitness[i] = get_fitness(pop_list)
        print('the %d generation the %d chromosome fitness %f' % (each_generation + 1, i + 1, fitness[i]))
        print('the selected chromosome：', pop_list)
    print("Generation:", each_generation + 1, "Most fitted DNA: ", pop[np.argmax(fitness), :], "Fitness：",
          fitness[np.argmax(fitness)])
    pop = select(pop, fitness)
    pop_copy = pop.copy()
    for parent in pop:
        child = crossover(parent, pop_copy)
        child = mutate(child)
        parent = child


