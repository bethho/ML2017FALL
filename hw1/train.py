# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 11:53:01 2017

@author: Beth Ho
"""
import csv 
import numpy as np
from numpy.linalg import inv
import random
import math
import sys
import time

data = []
# 每一個維度儲存一種污染物的資訊
for i in range(18):
    data.append([])

n_row = 0
text = open('data/train.csv', 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for r in row:
    # 第0列沒有資訊
    if n_row != 0:
        # 每一列只有第3-27格有值(1天內24小時的數值)
        for i in range(3,27):
            if r[i] != "NR":
                data[(n_row-1)%18].append(float(r[i]))
            else:
                data[(n_row-1)%18].append(float(0))	
    n_row = n_row+1
text.close()

# 選哪些汙染物
A = [8,9]
#A = range(18)
# 選擇連續幾小時
T = 9
# 每個月有幾筆資料
MT = 480-T;
#使用到幾次方
n = 2;

x = []
y = []
# 每 12 個月
for i in range(12):
    # 一個月取連續10小時的data可以有471筆
    for j in range(MT):
        x.append([])
        # 18種污染物
        for t in A:
            # 連續9小時
            for s in range(T):
                x[MT*i+j].append(data[t][480*i+j+s] )
        y.append(data[9][480*i+j+T])
x = np.array(x)
y = np.array(y)

# add square term
for i in range(n-1):
    q = i+2
    x = np.concatenate((x,x**q), axis=1)

# add bias
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)

w = np.zeros(len(x[0]))
l_rate = 10
repeat = 100000

# use close form to check whether ur gradient descent is good
# however, this cannot be used in hw1.sh 
w1 = np.matmul(np.matmul(inv(np.matmul(x.transpose(),x)),x.transpose()),y)

x_t = x.transpose()
s_gra = np.zeros(len(x[0]))

for i in range(repeat):
    hypo = np.dot(x,w)
    loss = hypo - y
    loss_t = np.sum(loss)
    cost = np.sum(loss**2) / len(x)
    cost_a  = math.sqrt(cost)
    gra = np.dot(x_t,loss)
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    w = w - l_rate * gra/ada
    print ('iteration: %d | Cost: %f | Loss: %f' % ( i,cost_a,loss_t))