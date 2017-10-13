# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 10:52:32 2017

@author: Beth Ho
"""

import csv 
import numpy as np
from numpy.linalg import inv
import random
import math
import sys
import time

t1 = time.time()

A = [8,9]
#A = range(18)
# 選擇連續幾小時
T = 9
# 每個月有幾筆資料
MT = 480-T;
#使用到幾次方
n = 2;
# read model
w = np.load('model_w.npy')

test_x = []
n_row = 0
text = open(sys.argv[1] ,"r")
row = csv.reader(text , delimiter= ",")
s = len(A)
for r in row:
    if n_row %18 == 0:
        test_x.append([])        
            
    if (n_row %18) in A:
        for i in range(11-T,11):
            if r[i] !="NR":
                test_x[n_row//18].append(float(r[i]))
            else:
                test_x[n_row//18].append(0)
    n_row = n_row+1
text.close()
test_x = np.array(test_x)

# add square term
for i in range(n-1):
    q = i+2
    test_x = np.concatenate((test_x,test_x**q), axis=1)
# test_x = np.concatenate((test_x,test_x**2), axis=1)

# add bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)

ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])
    ans[i].append(a)

filename = sys.argv[2]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()

t2 = time.time()
Tt = t2-t1
print(Tt/60)