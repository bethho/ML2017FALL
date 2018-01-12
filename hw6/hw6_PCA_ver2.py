# coding: utf-8
#!/usr/bin/python
# -*- coding: utf-8 -*-
from os import listdir
from os.path import isfile, isdir, join
from skimage import io
import numpy as np
import time
import argparse

t1 = time.time()
parser = argparse.ArgumentParser(description='PCA')
parser.add_argument('--img_dir',type=str)
parser.add_argument('--img_file', type=str)
parser.add_argument('--result', default = 'reconstruction.jpg', type=str)
args = parser.parse_args()
# make a list of file
# 指定要列出所有檔案的目錄
mypath = args.img_dir

# 取得所有檔案與子目錄名稱
files = listdir(mypath)
M = 600*600*3
N = len(files)
X = np.zeros([M,N],dtype=float)
# 以迴圈處理
for i in range(N):
    # 產生檔案的絕對路徑
    fullpath = join(mypath, files[i])
    # 判斷 fullpath 是檔案還是目錄
    if isfile(fullpath):
        img = io.imread(fullpath)
        img = img.astype('float32') / 255
        X[:,i] = img.flatten()

# calculate PCA
X_mean = np.zeros([M,1], dtype=float)
#X_mean[:,0] = np.sum(X, axis=1)/N
X_mean[:,0] = np.mean(X, axis=1)
U, s, V = np.linalg.svd(X-X_mean, full_matrices=False)

# recontruct face
file = args.img_file
id = files.index(file)
k = 4 # 10, 20 don't see the same face
y = np.array(X[:,id]).reshape(1,-1) - X_mean.T
W = np.dot(y, U[:,:k])
y_re = np.zeros([M,1], dtype=float)
for i in range(k):
    y_re = y_re + W[0,i]*U[:,i].reshape(-1,1)
y_re = y_re + X_mean
y_re -= np.min(y_re)
y_re /= np.max(y_re)
y_re = (y_re * 255).astype(np.uint8)
print(y_re.shape)
y_re = y_re.reshape(600, 600, 3)

io.imsave(args.result, y_re)

t2 = time.time()

print('diff_time')
print(t2-t1)
