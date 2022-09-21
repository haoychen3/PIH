import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import pathlib
from statistics import mean
from DISTS_pytorch import DISTS
import torch

path1 = pathlib.Path('./real/')
files1 = list(path1.glob('*'))

path2 = pathlib.Path('./extracted/')
files2 = list(path2.glob('*'))

X=[]
Y=[]
for i in range(len(files1)):
    original = cv2.imread(str(files1[i]))
    original = np.moveaxis(original, -1, 0)
    contrast = cv2.imread(str(files2[i]))
    contrast = np.moveaxis(contrast, -1, 0)
    X.append(original)
    Y.append(contrast)

X = np.array(X)
Y = np.array(Y)
X = torch.tensor(X, dtype=torch.float)
Y = torch.tensor(Y, dtype=torch.float)

D = DISTS()
# calculate DISTS between X, Y (a batch of RGB images, data range: 0~1)
# X: (N,C,H,W)
# Y: (N,C,H,W)
dists_value = D(X, Y)
# set 'require_grad=True, batch_average=True' to get a scalar value as loss.
dists_loss = D(X, Y, require_grad=True, batch_average=True)
dists_loss.backward()

print(dists_value.mean())