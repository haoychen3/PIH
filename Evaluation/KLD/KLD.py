import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
import pathlib

path0 = pathlib.Path('./0-G/')
files0 = list(path0.glob('*/*'))

path1 = pathlib.Path('./1-G/')
files1 = list(path1.glob('*/*'))

dist_0 = []

for i in range(len(files0)):

    model = torch.load(str(files0[i]), map_location='cpu')
    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    params = torch.cat(params)

    x = params.detach().numpy()

    weights = np.ones_like(x) / len(x)

    dist, bins, patches = plt.hist(x, range=(-1, 1), bins=50, weights=weights)
    plt.xlabel('Value of parameter')
    plt.ylabel('Numbers (normalized)')
    dist_0.append(dist)

dist_0 = np.asarray(dist_0)

plt.savefig('total_dist_0.pdf', bbox_inches='tight')



dist_1=[]

for i in range(len(files1)):

    model = torch.load(str(files1[i]),map_location ='cpu')

    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    params = torch.cat(params)

    x = params.detach().numpy()

    weights = np.ones_like(x) / len(x)

    dist, bins, patches = plt.hist(x, range=(-1,1), bins=50, weights=weights)
    plt.xlabel('Value of parameter')
    plt.ylabel('Numbers (normalized)')
    dist_1.append(dist)


dist_1 = np.asarray(dist_1)

plt.savefig('total_dist_1.pdf', bbox_inches='tight')

kld_1 = []
for i in range(len(dist_0)):
    div = sum(scipy.special.kl_div(dist_0[i] + 1e-9, dist_1[i] + 1e-9))
    #     print(div)
    kld_1.append(div)

kld_1 = np.asarray(kld_1)

print(np.mean(kld_1))