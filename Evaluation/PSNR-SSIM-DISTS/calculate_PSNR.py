from math import log10, sqrt
import cv2
import numpy as np
import pathlib
from statistics import mean


def PSNR(original, compressed):
    original = original.astype(np.float64)
    compressed = compressed.astype(np.float64)
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

path1 = pathlib.Path('./real/')
files1 = list(path1.glob('*'))

path2 = pathlib.Path('./extracted/')
files2 = list(path2.glob('*'))

PSNR_list=[]

for i in range(len(files1)):
    original = cv2.imread(str(files1[i]), cv2.IMREAD_COLOR)
    contrast = cv2.imread(str(files2[i]), cv2.IMREAD_COLOR)
    value = PSNR(original, contrast)
    print(f"PSNR value is {value} dB")
    PSNR_list.append(value)

average = mean(PSNR_list)
print(f'average PSNR is {average} dB')