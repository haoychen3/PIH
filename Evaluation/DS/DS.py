import pandas as pd
import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

data=[]
with open('image_id.csv', 'w', newline='') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerow(['image_id'])
    for filename in os.listdir("gen_samples"):
        data.append(filename)
        writer.writerow(data)
        data=[]
writeFile.close()

df = pd.read_csv('image_id.csv')


class LeafData(Dataset):

    def __init__(self,
                 data,
                 directory,
                 transform=None):
        self.data = data
        self.directory = directory
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # import
        path = os.path.join(self.directory, self.data.iloc[idx]['image_id'])
        image = cv2.imread(path, cv2.IMREAD_COLOR)

        # augmentations
        if self.transform is not None:
            image = self.transform(image=image)['image']

        return image

# dataset
image_dataset = LeafData(data      = df,
                         directory = 'gen_samples',
                         transform = None)

# data loader
image_loader = DataLoader(image_dataset,
                          batch_size  = 5,
                          shuffle     = False,
                          num_workers = 0)

# read training image
train_image = cv2.imread('./real/02.jpg', cv2.IMREAD_COLOR)
train_image = torch.from_numpy(train_image).to(torch.long)

# placeholders
psum    = torch.zeros_like(train_image)
psum_sq = torch.zeros_like(train_image)

# loop through 25 sampled images
for inputs in tqdm(image_loader):
    inputs = inputs.to(torch.long)
    # print(inputs.size())
    psum    += inputs.sum(axis        = [0])
    psum_sq += (inputs ** 2).sum(axis = [0])

# image_size
count = 244 * 164

mean = psum / len(df)
var = psum_sq / len(df) - (mean ** 2)

std = torch.sqrt(var.sum(axis=[0, 1]) / count)
# print('average pixel-wise std for each channel:  '  + str(std))

std = torch.mean(std)
# print('average std:  '  + str(std))

####### Calculate the Standard Deviation of Traning Image

psum    = torch.tensor([0.0,0.0,0.0])
psum_sq = torch.tensor([0.0,0.0,0.0])

psum    += train_image.sum(axis        = [0, 1])
psum_sq += (train_image ** 2).sum(axis = [0, 1])

# mean and std
train_mean = psum / count
train_var  = (psum_sq / count) - (train_mean ** 2)
train_std  = torch.sqrt(train_var)
# print('pixel-wise average std of training image for each channel:  '  + str(train_std))

train_std = torch.mean(train_std)
# print('average std of training image:  '  + str(train_std))

####### Normalize to get diversity score

div_score = std / train_std
print('diversity score:  '  + str(div_score))