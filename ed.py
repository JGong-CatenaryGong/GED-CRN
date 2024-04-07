import numpy as np
import math

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from wfn_reader import read_wfn
from wfn_calc import *

from model import *
from global_constant import BATCH_SIZE, BOX_SIZE

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


model = torch.load("./saved_models/512conv/withres/CNN_grad_adam.pth").to(device)

bs = BATCH_SIZE

class Ds(Dataset):
    def __init__(self, pro, pot, anal):
        self.box_size = BOX_SIZE
        self.pro = pro.reshape(pro.shape[0], 1, self.box_size, self.box_size, self.box_size)
        self.pot = pot.reshape(pot.shape[0], 1, self.box_size, self.box_size, self.box_size)
        self.anal = anal.reshape(anal.shape[0], 1, self.box_size, self.box_size, self.box_size)

    def __getitem__(self, index):
        return self.pro[index], self.pot[index], self.anal[index]
    
    def __len__(self):
        return self.anal.shape[0]
    
    def item_shape(self):
        return self.box_size

train_data = np.load("train.npy.npz")
train_x = train_data['pro']
train_pot = np.log10(np.array(train_data['pot']))
train_y = train_data['y']
train_ds = Ds(train_x, train_pot, train_y)

val_data = np.load("val.npy.npz")
val_x = val_data['pro']
val_pot = np.log10(np.array(val_data['pot']))
val_y = val_data['y']
val_ds = Ds(val_x, val_pot, val_y)

train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=bs, shuffle=True, drop_last=True)

criterion = nn.L1Loss().to(device)

eloss = []
for i, data in enumerate(train_loader):
    model.zero_grad()
    model.eval()

    x, pot, y = [d.to(device) for d in data]
    y = y.to(torch.float32)

    preds = model(x, pot)
    err = preds.view(-1) - y.view(-1)
    eloss.append(err.cpu())

for i, data in enumerate(val_loader):
    model.zero_grad()
    model.eval()

    x, pot, y = [d.to(device) for d in data]
    y = y.to(torch.float32)

    preds = model(x, pot)
    err = preds.view(-1) - y.view(-1)
    eloss.append(err.cpu())

eloss = torch.concatenate(eloss)

np.savetxt("eloss2.csv", eloss.detach().numpy())

