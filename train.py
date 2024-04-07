import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py
from model import GED_CNN2
from global_constant import *

import visdom
viz = visdom.Visdom(env = "ged_cnn")

viz.line(
    X = [0.],
    Y = [0.],
    win = "line1",
    update = 'append',
    opts = {
        'showlegend': True,
        'xlabel': "epoch",
        'ylabel': "value",
    }
)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

NUM_EPOCHS = 1000
bs = BATCH_SIZE

print(f"The model is running on {device}.")

class Ds(Dataset):
    def __init__(self, pro, pot, anal):
        self.pro = torch.from_numpy(pro)
        self.pot = torch.from_numpy(pot)
        self.anal = torch.from_numpy(anal)

    def __getitem__(self, index):
        return self.pro[index], self.pot[index], self.anal[index]
    
    def __len__(self):
        return len(self.anal)
    
    def item_shape(self):
        return self.anal[0].shape

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

# train_ds = Ds(train_y, train_x)
# vali_ds = Ds(vali_y, vali_x)

# train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
# vali_loader = DataLoader(vali_ds, batch_size=bs, shuffle=False)
    


model = GED_CNN2(val_ds.item_shape(), bs).to(device)

# model = torch.load("./saved_models/fixed/ged_cnn.pth").to(device)

criterion = nn.L1Loss().to(device)
optim = optim.Adam(model.parameters(), lr = 0.0001)

mae = nn.L1Loss().to(device)
    
loss = []
val_loss = []

allmae = []
allvalmae = []

for epoch in range(NUM_EPOCHS):
    eloss = []
    emae = []
    for i, data in enumerate(train_loader):
        model.zero_grad()

        x, pot, y = [d.to(device) for d in data]
        y = y.to(torch.float32)

        preds = model(x, pot)

        err = criterion(preds, y)
        err.backward()

        tmae = mae(preds, y)

        eloss.append(err.cpu().item())
        emae.append(tmae.cpu().item())

        if i % 100 == 0:
            print(f"EPOCH {epoch}/{NUM_EPOCHS} BATCH {i} Loss: {np.mean(np.array(eloss))} MAE: {tmae.cpu().item()}")

        optim.step()

    loss.append(np.mean(np.array(eloss)))

    allmae.append(np.mean(np.array(emae)))


    vloss = []
    valmae = []
    for i, data in enumerate(val_loader):
        model.zero_grad()
        model.eval()

        x, pot, y = [d.to(device) for d in data]

        preds = model(x, pot)

        err = criterion(preds, y)

        vmae = mae(preds, y)

        vloss.append(err.cpu().item())
        valmae.append(vmae.cpu().item())

    val_loss.append(np.mean(np.array(vloss)))

    allvalmae.append(np.mean(np.array(valmae)))

    print(f"EPOCH {epoch}/{NUM_EPOCHS} tLoss: {np.mean(np.array(eloss))} invLoss: {np.mean(np.array(vloss))}")
    print(f"EPOCH {epoch}/{NUM_EPOCHS} tMAE: {np.mean(np.array(emae))} invMAE: {np.mean(np.array(valmae))}")

    viz.line(X = [epoch], Y = [loss[-1]], name = 'train_loss', win = 'line1', update = 'append')
    viz.line(X = [epoch], Y = [val_loss[-1]], name = 'val_loss', win = 'line1', update = 'append')
    viz.line(X = [epoch], Y = [allmae[-1]], name = 'train_mae', win = 'line1', update = 'append')
    viz.line(X = [epoch], Y = [allvalmae[-1]], name = 'val_mae', win = 'line1', update = 'append')
    
    model_traced = torch.jit.trace(model, (x, pot))
    torch.jit.save(model_traced, 'ged_cnn_jit.pth')
    np.savetxt("ged_loss.csv", np.array([np.array(loss), np.array(val_loss), np.array(allmae), np.array(allvalmae)]).T)


