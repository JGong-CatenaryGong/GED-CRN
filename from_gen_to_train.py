import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py
from model import *
from global_constant import *

# import logging
# FORMAT = '%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s'
# logging.basicConfig(format=FORMAT)
# logging.getLogger().setLevel(logging.INFO)

import visdom
viz = visdom.Visdom(env = "ged_cnn")
 
# import gen_dataset

exec(open("./gen_dataset.py").read())

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
    
def rot_cat(boxes, axises):
    x90 = torch.rot90(boxes, 1, axises)
    x180 = torch.rot90(boxes, 2, axises)
    return torch.cat((x90, x180), dim=0)
    
def gen_rot_boxes(pro, pot, anal):
    pro_rot_x = rot_cat(pro, [2, 3])
    pot_rot_x = rot_cat(pot, [2, 3])
    anal_rot_x = rot_cat(anal, [2, 3])

    pro_rot_y = rot_cat(pro, [1, 3])
    pot_rot_y = rot_cat(pot, [1, 3])
    anal_rot_y = rot_cat(anal, [1, 3])

    pro_rot_z = rot_cat(pro, [1, 2])
    pot_rot_z = rot_cat(pot, [1, 2])
    anal_rot_z = rot_cat(anal, [1, 2])

    return torch.cat((pro, pro_rot_x, pro_rot_y, pro_rot_z), dim=0), \
        torch.cat((pot, pot_rot_x, pot_rot_y, pot_rot_z), dim=0), \
        torch.cat((anal, anal_rot_x, anal_rot_y, anal_rot_z), dim=0)


train_data = np.load("train.npy.npz")
train_pro = torch.from_numpy(train_data['pro'])
train_pot = torch.from_numpy(np.log10(np.array(train_data['pot'])))
train_anal = torch.from_numpy(train_data['y'])

# train_pro_x90 = torch.rot90(train_pro, 1, [2, 3])
# train_pot_x90 = torch.rot90(train_pot, 1, [2, 3])
# train_anal_x90 = torch.rot90(train_anal, 1, [2, 3])

# train_pro_y90 = torch.rot90(train_pro, 1, [1, 3])
# train_pot_y90 = torch.rot90(train_pot, 1, [1, 3])
# train_anal_y90 = torch.rot90(train_anal, 1, [1, 3])

# train_pro_z90 = torch.rot90(train_pro, 1, [1, 2])
# train_pot_z90 = torch.rot90(train_pot, 1, [1, 2])
# train_anal_z90 = torch.rot90(train_anal, 1, [1, 2])

# train_pro_all = torch.cat((train_pro, train_pro_x90, train_pro_y90, train_pro_z90), 0)
# train_pot_all = torch.cat((train_pot, train_pot_x90, train_pot_y90, train_pot_z90), 0)
# train_anal_all = torch.cat((train_anal, train_anal_x90, train_anal_y90, train_anal_z90), 0)

train_pro_all, train_pot_all, train_anal_all = gen_rot_boxes(train_pro, train_pot, train_anal)

train_ds = Ds(train_pro_all, train_pot_all, train_anal_all)

val_data = np.load("val.npy.npz")
val_x = torch.from_numpy(val_data['pro'])
val_pot = torch.from_numpy(np.log10(np.array(val_data['pot'])))
val_y = torch.from_numpy(val_data['y'])
val_ds = Ds(val_x, val_pot, val_y)

train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=bs, shuffle=True, drop_last=True)

# train_ds = Ds(train_y, train_x)
# vali_ds = Ds(vali_y, vali_x)

# train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
# vali_loader = DataLoader(vali_ds, batch_size=bs, shuffle=False)


    
def train_model(model, model_name: str, crit = 'mae', opt = 'adam'):
    mn = model_name

    model = model.to(device)

    if opt == 'adam':
        optimizer = optim.NAdam(model.parameters(), lr = 0.0002)
    elif opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)

    # mae = nn.L1Loss().to(device)
        
    loss = []
    val_loss = []

    criterion = nn.L1Loss().to(device)
    grad_criterion = GradientDifferenceLoss().to(device)

    # allmae = []
    # allvalmae = []

    for epoch in range(NUM_EPOCHS):
        eloss = []
        # emae = []
        for i, data in enumerate(train_loader):
            model.zero_grad()

            x, pot, y = [d.to(device) for d in data]
            y = y.to(torch.float32)

            preds = model(x, pot)

            if crit == 'mae':
                err = criterion(preds, y)
            elif crit =='grad':
                err = grad_criterion(preds, y) + criterion(preds, y)
            err.backward()

            # tmae = mae(preds, y)

            eloss.append(err.cpu().item())
            # emae.append(tmae.cpu().item())

            if i % 100 == 0:
                print(f"EPOCH {epoch}/{NUM_EPOCHS} BATCH {i} Loss: {np.mean(np.array(eloss))}")

            optimizer.step()

        loss.append(np.mean(np.array(eloss)))

        # allmae.append(np.mean(np.array(emae)))


        vloss = []
        # valmae = []
        for i, data in enumerate(val_loader):
            model.zero_grad()
            model.eval()

            x, pot, y = [d.to(device) for d in data]

            preds = model(x, pot)

            err = criterion(preds, y)

            # vmae = mae(preds, y)

            vloss.append(err.cpu().item())
            # valmae.append(vmae.cpu().item())

        val_loss.append(np.mean(np.array(vloss)))

        # allvalmae.append(np.mean(np.array(valmae)))

        print(f"EPOCH {epoch}/{NUM_EPOCHS} tLoss: {np.mean(np.array(eloss))} invLoss: {np.mean(np.array(vloss))}")
        # print(f"EPOCH {epoch}/{NUM_EPOCHS} tMAE: {np.mean(np.array(emae))} invMAE: {np.mean(np.array(valmae))}")

        viz.line(
            X = [epoch], Y = [loss[-1]], 
            name = 'train_loss', 
            win = f'{mn}_{crit}_{opt}', 
            update = 'append',
                opts = {
            'showlegend': True,
            'xlabel': "epoch",
            'ylabel': "value",
        })
        viz.line(X = [epoch], Y = [val_loss[-1]], 
                name = 'val_loss', 
                win = f'{mn}_{crit}_{opt}', 
                update = 'append',     
                opts = {
            'showlegend': True,
            'xlabel': "epoch",
            'ylabel': "value",
        })
        # viz.line(X = [epoch], Y = [allmae[-1]], 
        #         name = 'train_mae', 
        #         win = f'{mn}', 
        #         update = 'append',     
        #         opts = {
        #     'showlegend': True,
        #     'xlabel': "epoch",
        #     'ylabel': "value",
        # })
        # viz.line(X = [epoch], Y = [allvalmae[-1]], 
        #         name = 'val_mae', 
        #         win = f'{mn}', 
        #         update = 'append',     
        #         opts = {
        #     'showlegend': True,
        #     'xlabel': "epoch",
        #     'ylabel': "value",
        # })
        
        torch.save(model, f"{mn}_{crit}_{opt}.pth")
        np.savetxt(f"{mn}_{crit}_{opt}_loss.csv", np.array([np.array(loss), np.array(val_loss)]).T)

models = {
    'CNN': GED_CNN1(val_ds.item_shape(), bs, with_res = True),
}



for name, model in models.items():
    for crit in ['mae', 'grad']:
        for opt in ['adam', 'sgd']:
            train_model(model, name, crit, opt)


