import numpy as np
import math

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from wfn_reader import read_wfn
from wfn_calc import *

from model import *
from global_constant import BATCH_SIZE, BOX_SIZE

from gencube import *

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = torch.load("./saved_models/512conv/withres/CNN_mae_sgd.pth").to(device)

bs = BATCH_SIZE

filename = "./training_wfns/pyrene_sp.wfn"

mol, functions, prim_matrix, occ_MOs = read_wfn(filename)

grid_of_box = gen_grids_of_box(mol, BOX_SIZE)

prodens = torch.tensor(calc_box_prodens(mol, grid_of_box)).to(device)
potential = torch.tensor(np.log10(np.array(calc_potential(mol, grid_of_box)))).to(device)
analdens = calc_box_dens(mol, functions, prim_matrix, grid_of_box, occ_MOs)

x, y, z = prodens.shape

print(x,y,z)
smallest_dim = min(x, y, z)
zoi = int(np.floor(z / 2))

xn = int(x / BOX_SIZE)
yn = int(y / BOX_SIZE)
zn = int(z / BOX_SIZE)

pred_result = torch.zeros((x, y, z)).to(device)

with torch.no_grad():
    model.eval()

    pro_box = torch.zeros((bs, BOX_SIZE, BOX_SIZE, BOX_SIZE)).to(device)
    pot_box = torch.zeros((bs, BOX_SIZE, BOX_SIZE, BOX_SIZE)).to(device)

    for i in range(xn):
        for j in range(yn):
            for k in range(zn):
                pro_box = prodens[i * BOX_SIZE:(i + 1) * BOX_SIZE, j * BOX_SIZE:(j + 1) * BOX_SIZE, k * BOX_SIZE:(k + 1) * BOX_SIZE].repeat(bs, 1, 1, 1).view(bs, 1, BOX_SIZE, BOX_SIZE, BOX_SIZE)
                pot_box = potential[i * BOX_SIZE:(i + 1) * BOX_SIZE, j * BOX_SIZE:(j + 1) * BOX_SIZE, k * BOX_SIZE:(k + 1) * BOX_SIZE].repeat(bs, 1, 1, 1).view(bs, 1, BOX_SIZE, BOX_SIZE, BOX_SIZE)

                pred_result[i * BOX_SIZE:(i + 1) * BOX_SIZE, j * BOX_SIZE:(j + 1) * BOX_SIZE, k * BOX_SIZE:(k + 1) * BOX_SIZE] = torch.mean(model(pro_box, pot_box), dim = 0)

    pro_box.detach()
    pot_box.detach()

pred_result = pred_result.cpu().detach().numpy()

print(f"MSighE: {np.mean(pred_result - analdens)}, MAE: {np.mean(np.abs(pred_result - analdens))}, rMAE: {np.mean(np.abs(pred_result - analdens)/analdens)}\n")
print(f"MaxErr: {np.max(np.abs(pred_result - analdens))}, MinErr: {np.min(np.abs(pred_result - analdens))}")

# truth = analdens.flatten()
# preds = pred_result.flatten()

# np.savetxt("./saved_models/512conv/withres/stage2/pred_result.csv", np.stack([preds, truth], axis = 1))
# np.savetxt("./saved_models/512conv/withres/stage2/pred_plane.csv", pred_result[:, :, zoi])
# np.savetxt("./saved_models/512conv/withres/stage2/truth_plane.csv", analdens[:, :, zoi])
# np.savetxt("./saved_models/512conv/withres/stage2/residual.csv", pred_result[:, :, zoi] - analdens[:, :, zoi])

gen_cube("pred_stage1.cub", mol, pred_result)


