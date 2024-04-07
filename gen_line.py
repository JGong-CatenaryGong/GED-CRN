import numpy as np
import math

from memory_profiler import profile

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from wfn_reader import read_wfn
from wfn_calc import *

from model import *
from global_constant import BATCH_SIZE, BOX_SIZE, RESOLUTION
import os, glob

from pyscf import dft

import rust_wfnkit
import time

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"The prediction is running on {device}")

bs = BATCH_SIZE
from props_test import *

def test_in(setname, model):

    files = sorted(glob.glob(f"./training_wfns/{setname}/*.wfn"))

    grids = 0

    for file in files:
        basename = os.path.basename(file).split('.')[0]
        mol, functions, prim_matrix, occ_MOs = read_wfn(file)
        grid_of_box = gen_grids_of_box(mol, BOX_SIZE)

        x, y, z, d = grid_of_box.shape

        grids += x * y * z

    return grids

model = torch.load("./CNN_mae_sgd.pth").to(device)

training_mae = test_in("training", model)
qm9_mae = test_in("qm9", model)
asbase_mae = test_in("asbase", model)

with open("./mae_result.txt", "a+") as f:
    f.write(f"training\t{0.4575 * 19 / training_mae}\n")
    f.write(f"qm9\t{0.4194 * 100 / qm9_mae}\n")
    f.write(f"asbase\t{0.3783 * 72 / asbase_mae}\n")