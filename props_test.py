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

# model = torch.load("./saved_models/512conv/withres/CNN_mae_sgd.pth").to(device)

bs = BATCH_SIZE

def test_wfn(filename, model):

    mol, functions, prim_matrix, occ_MOs = read_wfn(filename)

    basename = filename.split("/")[-1].split(".")[0]

    grid_of_box = gen_grids_of_box(mol, BOX_SIZE)

    T1 = time.time()
    prodens = torch.tensor(calc_box_prodens(mol, grid_of_box)).to(device)
    potential = torch.tensor(np.log10(np.array(calc_potential(mol, grid_of_box)))).to(device)
    T2 = time.time()

    analdens = calc_box_dens(mol, functions, prim_matrix, grid_of_box, occ_MOs)

    x, y, z = prodens.shape
    # smallest_dim = min(x, y, z)
    # x_of_interest = int(np.floor(x / 2))

    T3 = time.time()

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

    T4 = time.time()

    print(f"MSighE: {np.mean(pred_result - analdens)}, MAE: {np.mean(np.abs(pred_result - analdens))}, rMAE: {np.mean(np.abs(pred_result - analdens)/analdens)}\n")
    print(f"MaxErr: {np.max(np.abs(pred_result - analdens))}, MinErr: {np.min(np.abs(pred_result - analdens))}")

    #np.savetxt("pred_result.csv", np.stack([preds, truth], axis = 1))

    mae = np.mean(np.abs(pred_result - analdens))
    # maxae = np.max(np.abs(pred_result - analdens))
    # minae = np.min(np.abs(pred_result - analdens))

    atoms = np.array(mol.get_atoms())
    coords = np.array(mol.get_coordinates())

    cx = np.sum(atoms * coords[...,0]) /  np.sum(atoms)
    cy = np.sum(atoms * coords[...,1]) /  np.sum(atoms)
    cz = np.sum(atoms * coords[...,2]) /  np.sum(atoms)

    mc = np.array([cx, cy, cz])

    x_time = T2-T1
    pred_time = T4-T3

    return analdens, pred_result, grid_of_box, mol, mc, x_time, pred_time

def get_mass_center(grids_val, grids):
    '''
        Calculate the mass center of the electron density.
    '''

    shape = grids_val.shape
    #print(shape)

    idx_matrix = grids.reshape(shape + (3,))
    print(idx_matrix.shape)

    cx = np.sum(grids_val * idx_matrix[...,0]) / np.sum(grids_val)
    cy = np.sum(grids_val * idx_matrix[...,1]) / np.sum(grids_val)
    cz = np.sum(grids_val * idx_matrix[...,2]) / np.sum(grids_val)

    return cx,cy,cz

def get_gradient(grids):
    gradient = [g.flatten() for g in np.gradient(grids)]
    norm_gradient = [np.linalg.norm(np.array([gradient[0][i], gradient[1][i], gradient[2][i]])) for i in range(len(gradient[0]))]

    return norm_gradient

def get_vol(grid_vals, res = RESOLUTION, threshold = 0.002):
    '''
        Calculate the molecular volume by counting number of grids with electron density higher than the threshold.
    '''

    x,y,z = grid_vals.shape
    total_vol = x * y * z / RESOLUTION ** 3

    num_in_vdw = grid_vals[grid_vals >= threshold].size
    monte_carlo_vol = total_vol * num_in_vdw / (x * y * z)

    return monte_carlo_vol

def get_atom_ele_potential(atom_number, atom_coord, rho, grids):
    '''
        Calculate the attraction potential between one atom with all electrons.
    '''

    v_mat = atom_number * rho / np.sqrt((grids[...,0] - atom_coord[0]) ** 2 + (grids[...,1] - atom_coord[1]) ** 2 + (grids[...,2] - atom_coord[2]) ** 2)
    return np.sum(v_mat)

def get_nuc_ele_potential(mol, rho, grids, res = RESOLUTION):
    '''
        Calculate the attraction potential between all atoms with all electrons.
    '''

    atoms = mol.get_atoms()
    coordinates = mol.get_coordinates()

    v_list = np.array([get_atom_ele_potential(atom_number, atom_coord, rho, grids) / res ** 3 for atom_number, atom_coord in zip(atoms, coordinates)])
    return np.sum(v_list)




# truth_gradient = []
# preds_gradient = []

# truth_laplacian = []
# preds_laplacian = []



def test_all(stage, setname, model):

    files = sorted(glob.glob(f"./training_wfns/{setname}/*.wfn"))

    f1 = open(f"./wfn_for_props/results/{setname}/stage{stage}/dipoles.txt", "w")
    f1.write("filename\tpreds\ttruth\n")

    f3 = open(f"./wfn_for_props/results/{setname}/stage{stage}/int_rho.txt", "w")
    f3.write("filename\tpreds\ttruth\n")
    f4 = open(f"./wfn_for_props/results/{setname}/stage{stage}/volume.txt", "w")
    f4.write(f"filename\tpreds\ttruth\n")
    f5 = open(f"./wfn_for_props/results/{setname}/stage{stage}/vne.txt", "w")
    f5.write("filename\tpreds\ttruth\n")

    f1.close()
    f3.close()
    f4.close()
    f5.close()

    f2 = open(f"./wfn_for_props/results/{setname}/stage{stage}/exc_lda.txt", "w")
    f2.write("filename\tpreds\ttruth\n")
    f2.close()

    f6 = open(f"./wfn_for_props/results/{setname}/stage{stage}/exc_gga.txt", "w")
    f6.write("filename\tpreds\ttruth\n")
    f6.close()

    f7 = open(f"./wfn_for_props/results/{setname}/stage{stage}/time.txt", "w")
    f7.write("filename\tx_time\tpred_time\tn_grids\tn_atoms\n")
    f7.close()

    for file in files:
        basename = os.path.basename(file).split('.')[0]
        truth, preds, grids, mol, mc, x_time, pred_time = test_wfn(file, model)

        ngrids = truth.shape[0] * truth.shape[1] * truth.shape[2]

        f7 = open(f"./wfn_for_props/results/{setname}/stage{stage}/time.txt", "a+")
        f7.write(f"{basename}\t{x_time}\t{pred_time}\t{ngrids}\t{len(mol.get_atoms())}\n")

        tcx, tcy, tcz = get_mass_center(truth, grids)
        t_charge = np.sum(truth) / RESOLUTION**3

        print(f"Total charge: {t_charge}")

        pcx, pcy, pcz = get_mass_center(preds, grids)
        p_charge = np.sum(preds) / RESOLUTION**3

        print(f"Total charge (pred): {p_charge}")

        t_dipole = np.linalg.norm(np.array([tcx, tcy, tcz]) - mc) * t_charge
        p_dipole = np.linalg.norm(np.array([pcx, pcy, pcz]) - mc) * p_charge

        f1 = open(f"./wfn_for_props/results/{setname}/stage{stage}/dipoles.txt", "a")

        f1.write(f"{basename}\t{p_dipole}\t{t_dipole}\n")
        f1.close()

        f3 = open(f"./wfn_for_props/results/{setname}/stage{stage}/int_rho.txt", "a")
        f3.write(f"{basename}\t{p_charge}\t{t_charge}\n")
        f3.close()

        t_vol = get_vol(truth)
        p_vol = get_vol(preds)

        f4 = open(f"./wfn_for_props/results/{setname}/stage{stage}/volume.txt", "a")
        f4.write(f"{basename}\t{p_vol}\t{t_vol}\n")
        f4.close()

        grids = grids.reshape((-1,3))

        t_vne = rust_wfnkit.calc_vne(mol, truth.flatten(), grids) / RESOLUTION**3
        p_vne = rust_wfnkit.calc_vne(mol, preds.flatten(), grids) / RESOLUTION**3

        f5 = open(f"./wfn_for_props/results/{setname}/stage{stage}/vne.txt", "a")
        f5.write(f"{basename}\t{p_vne}\t{t_vne}\n")
        f5.close

        f2 = open(f"./wfn_for_props/results/{setname}/stage{stage}/exc_lda.txt", "a+")

        t_xc = dft.libxc.eval_xc('LDA_x,VWN', truth.flatten())
        p_xc = dft.libxc.eval_xc('LDA_x,VWN', preds.flatten())

        t_xc = np.sum(t_xc[0]) / (BOX_SIZE ** 3)
        p_xc = np.sum(p_xc[0]) / (BOX_SIZE ** 3)

        f2.write(f"{basename}\t{p_xc}\t{t_xc}\n")
        f2.close

        f6 = open(f"./wfn_for_props/results/{setname}/stage{stage}/exc_gga.txt", "a+")

        t_grad = [truth.flatten()] + [g.flatten() for g in np.gradient(truth)]
        p_grad = [preds.flatten()] + [g.flatten() for g in np.gradient(preds)]

        t_grad = np.array(t_grad)
        p_grad = np.array(p_grad)

        t_xc = dft.libxc.eval_xc('b88,lyp', t_grad)
        p_xc = dft.libxc.eval_xc('b88,lyp', p_grad)

        t_xc = np.sum(t_xc[0]) / (BOX_SIZE ** 3)
        p_xc = np.sum(p_xc[0]) / (BOX_SIZE ** 3)

        f6.write(f"{basename}\t{p_xc}\t{t_xc}\n")
        f6.close


def test_mae(stage, setname, model):

    files = sorted(glob.glob(f"./training_wfns/{setname}/*.wfn"))

    maes = []
    ns = []

    for file in files:
        basename = os.path.basename(file).split('.')[0]
        truth, preds, grids, mol, mc, x_time, pred_time = test_wfn(file, model)

        ngrids = truth.shape[0] * truth.shape[1] * truth.shape[2]

        mae = np.mean(np.abs(truth - preds))

        maes.append(mae)
        ns.append(ngrids)

    all_mae = np.sum(np.array(maes) * np.array(ns) / np.sum(ns))

    return all_mae, np.sum(ns)

# model = torch.load("./CNN_mae_sgd.pth").to(device)
# model2 = torch.load("./CNN_grad_sgd.pth").to(device)

# with open("./mae_result.txt", "w+") as f:
#     ab_mae, ab_ns = test_mae(1, "asbase", model)
#     qm9_mae, qm9_ns = test_mae(1, "qm9", model)
#     t_mae, t_ns = test_mae(1, "training", model)
#     f.write(f"asbase\t{ab_mae}\n")
#     f.write(f"qm9\t{qm9_mae}\n")
#     f.write(f"training\t{t_mae}\n")
#     avg_mae = (ab_mae * ab_ns + qm9_mae * qm9_ns + t_mae * t_ns) / (ab_ns + qm9_ns + t_ns)
#     f.write(f"all\t{avg_mae}\n")






        