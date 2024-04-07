from wfn_calc import calc_analytical_dens, calc_prodens, grid_vol, gen_samples, calc_box_dens, calc_box_prodens, calc_mol_potential, calc_potential
from wfn_reader import read_wfn
import numpy as np
import glob, os
import h5py
from global_constant import *
from tqdm import tqdm

# filename = './training_wfns/hr.wfn'

# mol, functions, prim_matrix, occs = read_wfn(filename)

# anal_dens = calc_analytical_dens(mol, functions, prim_matrix, occs)
# pro_dens = calc_prodens(mol)

# print(np.sum(anal_dens)/64)
# print(np.sum(pro_dens)/64)

dir = './training_wfns/'

suffix = '*.wfn'

main_files = np.array(glob.glob(os.path.join(dir, suffix)))
qm9_files = np.array(glob.glob(os.path.join('./training_wfns/qm9', suffix)))
asbase_files = np.array(glob.glob(os.path.join('./training_wfns/asbase', suffix)))

# qm9_samples = qm9_files[np.random.randint(0, len(qm9_files), 50)]
# asbase_samples = asbase_files[np.random.randint(0, len(asbase_files), 30)]
# files = np.concatenate((qm9_samples, main_files))

files = main_files

train_names = []
with tqdm(total=len(files)) as pbar:
    pbar.set_description('Processing training data:')
    for file in files:
        mol, functions, prim_matrix, occs = read_wfn(file)
        vol = grid_vol(mol)

        basename = os.path.basename(file)

        print(f"{file}, scale: {vol}")

        u_samples = int(np.ceil(vol * 0.8))
        n_samples = int(np.ceil(vol * 0.2))

        name_list = u_samples * [basename] + n_samples * [basename]

        train_names += name_list

        pbar.update(1)

np.savez('train_names.npy', name = train_names)

val_names = []
with tqdm(total=len(files)) as pbar2:
    pbar2.set_description('Processing validating data:')
    for file in files:
        mol, functions, prim_matrix, occs = read_wfn(file)
        vol = grid_vol(mol)

        print(f"{file}, scale: {vol}")

        u_samples = int(np.ceil(vol * 0.16))
        n_samples = int(np.ceil(vol * 0.04))
        #boxes = gen_samples(mol, u_samples, method = "uniform") + gen_samples(mol, n_samples, method = "normal")

        name_list = u_samples * [basename] + n_samples * [basename]

        val_names += name_list

        pbar2.update(1)

np.savez('val_names.npy', name = val_names)







