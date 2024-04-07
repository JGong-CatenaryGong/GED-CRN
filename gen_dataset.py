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
# qm9_files = np.array(glob.glob(os.path.join('./training_wfns/qm9', suffix)))
# asbase_files = np.array(glob.glob(os.path.join('./training_wfns/asbase', suffix)))

# qm9_samples = qm9_files[np.random.randint(0, len(qm9_files), 30)]
# asbase_samples = asbase_files[np.random.randint(0, len(asbase_files), 30)]
# files = np.concatenate((qm9_samples, main_files))

files = main_files

train_pro_box = []
train_anal_box = []
train_potential_box = []
train_names = []
with tqdm(total=len(files)) as pbar:
    pbar.set_description('Processing training data:')
    for file in files:
        basename = os.path.basename(file)

        mol, functions, prim_matrix, occs = read_wfn(file)
        vol = grid_vol(mol)

        print(f"{file}, scale: {vol}")

        u_samples = int(np.ceil(vol * 0.5))
        n_samples = int(np.ceil(vol * 0.5))
        boxes = gen_samples(mol, u_samples, method = "uniform") + gen_samples(mol, n_samples, method = "normal")

        name_list = u_samples * [basename] + n_samples * [basename]

        train_names += name_list

        anal_samples = []
        pro_samples = []
        pot_samples = []
        for box in boxes:
            dens_box = calc_box_dens(mol, functions, prim_matrix, box, occ_MOs = occs)
            anal_samples.append(dens_box)

            pro_box = calc_box_prodens(mol, box)
            pro_samples.append(pro_box)

            pot_box = calc_potential(mol, box)
            pot_samples.append(pot_box)

        train_pro_box += pro_samples
        train_anal_box += anal_samples
        train_potential_box += pot_samples

        pbar.update(1)

    train_dic = {
        "pro": train_pro_box,
        "pot": train_potential_box,
        "anal": train_anal_box,
        "name": train_names,
    }

np.savez('train.npy', pro = train_dic['pro'], pot = train_dic['pot'], y = train_dic['anal'])

val_pro_box = []
val_anal_box = []
val_potential_box = []
val_names = []
with tqdm(total=len(files)) as pbar2:
    pbar2.set_description('Processing validating data:')
    for file in files:
        basename = os.path.basename(file)

        mol, functions, prim_matrix, occs = read_wfn(file)
        vol = grid_vol(mol)

        print(f"{file}, scale: {vol}")

        u_samples = int(np.ceil(vol * 0.1))
        n_samples = int(np.ceil(vol * 0.1))
        boxes = gen_samples(mol, u_samples, method = "uniform") + gen_samples(mol, n_samples, method = "normal")

        name_list = u_samples * [basename] + n_samples * [basename]

        anal_samples = []
        pro_samples = []
        pot_samples = []
        for box in boxes:
            dens_box = calc_box_dens(mol, functions, prim_matrix, box, occ_MOs = occs)
            anal_samples.append(dens_box)

            pro_box = calc_box_prodens(mol, box)
            pro_samples.append(pro_box)

            pot_box = calc_potential(mol, box)
            pot_samples.append(pot_box)       

        val_pro_box += pro_samples
        val_anal_box += anal_samples
        val_potential_box += pot_samples

        pbar2.update(1)

    val_dic = {
        "pro": val_pro_box,
        "pot": val_potential_box,
        "anal": val_anal_box,
        "name":  val_names,
    }

np.savez('val.npy', pro = val_dic['pro'], pot = val_dic['pot'], y = val_dic['anal'])







