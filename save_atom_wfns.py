from wfn_reader import read_wfn
import numpy as np
import time
import os, glob
import h5py

dir = './atomic_wfns/'

suffix = '*.wfn'

files = glob.glob(os.path.join(dir, suffix))

for file in files:
    mol, functions, prim_matrix, occs = read_wfn(file)

    center_list = [f.get_center() for f in functions]
    type_list = [f.get_func_type() for f in functions]
    exp_list = [f.get_exponent() for f in functions]
    occs_list = occs
    
    dic = {
        "mol_atoms": mol.get_atoms(),
        "mol_geom": mol.get_coordinates(),
        "func_centers": center_list,
        "func_types": type_list,
        "func_exps": exp_list,
        "prims": prim_matrix,
        "occs": occs_list
    }

    with h5py.File(file.replace(".wfn", ".h5"), "w") as f:
        for key, val in dic.items():
            f.create_dataset(key, data = np.array(val))