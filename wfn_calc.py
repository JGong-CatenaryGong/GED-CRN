from wfn_reader import read_wfn
import rust_wfnkit
import numpy as np
import time
import glob
import h5py, os
import scipy
#from for_cell import pdb_to_cell

from global_constant import *

def gen_grids_of_box(mol, box_size, margin_ratio = 2.0, resolution = RESOLUTION):

    '''
        Generate grids coordinates that are integer multiples of the box size.
    '''

    coordinates = mol.get_coordinates()

    xs = [coord[0] for coord in coordinates]
    ys = [coord[1] for coord in coordinates]
    zs = [coord[2] for coord in coordinates]

    print(f"Generating grids ...")

    len_x = max(xs) - min(xs)
    len_y = max(ys) - min(ys)
    len_z = max(zs) - min(zs)

    cx = min(xs) + len_x / 2
    cy = min(ys) + len_y / 2
    cz = min(zs) + len_z / 2

    box_x = np.ceil(len_x * margin_ratio) if np.ceil(len_x * margin_ratio) > 16 else 16
    box_y = np.ceil(len_y * margin_ratio) if np.ceil(len_y * margin_ratio) > 16 else 16
    box_z = np.ceil(len_z * margin_ratio) if np.ceil(len_z * margin_ratio) > 16 else 16

    box_x = np.ceil(box_x / box_size) * box_size
    box_y = np.ceil(box_y / box_size) * box_size
    box_z = np.ceil(box_z / box_size) * box_size

    print(f"Center at {cx, cy, cz} with box {box_x, box_y, box_z}.")

    stride = 1 / resolution

    x_start = cx - box_x / 2
    y_start = cy - box_y / 2
    z_start = cz - box_z / 2

    x_end = cx + box_x / 2
    y_end = cy + box_y / 2
    z_end = cz + box_z / 2

    grids_coords = np.zeros((int(box_x * resolution), int(box_y * resolution), int(box_z * resolution), 3))
    for i in range(int(box_x * resolution)):
        for j in range(int(box_y * resolution)):
            for k in range(int(box_z * resolution)):
                x = x_start + i * stride
                y = y_start + j * stride
                z = z_start + k * stride
                grids_coords[i, j, k] = [x, y, z]

    return grids_coords


def gen_grids(mol, margin_ratio = 2.0, resolution = RESOLUTION):
    coordinates = mol.get_coordinates()

    xs = [coord[0] for coord in coordinates]
    ys = [coord[1] for coord in coordinates]
    zs = [coord[2] for coord in coordinates]

    print(f"Generating grids ...")

    len_x = max(xs) - min(xs)
    len_y = max(ys) - min(ys)
    len_z = max(zs) - min(zs)

    cx = min(xs) + len_x / 2
    cy = min(ys) + len_y / 2
    cz = min(zs) + len_z / 2

    box_x = np.ceil(len_x * margin_ratio) if np.ceil(len_x * margin_ratio) > 8 else 8
    box_y = np.ceil(len_y * margin_ratio) if np.ceil(len_y * margin_ratio) > 8 else 8
    box_z = np.ceil(len_z * margin_ratio) if np.ceil(len_z * margin_ratio) > 8 else 8

    print(f"Center at {cx, cy, cz} with box {box_x, box_y, box_z}.")

    stride = 1 / resolution

    x_start = cx - box_x / 2
    y_start = cy - box_y / 2
    z_start = cz - box_z / 2

    x_end = cx + box_x / 2
    y_end = cy + box_y / 2
    z_end = cz + box_z / 2

    grids_coords = np.zeros((int(box_x * resolution), int(box_y * resolution), int(box_z * resolution), 3))
    for i in range(int(box_x * resolution)):
        for j in range(int(box_y * resolution)):
            for k in range(int(box_z * resolution)):
                x = x_start + i * stride
                y = y_start + j * stride
                z = z_start + k * stride
                grids_coords[i, j, k] = [x, y, z]

    return grids_coords

def grid_vol(mol):
    coordinates = mol.get_coordinates()

    xs = [coord[0] for coord in coordinates]
    ys = [coord[1] for coord in coordinates]
    zs = [coord[2] for coord in coordinates]

    print(f"Generating grids ...")

    len_x = max(xs) - min(xs) if max(xs) - min(xs) > 4 else 4
    len_y = max(ys) - min(ys) if max(ys) - min(ys) > 4 else 4
    len_z = max(zs) - min(zs) if max(zs) - min(zs) > 4 else 4

    return len_x * len_y * len_z

def gen_samples(mol, nsamples: int, resolution = RESOLUTION, box_size = BOX_SIZE, method = "normal"):
    box_size = box_size / resolution
    coordinates = mol.get_coordinates()

    xs = [coord[0] for coord in coordinates]
    ys = [coord[1] for coord in coordinates]
    zs = [coord[2] for coord in coordinates]

    print(f"Generating boxes ...")

    len_x = max(xs) - min(xs)
    len_y = max(ys) - min(ys)
    len_z = max(zs) - min(zs)

    cx = min(xs) + len_x / 2
    cy = min(ys) + len_y / 2
    cz = min(zs) + len_z / 2

    if method == "normal":

        # Normalization distributed sampling
        x_0 = np.random.normal(cx, len_x/2, nsamples)
        y_0 = np.random.normal(cy, len_y/2, nsamples)
        z_0 = np.random.normal(cz, len_z/2, nsamples)

    elif method == "uniform":
        
        # Uniform randomly sampling
        x_0 = np.random.uniform(cx - len_x, cx + len_x, nsamples)
        y_0 = np.random.uniform(cy - len_y, cy + len_y, nsamples)
        z_0 = np.random.uniform(cz - len_z, cz + len_z, nsamples)

    stride = 1/resolution

    boxes = []
    for i, box in enumerate(x_0):
        grid_coords = np.zeros((int(box_size * resolution), int(box_size * resolution), int(box_size * resolution), 3))
        for j in range(int(box_size * resolution)):
            for k in range(int(box_size * resolution)):
                for l in range(int(box_size * resolution)):
                    grid_coords[j, k, l] = [x_0[i] + j * stride, y_0[i] + k * stride, z_0[i] + l * stride]
        boxes.append(grid_coords)

    return boxes

def calc_wfn(mol, functions, prim_matrix):

    center_list = [f.get_center() for f in functions]
    type_list = [f.get_func_type() for f in functions]
    exp_list = [f.get_exponent() for f in functions]

    prim_matrix = prim_matrix.T

    prim_matrix = rust_wfnkit.PrimMat(prim_matrix.reshape(-1), prim_matrix.shape[0], prim_matrix.shape[1])

    grids = gen_grids(mol)
    xn, yn, zn, d = grids.shape
    serialized_grids = grids.reshape(xn * yn * zn, 3)

    print("Calculating wavefunctions of every MOs on every grids ...")

    T1 = time.perf_counter()

    wfn = rust_wfnkit.calc_wfn_grids(mol, center_list, type_list, exp_list, serialized_grids, prim_matrix)

    T2 = time.perf_counter()

    print(f"Total calculation time: {(T2 - T1)} s")

    wfn = np.array(wfn).reshape(xn, yn, zn, -1)
    return wfn

# from memory_profiler import profile
# @profile
def calc_box_dens(mol, functions, prim_matrix, grids, occ_MOs):

    center_list = [f.get_center() for f in functions]
    type_list = [f.get_func_type() for f in functions]
    exp_list = [f.get_exponent() for f in functions]

    prim_matrix = prim_matrix.T

    prim_matrix = rust_wfnkit.PrimMat(prim_matrix.reshape(-1), prim_matrix.shape[0], prim_matrix.shape[1])

    xn, yn, zn, d = grids.shape
    serialized_grids = grids.reshape(xn * yn * zn, 3)

    T1 = time.perf_counter()

    dens = np.array(rust_wfnkit.calc_dens_grids(mol, center_list, type_list, exp_list, serialized_grids, prim_matrix, occ_MOs)).reshape(xn, yn, zn)

    T2 = time.perf_counter()

    # wfn = np.array(wfn).reshape(xn, yn, zn, -1)
    
    # dens = np.sum((occ_MOs * np.square(wfn)), axis = -1)
    return dens

def gen_atom_wavefunctions():
    atomic_centers = [[1]] * 118
    atomic_types = [[1]] * 118
    atomic_exponents = [[0.0]] * 118
    atomic_prims = [[[0.0]]] * 118
    atomic_occs = [[2.0]] * 118

    suffix = '*.h5'
    search_direction = './atomic_wfns'

    files = glob.glob(os.path.join(search_direction, suffix))
    print(files)

    print(f"Searching h5 files ...")
    for i in range(118):
        if os.path.join(search_direction, ELEMENTS[i - 1] + '.h5') in files:
            with h5py.File(os.path.join(search_direction, ELEMENTS[i - 1] + '.h5'), 'r') as f:
                atomic_centers[i - 1] = np.array(f['func_centers'])
                atomic_types[i - 1] = np.array(f['func_types'])
                atomic_exponents[i - 1] = np.array(f['func_exps'])
                atomic_prims[i - 1] = np.array(f['prims']).T
                atomic_occs[i - 1] = np.array(f['occs']).T
            print(f"found {ELEMENTS[i - 1]}.h5 with {len(atomic_centers[i - 1])} GTFs.")

    #print(atomic_centers)
            
    print(atomic_occs)

    at_wfns = rust_wfnkit.Wavefunction(atomic_centers, atomic_types, atomic_exponents, atomic_prims, atomic_occs)
    return at_wfns

# def calc_analytical_dens(mol, functions, prim_matrix, occ_MOs):
#     wfn = calc_wfn(mol, functions, prim_matrix)
#     dens = np.sum((occ_MOs * np.square(wfn)), axis = -1)
#     return dens

def calc_analytical_dens(mol, functions, prim_matrix, occ_MOs):

    grids = gen_grids(mol)
    xn, yn, zn, d = grids.shape
    serialized_grids = grids.reshape(xn * yn * zn, 3)

    center_list = [f.get_center() for f in functions]
    type_list = [f.get_func_type() for f in functions]
    exp_list = [f.get_exponent() for f in functions]

    prim_matrix = prim_matrix.T

    prim_matrix = rust_wfnkit.PrimMat(prim_matrix.reshape(-1), prim_matrix.shape[0], prim_matrix.shape[1])

    xn, yn, zn, d = grids.shape
    serialized_grids = grids.reshape(xn * yn * zn, 3)

    T1 = time.perf_counter()

    dens = np.array(rust_wfnkit.calc_dens_grids(mol, center_list, type_list, exp_list, serialized_grids, prim_matrix, occ_MOs)).reshape(xn, yn, zn)

    T2 = time.perf_counter()

    # wfn = np.array(wfn).reshape(xn, yn, zn, -1)
    
    # dens = np.sum((occ_MOs * np.square(wfn)), axis = -1)
    return dens

def calc_prodens(mol):

    at_wfns = gen_atom_wavefunctions()

    grids = gen_grids(mol)
    xn, yn, zn, d = grids.shape
    serialized_grids = grids.reshape(xn * yn * zn, 3)

    T1 = time.perf_counter()

    prodens = rust_wfnkit.calc_pro_dens2(mol, at_wfns, serialized_grids)

    T2 = time.perf_counter()

    print(f"Total calculation time: {(T2 - T1)} s")

    prodens = np.array(prodens).reshape(xn, yn, zn)
    return prodens

def calc_box_prodens(mol, grids):

    at_wfns = gen_atom_wavefunctions()

    xn, yn, zn, d = grids.shape
    serialized_grids = grids.reshape(xn * yn * zn, 3)

    T1 = time.perf_counter()

    prodens = rust_wfnkit.calc_pro_dens2(mol, at_wfns, serialized_grids)

    T2 = time.perf_counter()

    prodens = np.array(prodens).reshape(xn, yn, zn)
    return prodens

def calc_potential(mol, grids):

    xn, yn, zn, d = grids.shape
    serialized_grids = grids.reshape(xn * yn * zn, 3)

    T1 = time.perf_counter()

    potential = rust_wfnkit.calc_potential(mol, serialized_grids)

    T2 = time.perf_counter()

    potential = np.array(potential).reshape(xn, yn, zn)
    return potential

def calc_mol_potential(mol):
    grids = gen_grids(mol)

    potential = calc_potential(mol, grids)

    return potential

# def calc_cell_prodens(cell):
#     fA = cell.a
#     fB = cell.b
#     fC = cell.c

#     print(cell.atoms())

#     at_wfns = gen_atom_wavefunctions()

#     atoms = cell.atoms()
#     coordinates = cell.coords()

#     mol = rust_wfnkit.Molecule(atoms, coordinates)

#     cell_info = [fA, fB, fC]

#     grids = gen_grids(mol)
#     xn, yn, zn, d = grids.shape
#     serialized_grids = grids.reshape(xn * yn * zn, 3)

#     T1 = time.perf_counter()

#     prodens = rust_wfnkit.calc_pro_dens_for_cell(mol, at_wfns, serialized_grids, cell_info)

#     T2 = time.perf_counter()

#     print(f"Total calculation time: {(T2 - T1)} s")

#     prodens = np.array(prodens).reshape(xn, yn, zn)
#     return prodens




# mol, functions, prim_matrix = read_wfn("./hr.wfn")

# wfn = calc_wfn(mol, functions, prim_matrix)
# np.savetxt("plane.csv", np.sum(np.square(wfn[16,...]), axis = 2))

# prodens = calc_prodens(mol)
# np.savetxt("prodens.csv", prodens[16,...])

# np.savetxt("deviation.csv", np.sum(np.square(wfn[16,...]), axis = 2) - prodens[16,...])

