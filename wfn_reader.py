import numpy as np
import rust_wfnkit
import re

import numpy as np
import time
import os, glob
import h5py

def read_wfn(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    print("===pyWFNreader by JUN (02/25/2024)===")

    title_line = lines[0]
    basic_info = lines[1].split()

    n_omo = int(basic_info[1])
    n_gfunc = int(basic_info[4])
    n_atoms = int(basic_info[6])

    print(f"Basic information: {n_omo} occupied MOs, {n_gfunc} Gaussian functions, {n_atoms} atoms")

    atom_lines = lines[2:2+n_atoms]

    atom_array = [None] * n_atoms
    atom_coords = [None] * n_atoms
    
    for i in range(n_atoms):
        atom_info = atom_lines[i].split()
        #print(atom_info)
        if len(atom_info[0]) >= 3:
            atom_array[i] = atom_info[0][:2]
        else:
            atom_array[i] = atom_info[0]
        atom_coords[i] = np.float64(np.array(re.findall(r"-?[0-9|.]{5,}", atom_lines[i])))

    atom_array = rust_wfnkit.convert_symbol_array(atom_array)

    print(atom_coords)
    mol = rust_wfnkit.Molecule(atom_array, atom_coords) # Create a Molecule object

    center_lines = []
    type_lines = []
    exponent_lines = []
    MO_lines = []
    energy_lines = []
    other_lines = []

    for line in lines[2+n_atoms:]:
        if line.startswith('CENTRE'):
            center_lines.append(line)
        elif line.startswith('TYPE'):
            type_lines.append(line)
        elif line.startswith('EXPONENTS'):
            exponent_lines.append(line)
        elif line.startswith('MO') or re.search(r"[-]?[0-9][.][0-9]+[E|D][+|-][0-9]{2}", line):
            MO_lines.append(line)
        elif line.startswith(' THE '):
            energy_lines.append(line)
        else:
            other_lines.append(line)

    print(f"There are {len(center_lines)} lines for centers, processing ...")
    center_assignment = []
    for center_line in center_lines:
        for matches in re.findall(r"S\s\s|[\s|0-9][\s|0-9][0-9]", center_line):
            if matches.endswith(" "):
                pass
            else:
                center_assignment.append(int(matches))

    print(f"There are {len(type_lines)} lines for types, processing ...")
    type_assignment = []
    for type_line in type_lines:
        for matches in re.findall(r"[0-9]+", type_line):
            if matches.endswith(" "):
                pass
            else:
                type_assignment.append(int(matches))

    print(f"There are {len(exponent_lines)} lines for exponents, processing ...")
    exponents = []
    for exponent_line in exponent_lines:
        for matches in re.findall(r"[0-9][.][0-9]+[D|E][+|-][0-9]{2}", exponent_line):
            exponents.append(float(matches.replace('D', 'E')))

    print(f"There are {len(exponents)} exponents.")

    assert len(center_assignment) == len(type_assignment) == len(exponents) == n_gfunc, "The number of centers, types, exponents and gaussian functions are not equal."

    occ_MOs = [None] * n_omo
    MO_energy = [None] * n_omo
    prim_matrix = np.zeros((n_omo, n_gfunc))

    prims = []
    line_no = 0
    for line in MO_lines:
        if line.startswith("MO"):
            matches = re.findall(r"[+|-]?[0-9]+[.][0-9]+", line)
            occ_MOs[line_no] = float(matches[-2])
            MO_energy[line_no] = float(matches[-1])
            line_no += 1
        else:
            for matches in re.findall(r"[-]?[0-9][.][0-9]+[E|D][+|-][0-9]{2}", line):
                #print(matches)
                prims.append(float(matches.replace('D', 'E')))

    # print(f"Occupitioons: {occ_MOs}")
    # print(f"Orbital energy: {MO_energy}")

    print(f"There are {len(prims)} primitives from {n_gfunc} functions on {n_omo} MOs.")
    assert len(prims) == n_gfunc * n_omo, "The number of primitives and MOs are not matched."

    prim_matrix = np.array(prims).reshape((n_omo, n_gfunc)).T

    functions = [None] * n_gfunc
    for i in range(n_gfunc):
        functions[i] = rust_wfnkit.GaussianFunc(center_assignment[i], type_assignment[i], exponents[i])

    print(f"{n_gfunc} Gaussian functions generated.")

    # vir_energy = []
    # print(energy_lines)
    # for matches in re.findall(r"[-]?[0-9][.][0-9]+", energy_lines[0]):
    #     vir_energy.append(float(matches))

    # virial = vir_energy[0]
    # total_energy = vir_energy[1]

    return mol, functions, prim_matrix, occ_MOs

# mol, functions, prim_matrix = read_wfn("hr.wfn")
# print(prim_matrix.shape)

# dir = './atomic_wfns/'

# suffix = '*.wfn'

# files = glob.glob(os.path.join(dir, suffix))

# for file in files:
#     mol, functions, prim_matrix = read_wfn(file)

#     center_list = [f.get_center() for f in functions]
#     type_list = [f.get_func_type() for f in functions]
#     exp_list = [f.get_exponent() for f in functions]
    
#     dic = {
#         "mol_atoms": mol.get_atoms(),
#         "mol_geom": mol.get_coordinates(),
#         "func_centers": center_list,
#         "func_types": type_list,
#         "func_exps": exp_list,
#         "prims": prim_matrix
#     }

#     with h5py.File(file.replace(".wfn", ".h5"), "w") as f:
#         for key, val in dic.items():
#             f.create_dataset(key, data = np.array(val))