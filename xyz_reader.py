import numpy as np
import rust_wfnkit
import re

def read_xyz(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    print(lines)

    atoms = []
    coords = []
    for line in lines[2:]:
        atoms.append(line.split()[0])
        coords.append([float(m) / 0.52918 for m in re.findall(r"-?[0-9|.]{5,}", line)])

    atoms = rust_wfnkit.convert_symbol_array(atoms)
    mol = rust_wfnkit.Molecule(atoms, coords)

    return(mol)

def read_pdb(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    atoms = []
    coords = []

    resnames = []
    res_seq = []

    for line in lines:
        if line.startswith('CRYST1'):
            #print(line)
            a, b, c, alpha, beta, gamma = [float(m) for m in re.findall(r"[0-9|.]{3,}", line)]
        elif line.startswith('SCALE1'):
            s11, s12, s13, offset_a = [float(m) for m in re.findall(r"[0-9|.]{3,}", line)]
        elif line.startswith('SCALE2'):
            s21, s22, s23, offset_b = [float(m) for m in re.findall(r"[0-9|.]{3,}", line)]
        elif line.startswith('SCALE3'):
            s31, s32, s33, offset_c = [float(m) for m in re.findall(r"[0-9|.]{3,}", line)]
        elif line.startswith('HETATM'):
            serial = int(line[6:11])
            symbol = line[12:14].strip()
            atoms.append(symbol)
            resnames.append(line[17:20].strip())
            res_seq.append(int(line[22:26]))

            x_coord = float(line[30:38]) / 0.52918
            y_coord = float(line[38:46]) / 0.52918
            z_coord = float(line[46:54]) / 0.52918
            coords.append([x_coord, y_coord, z_coord])

            #print(x_coord, y_coord, z_coord)

    atoms = rust_wfnkit.convert_symbol_array(atoms)
    mol = rust_wfnkit.Molecule(atoms, coords)

    s1_vec = np.array([s11, s12, s13])
    s2_vec = np.array([s21, s22, s23])
    s3_vec = np.array([s31, s32, s33])
    #print(a_vec, b_vec, c_vec)


    crystal_info = [a, b, c, alpha, beta, gamma]

    return mol, crystal_info, s1_vec, s2_vec, s3_vec


#read_pdb('PBO.pdb')


