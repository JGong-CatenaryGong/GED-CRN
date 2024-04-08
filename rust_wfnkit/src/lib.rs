use pyo3::prelude::*;
use std::collections::HashMap;
extern crate indicatif;
extern crate rayon;
use indicatif::ParallelProgressIterator;
use rayon::prelude::*;
use ndarray::{Array, Array2, Axis};
use log::{info, trace};
///use num_traits::exp;
/// use std::num;

const TYPE2EXP: [[i32; 3]; 35] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [2, 0, 0],
    [0, 2, 0],
    [0, 0, 2],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [3, 0, 0],
    [0, 3, 0],
    [0, 0, 3],
    [2, 1, 0],
    [2, 0, 1],
    [0, 2, 1],
    [1, 2, 0],
    [1, 0, 2],
    [0, 1, 2],
    [1, 1, 1],
    [0, 0, 4],
    [0, 1, 3],
    [0, 2, 2],
    [0, 3, 1],
    [0, 4, 0],
    [1, 0, 3],
    [1, 1, 2],
    [1, 2, 1],
    [1, 3, 0],
    [2, 0, 2],
    [2, 1, 1],
    [2, 2, 0],
    [3, 0, 1],
    [3, 1, 0],
    [4, 0, 0],
];

//const CELL_EXTEND: i8 = 2;

const PERIODIC_TABLE: [&str; 118] = ["H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og",];

fn atomic_number(symbol: &str) -> Option<i32> {
    let element_dict: HashMap<&str, i32> = [
        ("H", 1),("He", 2),("Li", 3),("Be", 4),("B", 5),("C", 6),("N", 7),("O", 8),("F", 9),("Ne", 10),("Na", 11),("Mg", 12),("Al", 13),("Si", 14),("P", 15),("S", 16),("Cl", 17),("Ar", 18),("K", 19),("Ca", 20),("Sc", 21),("Ti", 22),("V", 23),("Cr", 24),("Mn", 25),("Fe", 26),("Co", 27),("Ni", 28),("Cu", 29),("Zn", 30),("Ga", 31),("Ge", 32),("As", 33),("Se", 34),("Br", 35),("Kr", 36),("Rb", 37),("Sr", 38),("Y", 39),("Zr", 40),("Nb", 41),("Mo", 42),("Tc", 43),("Ru", 44),("Rh", 45),("Pd", 46),("Ag", 47),("Cd", 48),("In", 49),("Sn", 50),("Sb", 51),("Te", 52),("I", 53),("Xe", 54),("Cs", 55),("Ba", 56),("La", 57),("Ce", 58),("Pr", 59),("Nd", 60),("Pm", 61),("Sm", 62),("Eu", 63),("Gd", 64),("Tb", 65),("Dy", 66),("Ho", 67),("Er", 68),("Tm", 69),("Yb", 70),("Lu", 71),("Hf", 72),("Ta", 73),("W", 74),("Re", 75),("Os", 76),("Ir", 77),("Pt", 78),("Au", 79),("Hg", 80),("Tl", 81),("Pb", 82),("Bi", 83),("Po", 84),("At", 85),("Rn", 86),("Fr", 87),("Ra", 88),("Ac", 89),("Th", 90),("Pa", 91),("U", 92),("Np", 93),("Pu", 94),("Am", 95),("Cm", 96),("Bk", 97),("Cf", 98),("Es", 99),("Fm", 100),("Md", 101),("No", 102),("Lr", 103),("Rf", 104),("Db", 105),("Sg", 106),("Bh", 107),("Hs", 108),("Mt", 109),("Ds", 110),("Rg", 111),("Cn", 112),("Nh", 113),("Fl", 114),("Mc", 115),("Lv", 116),("Ts", 117),("Og", 118),
    ].iter().cloned().collect();

    return element_dict.get(symbol).cloned()
}

#[pyfunction]
fn convert_symbol_array(symbols: Vec<String>) -> Vec<Option<i32>> {
    let mut result: Vec<Option<i32>> = vec![Some(0); symbols.len()];
    for i in 0..symbols.len() {
        result[i] = atomic_number(symbols[i].as_str());
    }
    result
}

#[pyclass]
struct Molecule {
    atoms: Vec<i32>,
    coordinates: Vec<[f64; 3]>,
}

#[pymethods]
impl Molecule {
    #[new]
    fn new(atoms: Vec<i32>, coordinates: Vec<[f64; 3]>) -> Self {
        Self {atoms, coordinates}
    }

    fn get_atoms(&self) -> Vec<i32> {
        self.atoms.clone()
    }

    fn get_coordinates(&self) -> Vec<[f64; 3]> {
        self.coordinates.clone()
    }

    fn get_distance_matrix(&self) -> Vec<Vec<f64>> {
        let mut distance_matrix: Vec<Vec<f64>> = vec![vec![0.0; self.atoms.len()]; self.atoms.len()];
        for i in 0..self.atoms.len() {
            for j in 0..self.atoms.len() {
                let dx: f64 = self.coordinates[i][0] - self.coordinates[j][0];
                let dy: f64 = self.coordinates[i][1] - self.coordinates[j][1];
                let dz: f64 = self.coordinates[i][2] - self.coordinates[j][2];
                distance_matrix[i][j] = f64::sqrt(dx*dx + dy*dy + dz*dz);
            }
        }
        return distance_matrix;
    }

    fn get_repulsion_matrix(&self) -> Vec<Vec<f64>> {
        let distance_matrix: Vec<Vec<f64>> = self.get_distance_matrix();
        let atoms: Vec<i32> = self.atoms.clone();
        let mut repulsion_matrix: Vec<Vec<f64>> = vec![vec![0.0; self.atoms.len()]; self.atoms.len()];
        for i in 0..self.atoms.len() {
            for j in 0..self.atoms.len() {
                if i == j {
                    repulsion_matrix[i][j] = 0.0;
                    continue;
                }
                repulsion_matrix[i][j] = (atoms[i] * atoms[j]) as f64 / distance_matrix[i][j];
            }
        }
        return repulsion_matrix;
    }
}

#[pyclass]
struct GaussianFunc {
    center: i32,
    func_type: i32,
    exponent: f64
}

#[pymethods]
impl GaussianFunc {
    #[new]
    fn new(center: i32, func_type: i32, exponent: f64) -> Self {
        Self { center, func_type, exponent }
    }

    fn get_center(&self) -> i32 {
        self.center.clone()
    }

    fn get_func_type(&self) -> i32 {
        self.func_type.clone()
    }

    fn get_exponent(&self) -> f64 {
        self.exponent.clone()
    }
}

fn calc_gtf_value(mol: &Molecule, center: i32, func_type: i32, exponent: f64, coords: [f64; 3]) -> f64 {

    let exps: [i32; 3] = TYPE2EXP[(func_type - 1) as usize].clone();

    let expx: i32 = exps[0];
    let expy: i32 = exps[1];
    let expz: i32 = exps[2];

    let center_atom_coords: [f64; 3] = mol.get_coordinates()[(center - 1) as usize].clone();

    let center_x: f64 = center_atom_coords[0];
    let center_y: f64 = center_atom_coords[1];
    let center_z: f64 = center_atom_coords[2];

    let rr: f64 = (coords[0] - center_x).powi(2) + (coords[1] - center_y).powi(2) + (coords[2] - center_z).powi(2);

    let exp_term: f64 = (-exponent * rr).exp();

    let gtf_val: f64 = (coords[0] - center_x).powi(expx) * (coords[1] - center_y).powi(expy) * (coords[2] - center_z).powi(expz) * exp_term;

    return gtf_val
}

fn calc_wfn_vals_of_mo(mol: &Molecule, centers: Vec<i32>, func_types: Vec<i32>, exponents: Vec<f64>, coords: [f64; 3], prims_array: Array2<f64>) -> Vec<f64> {

    let gtf_vals: Vec<f64> = (0..centers.len()).into_iter()
        .map(|i| calc_gtf_value(mol, centers[i], func_types[i], exponents[i], coords))
        .collect();

    let gtf_array = Array::from(gtf_vals);

    let wfn_vals = prims_array.dot(&gtf_array).to_vec();
    return wfn_vals
}

/// Threaded version
#[pyfunction]
fn calc_wfn_grids(mol: &Molecule, centers: Vec<i32>, func_types: Vec<i32>, exponents: Vec<f64>, grids: Vec<[f64; 3]>, prims: &PrimMat) -> Vec<Vec<f64>> {


    let flatten_prims = prims.get_values().clone();
    let shape: (usize, usize) = (prims.get_n_mos() as usize, prims.get_n_gtfs() as usize);
    let prim_array = Array::from_shape_vec(shape, flatten_prims).ok().unwrap();

    info!("Shape of prim matrix: {:?}", prim_array.shape());

    let wfns: Vec<Vec<f64>> = grids.par_iter()
        .progress_count(grids.len() as u64)
        .map(|g| calc_wfn_vals_of_mo(mol, centers.clone(), func_types.clone(), exponents.clone(), g.clone(), prim_array.clone()))
        .collect();
    return wfns
}

fn calc_dens_vals_of_mo(mol: &Molecule, centers: Vec<i32>, func_types: Vec<i32>, exponents: Vec<f64>, coords: [f64; 3], prims_array: Array2<f64>, occs: Vec<f64>) -> Vec<f64> {


    let gtf_vals: Vec<f64> = centers.iter().zip(func_types.iter()).zip(exponents.iter())
        .map(|((c, f), e)| calc_gtf_value(mol, *c, *f, *e, coords))
        .collect();

    // let gtf_vals: Vec<f64> = (0..centers.len()).into_iter()
    //     .map(|i| calc_gtf_value(mol, centers[i], func_types[i], exponents[i], coords))
    //     .collect();

    let gtf_array = Array::from(gtf_vals);

    let dens_vals = prims_array.dot(&gtf_array).to_vec().into_iter().zip(occs.iter())
        .map(|(x, y)| x.powi(2) * y).collect::<Vec<f64>>();

    //let dens_vals = (0..wfn_vals.len() as usize).into_iter().map(|i| (occs[i] * wfn_vals[i]).powi(2)).collect();

    return dens_vals
}

#[pyfunction]
fn calc_dens_grids(mol: &Molecule, centers: Vec<i32>, func_types: Vec<i32>, exponents: Vec<f64>, grids: Vec<[f64; 3]>, prims: &PrimMat, occs: Vec<f64>) -> Vec<f64> {


    let flatten_prims = prims.get_values().clone();
    let shape: (usize, usize) = (prims.get_n_mos() as usize, prims.get_n_gtfs() as usize);
    let prim_array = Array::from_shape_vec(shape, flatten_prims).ok().unwrap();

    info!("Shape of prim matrix: {:?}", prim_array.shape());

    let dens: Vec<f64> = grids.par_iter()
        .progress_count(grids.len() as u64)
        .map(|g| {
            calc_dens_vals_of_mo(mol, centers.clone(), func_types.clone(), exponents.clone(), g.clone(), prim_array.clone(), occs.clone()).iter().sum()
        })
        .collect();

    let int_dens: f64 = dens.iter().sum();
    info!("The integration of electron density: {:?}", int_dens);

    return dens
}

#[pyclass]
struct PrimMat {
    values: Vec<f64>,
    n_mos: i32,
    n_gtfs: i32,
}

#[pymethods]
impl PrimMat {
    #[new]
    fn new(values: Vec<f64>, n_mos: i32, n_gtfs: i32) -> Self {
        PrimMat { values, n_mos, n_gtfs }
    }

    fn get_values(&self) -> Vec<f64> {
        self.values.clone()
    }

    fn get_n_mos(&self) -> i32 {
        self.n_mos.clone()
    }

    fn get_n_gtfs(&self) -> i32 {
        self.n_gtfs.clone()
    }
}

#[pyclass]
struct Wavefunction {
    centers: [Vec<i32>; 118],
    types: [Vec<i32>; 118],
    exponents: [Vec<f64>; 118],
    prims: [Vec<Vec<f64>>; 118],
    occs: [Vec<f64>; 118],
}

#[pymethods]
impl Wavefunction {
    #[new]
    fn new(centers: [Vec<i32>; 118], types: [Vec<i32>; 118], exponents:[Vec<f64>; 118], prims: [Vec<Vec<f64>>; 118], occs: [Vec<f64>; 118]) -> Self {
        Wavefunction { centers, types, exponents, prims, occs }
    }

    fn get_centers(&self) -> [Vec<i32>; 118] {
        self.centers.clone()
    }

    fn get_types(&self) -> [Vec<i32>; 118] {
        self.types.clone()
    }

    fn get_exponents(&self) -> [Vec<f64>; 118] {
        self.exponents.clone()
    }

    fn get_prims(&self) -> [Vec<Vec<f64>>; 118] {
        self.prims.clone()
    }

    // fn get_prim_instance(&self) -> [PrimMat; 118] {
    //     let mats: [PrimMat; 118] = self.prims.clone().iter()
    //     .map(|p| {
    //         let n_mo: i32 = p.len() as i32; 
    //         let n_gtf: i32 = p[0].len() as i32;
    //         PrimMat::new(p.iter().flatten().cloned().collect(), n_mo, n_gtf)
    //     }).collect();
    //     mats
    // }
    
    fn get_occs(&self) -> [Vec<f64>; 118] {
        self.occs.clone()
    }
}

#[pyfunction]
fn calc_pro_dens(mol: &Molecule, wfn_list: &Wavefunction, grids: Vec<[f64; 3]>) -> Vec<f64> {
    
    let atoms = mol.atoms.clone();
    let dots = mol.coordinates.clone();
    let centers = wfn_list.centers.clone();
    info!("There are {:?} wfns stored.", centers.len());
    let types = wfn_list.types.clone();
    let exponents = wfn_list.exponents.clone();
    let prims = wfn_list.prims.clone();
    let occss = wfn_list.occs.clone();
    let mut pro_dens: Vec<f64> = vec![0.0; grids.len()];
    let mut total_electrons: f64 = 0.0;
    for i in 0..atoms.len() {
        let idx: usize = (atoms[i] - 1) as usize;
        
        let atom_centers = centers[idx].clone();
        let atom_types = types[idx].clone();
        let atom_exponents = exponents[idx].clone();
        let atom_prims = prims[idx].clone();

        let primat = PrimMat::new(atom_prims.iter().flatten().cloned().collect(), atom_prims.len() as i32, atom_prims[0].len() as i32);

        let atom_mol = Molecule::new(vec![atoms[i]], vec![dots[i]]);

        let atom_occs = occss[idx].clone();

        info!("Calculating atomic wavefunction for atom {:?} at {:?}", PERIODIC_TABLE[(atom_mol.get_atoms()[0] - 1) as usize], atom_mol.get_coordinates());
        info!("{:?} GTFs found in the wfn", atom_centers.len());
        trace!("The occupytion: {:?}", atom_occs);

        let n_electrons:f64 = atom_occs.iter().sum();
        total_electrons += n_electrons;

        trace!("Atom electrons {:?}", n_electrons);
        trace!("The size of grids: {}", grids.len());
        let atom_dens = calc_dens_grids(&atom_mol, atom_centers, atom_types, atom_exponents, grids.clone(), &primat, atom_occs);
        info!("Finished! {}", atom_dens.len());
        for j in 0..grids.len() {
            pro_dens[j] += atom_dens[j]
        }
    };
    trace!("Total electrons: {:?}", total_electrons);
    return pro_dens;
}

#[pyfunction]
fn calc_pro_dens2(mol: &Molecule, wfn_list: &Wavefunction, grids: Vec<[f64; 3]>) -> Vec<f64> {
    let atoms = mol.atoms.clone();
    let dots = mol.coordinates.clone();

    info!("Calculating the promolecular density for the molecule with {:?}", atoms.len());

    let pro_dens_before: Vec<f64> = atoms.clone().into_iter().zip(dots.into_iter())
        .map(|(atom, dot)| {
            let idx: usize = (atom - 1) as usize;
            
            let center = wfn_list.centers[idx].clone();
            let typ = wfn_list.types[idx].clone();
            let exp = wfn_list.exponents[idx].clone();
            let prim = wfn_list.prims[idx].clone();
            let occ = wfn_list.occs[idx].clone();

            let primat = PrimMat::new(prim.iter().flatten().cloned().collect(), prim.len() as i32, prim[0].len() as i32);

            let fake_mol = Molecule::new(vec![atom], vec![dot]);

            let atom_grids: Vec<f64> = calc_dens_grids(&fake_mol, center, typ, exp, grids.clone(), &primat, occ);
            return atom_grids;
        }).flatten().collect();

    let shape: (usize, usize) = (atoms.len() as usize, grids.len() as usize);

    Array::from_shape_vec(shape, pro_dens_before).ok().unwrap().sum_axis(Axis(0)).to_vec()
}


#[pyfunction]
fn calc_potential(mol: &Molecule, grids: Vec<[f64; 3]>) -> Vec<f64> {

    info!("Calculating nuclei potential map...");

    let mut pot = vec![0.0; grids.len()];
    let atoms = mol.atoms.clone();
    let dots = mol.coordinates.clone();

    for i in 0..atoms.len() {
        let ax = dots[i][0];
        let ay = dots[i][1];
        let az = dots[i][2];

        let atom_pot: Vec<f64> = grids.par_iter()
            .map(|g| atoms[i] as f64/((g[0] - ax).powi(2) + (g[1] - ay).powi(2) + (g[2] - az).powi(2)).sqrt()).collect();

        for i in 0..atom_pot.len() {
            pot[i] += atom_pot[i];
        }
    }

    return pot
}

fn fn_calc_atom_vne(atom_number: i32, atom_coords: [f64; 3], rho: Vec<f64>, grids: Vec<[f64; 3]>) -> f64 {
    let v_mat: f64 = (0..grids.len()).into_par_iter()
        .map(|i| atom_number as f64 * rho[i] / ((grids[i][0] - atom_coords[0]).powi(2) + (grids[i][1] - atom_coords[1]).powi(2) + (grids[i][2] - atom_coords[2]).powi(2)).sqrt()).sum();
    return v_mat;
}

#[pyfunction]
fn calc_vne(mol: &Molecule, rho: Vec<f64>, grids: Vec<[f64; 3]>) -> f64 {
    let atoms = mol.atoms.clone();
    let dots = mol.coordinates.clone();

    let vne: f64 = (0..atoms.len()).into_par_iter()
        .map(|i| fn_calc_atom_vne(atoms[i], dots[i], rho.clone(), grids.clone())).sum();
    return vne;
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust_wfnkit(_py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();

    m.add_function(wrap_pyfunction!(convert_symbol_array, m)?)?;
    m.add_class::<Molecule>()?;
    m.add_class::<PrimMat>()?;
    m.add_class::<GaussianFunc>()?;
    m.add_class::<Wavefunction>()?;
    m.add_function(wrap_pyfunction!(calc_wfn_grids, m)?)?;
    m.add_function(wrap_pyfunction!(calc_dens_grids, m)?)?;
    m.add_function(wrap_pyfunction!(calc_pro_dens, m)?)?;
    m.add_function(wrap_pyfunction!(calc_pro_dens2, m)?)?;
    m.add_function(wrap_pyfunction!(calc_potential, m)?)?;
    m.add_function(wrap_pyfunction!(calc_vne, m)?)?;
    Ok(())
}
