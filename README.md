# GED-CRN
A convolutional residual network for generation of electron density.
Coded within Python and Rust.

## Core dependencies
- Python 3.11
- Torch
- Numpy
- maturin
- memory_profiler

## Install the Rust package as a Python module
Create a virtual environment and install the dependencies.

Under the `rust_wfnkit` directory, run the following command:
```
    maturin develop
```

## Run the training
Run the training script.

Under the root directory, run the following command:
```
    python train.py
```

## Run the prediction
The script `props_test.py` contains a suite of prediction functions.

## Directory structure
- atomic_wfns: spherical symmetric wavefunctions for atoms in .wfn format
- training_wfns: wavefunctions for training molecules in .wfn format
- wfn_for_props\results: prediction results
- saved_models: trained models

## Global constant
The global variables were set in the file `global_constant.py`:
- RESOLUTION: the number of sampled grids in one dimension per bohr
- ELEMENTS: the periodic table of elements
- BATCH_SIZE: the number of cubes per batch in training process 
- BOX_SIZE: the size of the cubic box in number of grids in one dimension