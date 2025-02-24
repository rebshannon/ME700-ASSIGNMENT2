# ME700 Assignment 2

[![python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
![os](https://img.shields.io/badge/os-ubuntu%20|%20macos%20|%20windows-blue.svg)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/sandialabs/sibl#license)

[![codecov](https://codecov.io/gh/rebshannon/ME700-ASSIGNMENT2/graph/badge.svg?token=d5QRPml5Wg)](https://codecov.io/gh/rebshannon/ME700-ASSIGNMENT2)
[![tests](https://github.com/rebshannon/ME700-ASSIGNMENT2/actions/workflows/test.yml/badge.svg)](https://github.com/rebshannon/ME700-ASSIGNMENT2/actions)

## Table of Contents

* [Set Up](#setup)
* [Instructions](#inst)
* [Tutorial](#tutorial)

## Set Up <a name=setup></a>
Set up the conda environment and test that the code is functioning. Note: The environemnt name for all parts of Assignemnt 1 refers to the bisection method.  

1. Create a new conda environment and activate it.  
    ```bash 
    conda create --name A2-matrix_struct_analysis python=3.12
    ```
    ```bash
    conda activate A2-matrix_struct_analysis
    ``` 
2. Confirm that the right version of Python and pip setuptools are being used. Python should be version 3.12; the second command will update setuptools if necessary.  
    ```bash
    python --version
    ```
    ```bash
    pip install --upgrade pip setuptools wheel
    ```
3. Make sure you are in the ME700-Assignment1 directory.  
4. Install an editable version of the code. Note that the *pyproject.toml* file is required for this.  
    ```bash
    pip install -e .
    ```
5. Test the code with pytest. This command returns code coverage for all files in the directory. Coverage for each of the solvers should be 100% and all tests should pass.  
    ```bash
    pytest -v --cov=src  --cov-report term-missing
    ```

## Instructions <a name=inst></a>

The matrix structural analysis solver solves for nodal displacement and reaction forces for a provided frame and applied load. 

### Inputs

#### Node Class

Defines geometry and loading related to each node. Note that the nodes are defined by their row index, so the corrdinate, load, and type for one node should be entered in the same row for each variable.

`nodes` -  (x, y, z) coordinates of each node; should be an N x 3 array where N is the number of nodes

`load` - the applied load at each node, should be an N x 6 array where N is the number of nodes; each column is defined: Fx, Fy, Fz, Mx, My, and Mz

`nodeType` - the type of boundary condition for each node, should be an N x 1 array; node types are as follows:  
    0 = free node  
    1 = fixed node  
    2 = pinned node  

#### Element Class

Defines geometry and material properties related to each element. Like the nodes, each element is identified by its row index.

`connections` - which nodes are connected to make an element, nodes are numbered by their index in the `nodes` array, should be an M by 2 array where M is the number of elements

Material Properties - define the material properties for each element, each should be a 1 by M array

`E` - Young'e modulus  
`nu` - Poisson's ratio  
`A` - Cross-sectional area  
`Iz` - Moment of inertia about z-axis  
`Iy` -Moment of inertia about y-axis  
`Ip` - Polar moment of inertia 
`J` - Torsional constant

Returns the unknown nodal displacements and the unknown reaction forces.  
Note: The node/element order may be messed up on the result.

## Tutorial <a name=tutorial></a>

The tutorial for this solver can be accessed with the following Jupyter notebook.

```bash
pip install jupyter
```
```bash
cd tutorials/
```
```bash
jupyter notebook tut_MSA.ipynb
```