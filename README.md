# ME700 Assignment 2

[![python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
![os](https://img.shields.io/badge/os-ubuntu%20|%20macos%20|%20windows-blue.svg)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/sandialabs/sibl#license)

[![codecov](https://codecov.io/gh/rebshannon/ME700-ASSIGNMENT1/graph/badge.svg?token=S3ZCVIAW6K)](https://codecov.io/gh/rebshannon/ME700-ASSIGNMENT1)
[![tests](https://github.com/rebshannon/ME700-ASSIGNMENT1/actions/workflows/test.yml/badge.svg)](https://github.com/rebshannon/ME700-ASSIGNMENT1/actions)

## Table of Contents

* [Set Up](#setup)
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