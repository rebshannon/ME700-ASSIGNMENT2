import numpy as np
import math
import pytest
import src.matrixStructAnalysis as MSA

from matrixStructAnalysis import Nodes 
from matrixStructAnalysis import Elements 

@pytest.fixture
def NC():
    nodes = np.array([[0, 0, 10],[15, 0, 10],[15, 0, 0]])
    load = np.array([[0,0,0,0,0,0],[0.1,0.05,-0.07,0.05,-0.1,0.25],[0,0,0,0,0,0]])
    #nodeType = np.array([1,0])
    BC = np.array([[0,0,0,0,0,0],[1,1,1,1,1,1],[0,0,0,1,1,1]])
    return Nodes(nodes=nodes,load=load,BC=BC)

@pytest.fixture
def EC():
    connections = np.array([[0, 1],[1,2]])

    b = 0.5
    h = 1

    E = np.array([1000,1000])
    nu = np.array([0.3,0.3])
    A = np.array([b*h,b*h])
    Iz = np.array([b*h**3/12,b*h**3/12])
    Iy = np.array([h*b**3/12,h*b**3/12])
    Ip = np.array([b*h/(12*(b**2+h**2)),b*h/(12*(b**2+h**2))])
    J = np.array([0.02861,0.02861])

    local_z = np.array([[0,0,1],[1,0,0]])


    return Elements(connections,E,nu,A,Iz,Iy,Ip,J,local_z)

def test_find_global_frame_stiffness(NC,EC):
    found = MSA.find_global_frame_stiffness(NC,EC)
    known = np.array([
      [ 3.33333333e+01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00,-3.33333333e+01, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00],
      [ 0.00000000e+00, 1.48148148e-01, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 1.11111111e+00, 0.00000000e+00,-1.48148148e-01,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.11111111e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00],
      [ 0.00000000e+00, 0.00000000e+00, 3.70370370e-02, 0.00000000e+00,
        -2.77777778e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        -3.70370370e-02, 0.00000000e+00,-2.77777778e-01, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00],
      [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 7.33589744e-01,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00,-7.33589744e-01, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00],
      [ 0.00000000e+00, 0.00000000e+00,-2.77777778e-01, 0.00000000e+00,
        2.77777778e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        2.77777778e-01, 0.00000000e+00, 1.38888889e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00],
      [ 0.00000000e+00, 1.11111111e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 1.11111111e+01, 0.00000000e+00,-1.11111111e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 5.55555556e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00],
      [-3.33333333e+01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 3.34583333e+01, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00,-6.25000000e-01, 0.00000000e+00,
        -1.25000000e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        -6.25000000e-01, 0.00000000e+00],
      [ 0.00000000e+00,-1.48148148e-01, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00,-1.11111111e+00, 0.00000000e+00, 6.48148148e-01,
        0.00000000e+00, 2.50000000e+00, 0.00000000e+00,-1.11111111e+00,
        0.00000000e+00,-5.00000000e-01, 0.00000000e+00, 2.50000000e+00,
        0.00000000e+00, 0.00000000e+00],
      [ 0.00000000e+00, 0.00000000e+00,-3.70370370e-02, 0.00000000e+00,
        2.77777778e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        5.00370370e+01, 0.00000000e+00, 2.77777778e-01, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00,-5.00000000e+01, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00],
      [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,-7.33589744e-01,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.50000000e+00,
        0.00000000e+00, 1.74002564e+01, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00,-2.50000000e+00, 0.00000000e+00, 8.33333333e+00,
        0.00000000e+00, 0.00000000e+00],
      [ 0.00000000e+00, 0.00000000e+00,-2.77777778e-01, 0.00000000e+00,
        1.38888889e+00, 0.00000000e+00,-6.25000000e-01, 0.00000000e+00,
        2.77777778e-01, 0.00000000e+00, 6.94444444e+00, 0.00000000e+00,
        6.25000000e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        2.08333333e+00, 0.00000000e+00],
      [ 0.00000000e+00, 1.11111111e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 5.55555556e+00, 0.00000000e+00,-1.11111111e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.22114957e+01,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00,-1.10038462e+00],
      [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00,-1.25000000e-01, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 6.25000000e-01, 0.00000000e+00,
        1.25000000e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        6.25000000e-01, 0.00000000e+00],
      [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,-5.00000000e-01,
        0.00000000e+00,-2.50000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 5.00000000e-01, 0.00000000e+00,-2.50000000e+00,
        0.00000000e+00, 0.00000000e+00],
      [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        -5.00000000e+01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 5.00000000e+01, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00],
      [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.50000000e+00,
        0.00000000e+00, 8.33333333e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00,-2.50000000e+00, 0.00000000e+00, 1.66666667e+01,
        0.00000000e+00, 0.00000000e+00],
      [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00,-6.25000000e-01, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 2.08333333e+00, 0.00000000e+00,
        6.25000000e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        4.16666667e+00, 0.00000000e+00],
      [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,-1.10038462e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 1.10038462e+00]])
    assert np.allclose(known, found)

def test_partition_matrices(NC,EC):
    K_global = np.array([
      [ 3.33333333e+01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00,-3.33333333e+01, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00],
      [ 0.00000000e+00, 1.48148148e-01, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 1.11111111e+00, 0.00000000e+00,-1.48148148e-01,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.11111111e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00],
      [ 0.00000000e+00, 0.00000000e+00, 3.70370370e-02, 0.00000000e+00,
        -2.77777778e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        -3.70370370e-02, 0.00000000e+00,-2.77777778e-01, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00],
      [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 7.33589744e-01,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00,-7.33589744e-01, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00],
      [ 0.00000000e+00, 0.00000000e+00,-2.77777778e-01, 0.00000000e+00,
        2.77777778e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        2.77777778e-01, 0.00000000e+00, 1.38888889e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00],
      [ 0.00000000e+00, 1.11111111e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 1.11111111e+01, 0.00000000e+00,-1.11111111e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 5.55555556e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00],
      [-3.33333333e+01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 3.34583333e+01, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00,-6.25000000e-01, 0.00000000e+00,
        -1.25000000e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        -6.25000000e-01, 0.00000000e+00],
      [ 0.00000000e+00,-1.48148148e-01, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00,-1.11111111e+00, 0.00000000e+00, 6.48148148e-01,
        0.00000000e+00, 2.50000000e+00, 0.00000000e+00,-1.11111111e+00,
        0.00000000e+00,-5.00000000e-01, 0.00000000e+00, 2.50000000e+00,
        0.00000000e+00, 0.00000000e+00],
      [ 0.00000000e+00, 0.00000000e+00,-3.70370370e-02, 0.00000000e+00,
        2.77777778e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        5.00370370e+01, 0.00000000e+00, 2.77777778e-01, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00,-5.00000000e+01, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00],
      [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,-7.33589744e-01,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.50000000e+00,
        0.00000000e+00, 1.74002564e+01, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00,-2.50000000e+00, 0.00000000e+00, 8.33333333e+00,
        0.00000000e+00, 0.00000000e+00],
      [ 0.00000000e+00, 0.00000000e+00,-2.77777778e-01, 0.00000000e+00,
        1.38888889e+00, 0.00000000e+00,-6.25000000e-01, 0.00000000e+00,
        2.77777778e-01, 0.00000000e+00, 6.94444444e+00, 0.00000000e+00,
        6.25000000e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        2.08333333e+00, 0.00000000e+00],
      [ 0.00000000e+00, 1.11111111e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 5.55555556e+00, 0.00000000e+00,-1.11111111e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.22114957e+01,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00,-1.10038462e+00],
      [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00,-1.25000000e-01, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 6.25000000e-01, 0.00000000e+00,
        1.25000000e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        6.25000000e-01, 0.00000000e+00],
      [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,-5.00000000e-01,
        0.00000000e+00,-2.50000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 5.00000000e-01, 0.00000000e+00,-2.50000000e+00,
        0.00000000e+00, 0.00000000e+00],
      [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        -5.00000000e+01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 5.00000000e+01, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00],
      [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.50000000e+00,
        0.00000000e+00, 8.33333333e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00,-2.50000000e+00, 0.00000000e+00, 1.66666667e+01,
        0.00000000e+00, 0.00000000e+00],
      [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00,-6.25000000e-01, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 2.08333333e+00, 0.00000000e+00,
        6.25000000e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        4.16666667e+00, 0.00000000e+00],
      [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,-1.10038462e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 1.10038462e+00]])

    found_partK_DOF, found_partK_force, foundF, foundForceInd,founddofInd = MSA.partition_matrices(NC,EC,K_global)

    knownF = np.reshape(np.array([0.1, 0.05, -0.07, 0.05, -0.1, 0.25, 0, 0, 0]),(9,1))

    known_partK_DOF = np.array([
      [33.45833333, 0.,  0.,  0., -0.625, 0., 0., -0.625, 0.],
      [ 0.,  0.64814815,  0.,  2.5, 0., -1.11111111, 2.5, 0., 0.],
      [ 0.,  0., 50.03703704,  0.,  0.27777778,  0., 0.,  0.,  0.],
      [ 0.,  2.5, 0., 17.40025641, 0.,  0., 8.33333333,  0.,  0.],
      [-0.625, 0.,  0.27777778,  0.,  6.94444444,  0., 0.,  2.08333333,  0.],
      [ 0., -1.11111111,  0.,  0.,  0., 12.21149573, 0.,  0., -1.10038462],
      [ 0.,  2.5, 0.,  8.33333333,  0.,  0., 16.66666667,  0.,  0.],
      [-0.625, 0.,  0.,  0.,  2.08333333,  0., 0.,  4.16666667,  0.],
      [ 0.,  0.,  0.,  0.,  0., -1.10038462, 0.,  0.,  1.10038462]])

    known_partK_force = np.array([
      [-3.33333333e+01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00],
      [ 0.00000000e+00, -1.48148148e-01, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 1.11111111e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00],
      [ 0.00000000e+00, 0.00000000e+00, -3.70370370e-02, 0.00000000e+00,
        -2.77777778e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00],
      [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -7.33589744e-01,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00],
      [ 0.00000000e+00, 0.00000000e+00, 2.77777778e-01, 0.00000000e+00,
        1.38888889e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00],
      [ 0.00000000e+00, -1.11111111e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 5.55555556e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00],
      [-1.25000000e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        6.25000000e-01, 0.00000000e+00, 0.00000000e+00, 6.25000000e-01,
        0.00000000e+00],
      [ 0.00000000e+00, -5.00000000e-01, 0.00000000e+00, -2.50000000e+00,
        0.00000000e+00, 0.00000000e+00, -2.50000000e+00, 0.00000000e+00,
        0.00000000e+00],
      [ 0.00000000e+00, 0.00000000e+00, -5.00000000e+01, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00]])                    

    known_forceInd = [6,7,8,9,10,11,15,16,17]
    known_dofInd = [0,1,2,3,4,5,12,13,14]

    assert np.allclose(known_partK_DOF, found_partK_DOF)
    assert np.allclose(known_partK_force, found_partK_force)
    assert np.allclose(knownF, foundF)
    assert np.allclose(known_forceInd, foundForceInd)
    assert np.allclose(known_dofInd, founddofInd)

def test_run_MSA_solver(NC,EC):
    found_d,found_f = MSA.run_MSA_solver(NC,EC)
    known_f = np.reshape(np.array([0,-0.09468332,1,-0.03420124,2,0.00469541,3,0.1079876,4,-0.02359799,5,-0.76301861,12,-0.00531668,13,-0.01579876,14,0.06530459]),(9,2))
    known_d = np.reshape(np.array([6,2.84049953e-3,7,1.59843349,8,-1.30609178e-3,9,-1.47204342e-1,10,-1.67293339e-2,11,1.82343348e-1,15,-0.16616285,16,0.00879074,17,0.18234335]),(9,2))

    assert np.allclose(known_f, found_f)
    assert np.allclose(known_d, found_d)
