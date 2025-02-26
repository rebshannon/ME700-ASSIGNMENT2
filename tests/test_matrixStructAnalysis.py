import numpy as np
import math
import pytest
import src.matrixStructAnalysis as MSA

from matrixStructAnalysis import Nodes 
from matrixStructAnalysis import Elements 

@pytest.fixture
def NC():
    nodes = np.array([[8e3, 0, 0],[13e3, 0, 0]])
    load = np.array([[0,0,0,0,0,0],[5/math.sqrt(2),-5/math.sqrt(2),0,0,0,0]])
    #nodeType = np.array([1,0])
    BC = np.zeros((2,6))
    BC[1,:] = 1
    return Nodes(nodes=nodes,load=load,BC=BC)

@pytest.fixture
def EC():
    connections = np.array([[0, 1]])

    E = np.array([200])
    nu = np.array([0.3])
    A = np.array([6e3])
    Iz = np.array([200e6])
    Iy = np.array([200e6])
    Ip = np.array([10])
    J = np.array([300e3])

    return Elements(connections,E,nu,A,Iz,Iy,Ip,J)

def test_find_global_frame_stiffness(NC,EC):
    found = MSA.find_global_frame_stiffness(NC,EC)
    known = np.array([[ 2.40000000e+02,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        -2.40000000e+02,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [ 0.00000000e+00,  3.84000000e+00,  0.00000000e+00,0.00000000e+00,  0.00000000e+00,  9.60000000e+03, 
       0.00000000e+00, -3.84000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  9.60000000e+03],
       [ 0.00000000e+00,  0.00000000e+00,  3.84000000e+00, 0.00000000e+00, -9.60000000e+03,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00, -3.84000000e+00, 0.00000000e+00, -9.60000000e+03,  0.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 4.61538462e+03,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,-4.61538462e+03,  0.00000000e+00,  0.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00, -9.60000000e+03, 0.00000000e+00,  3.20000000e+07,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  9.60000000e+03, 0.00000000e+00,  1.60000000e+07,  0.00000000e+00],
       [ 0.00000000e+00,  9.60000000e+03,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  3.20000000e+07,
         0.00000000e+00, -9.60000000e+03,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  1.60000000e+07],
       [-2.40000000e+02,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         2.40000000e+02,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [ 0.00000000e+00, -3.84000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00, -9.60000000e+03,
         0.00000000e+00,  3.84000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00, -9.60000000e+03],
       [ 0.00000000e+00,  0.00000000e+00, -3.84000000e+00, 0.00000000e+00,  9.60000000e+03,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  3.84000000e+00, 0.00000000e+00,  9.60000000e+03,  0.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -4.61538462e+03,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  4.61538462e+03,  0.00000000e+00,  0.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00, -9.60000000e+03, 0.00000000e+00,  1.60000000e+07,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  9.60000000e+03, 0.00000000e+00,  3.20000000e+07,  0.00000000e+00],
       [ 0.00000000e+00,  9.60000000e+03,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  1.60000000e+07,
         0.00000000e+00, -9.60000000e+03,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  3.20000000e+07]])
    assert np.allclose(known, found)

# def test_partition_matrices(NC,EC):

#     K_global = np.zeros((18,18))
#     K_global[0] =+ 1
#     K_global[1] =+ 2
#     K_global[2] =+ 3
#     K_global[3] =+ 4
#     K_global[4] =+ 5
#     K_global[5] =+ 6
#     K_global[6] =+ 7
#     K_global[7] =+ 8
#     K_global[8] =+ 9
#     K_global[9] =+ 10
#     K_global[10] =+ 11
#     K_global[11] =+ 12
#     K_global[12] =+ 13
#     K_global[13] =+ 14
#     K_global[14] =+ 15
#     K_global[15] =+ 16
#     K_global[16] =+ 17
#     K_global[17] =+ 18

#     known, knownF,knownDOF = MSA.partition_matrices(NC,EC,K_global)

#     found = np.zeros((18,18))
#     found[0] =+ 7
#     found[1] =+ 8
#     found[2] =+ 9
#     found[3] =+ 10
#     found[4] =+ 11
#     found[5] =+ 12
#     found[6] =+ 16
#     found[7] =+ 17
#     found[8] =+ 18
#     found[9] =+ 1
#     found[10] =+ 2
#     found[11] =+ 3
#     found[12] =+ 4
#     found[13] =+ 5
#     found[14] =+ 6
#     found[15] =+ 13
#     found[16] =+ 14
#     found[17] =+ 15

#     foundF=np.zeros((18,1))
#     foundF[6] = 5/math.sqrt(2)
#     foundF[7] = -5/math.sqrt(2)

#     foundDOF = 6 

#     assert np.all(knownF == foundF)
#     assert knownDOF == foundDOF
#     assert np.all(known == found)

def test_run_MSA_solver(NC,EC):
    found_d,found_f = MSA.run_MSA_solver(NC,EC)
    known_f = np.reshape(np.array([-3.53553391,-17.6776695,0,0,0,5.30330086e4]),(6,1))
    known_d = np.reshape(np.array([6,-1.47313913e-2,7,-1.84142391,8,0,9,0,10,0,11,1.10485435e-3]),(6,2))
    #known=np.size(np.zeros((9,1)))
    assert np.allclose(known_f, found_f)
    assert np.allclose(known_d, found_d)
