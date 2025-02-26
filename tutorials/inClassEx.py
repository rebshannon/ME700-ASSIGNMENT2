import numpy as np
import math
import matrixStructAnalysis as MSA
from matrixStructAnalysis import Nodes
from matrixStructAnalysis import Elements

# DEFINE GEOMETRY
# define the geometry of the problem

# each row is a node defined in (x, y, z) coordinates
nodes = np.array([[0, 0, 10],
                  [15, 0, 10],
                  [15, 0, 0]])

# the load applied at each node as (Fx, Fy, Fz, Mx, My, Mz)
load = np.array([[0,0,0,0,0,0],
                 [0.1,0.05,-0.07,0.05,-0.1,0.25],
                 [0,0,0,0,0,0]])

# the type of boundary condition
BC = np.array([[0,0,0,0,0,0],
               [1,1,1,1,1,1],
               [0,0,0,1,1,1]])

# which nodes are connected to form elements
# each row is a new elements with end points (node0, node1)
connections = np.array([[0, 1],
                        [1, 2]])

# material properties
# enter the material properties for each element
# the index corresponds to the element index

b = 0.5
h = 1

E = np.array([1000,1000])
nu = np.array([0.3,0.3])
A = np.array([b*h,b*h])
Iz = np.array([b*h**3/12,b*h**3/12])
Iy = np.array([h*b**3/12,h*b**3/12])
Ip = np.array([b*h/(12*(b**2+h**2)),b*h/(12*(b**2+h**2))])
J = np.array([0.02861,0.02861])

# DEFINE CLASSES
Nodes = Nodes(nodes=nodes,load=load,BC=BC)
Elements = Elements(connections,E,nu,A,Iz,Iy,Ip,J)

# RUN THE SOLVER
displacement, forces = MSA.run_MSA_solver(Nodes,Elements)

print("Displacements")
print(displacement)
print("Forces")
print(forces)

