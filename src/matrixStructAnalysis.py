import numpy as np
import math
import math_utils as MSA_math

class Nodes:
    def __init__(self,nodes,load, BC):
        self.nodes = nodes
        self.load = load
        self.BC = BC

        self.numNodes = np.size(nodes,0)
        self.numDOF = 6
        self.forces = np.reshape(load,(self.numNodes*6,1))

class Elements:
    def __init__(self,connections,E,nu,A,Iz,Iy,Ip,J,local_z=None):
        self.connections = connections
        self.E = E
        self.nu = nu
        self.A = A
        self.Iz = Iz
        self.Iy = Iy
        self.Ip = Ip
        self.J = J
        self.local_z = local_z

        self.numElements = np.size(self.connections,0)
        self.elem_load = np.zeros((12,self.numElements))
        self.L = np.zeros(self.numElements) + 1  


def run_MSA_solver(Nodes,Elements):
    
    K_global = find_global_frame_stiffness(Nodes,Elements)
    partK_DOF,partK_forces, partF, forceInd, dofInd = partition_matrices(Nodes,Elements,K_global)

    forceInd = np.reshape(np.array(forceInd),(np.size(forceInd),1))
    dofInd = np.reshape(np.array(dofInd),(np.size(dofInd),1))

    # solve for unknown displacement and forces
    Delta_f = np.linalg.solve(partK_DOF,partF)
    F_rxn = partK_forces @ Delta_f

    dof_displacement = np.hstack((forceInd, Delta_f))
    dof_force = np.hstack((dofInd,F_rxn[:np.size(dofInd)]))

    return dof_displacement, dof_force


def find_global_frame_stiffness(Nodes,Elements):

    K_global=np.zeros((Nodes.numNodes*6,Nodes.numNodes*6))
    for elm_ind in range(Elements.numElements):

        # find element end points
        node0 = Elements.connections[elm_ind][0]
        node1 = Elements.connections[elm_ind][1]

        x1 = Nodes.nodes[node0][0]
        y1 = Nodes.nodes[node0][1]
        z1 = Nodes.nodes[node0][2]
        x2 = Nodes.nodes[node1][0]
        y2 = Nodes.nodes[node1][1]
        z2 = Nodes.nodes[node1][2]

        # element length
        Elements.L[elm_ind] = math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2 )

        # DOF indices (add one for indexing)
        node1_dof_ind1 = 6 * node0
        node1_dof_ind2 = 6 * (node0 + 1)
        node2_dof_ind1 = 6 * node1
        node2_dof_ind2 = 6 * (node1 + 1)

        # material properties
        matProps = np.array([Elements.E[elm_ind], Elements.nu[elm_ind], Elements.A[elm_ind], Elements.L[elm_ind], Elements.Iy[elm_ind], Elements.Iz[elm_ind], Elements.J[elm_ind]])

        # find k_e = element stiffness local coordinates
        k_e = MSA_math.local_elastic_stiffness_matrix_3D_beam(*matProps)

        # transformation matrix
        # check for local z axis

        if Elements.local_z is None:
            gamma = MSA_math.rotation_matrix_3D(x1,y1,z1,x2,y2,z2)
        else:
            gamma = MSA_math.rotation_matrix_3D(x1,y1,z1,x2,y2,z2,Elements.local_z[elm_ind])
    
        Gamma = MSA_math.transformation_matrix_3D(gamma)
    
        # transform each element stiffness matrix to global coordinates
        k_eg = np.transpose(Gamma) @ k_e @ Gamma
                
        # assemble frame stiffness matrix

        K_global[node1_dof_ind1:node1_dof_ind2,node1_dof_ind1:node1_dof_ind2] += k_eg[0:6,0:6]
        K_global[node1_dof_ind1:node1_dof_ind2,node2_dof_ind1:node2_dof_ind2] += k_eg[0:6,6:12]
        K_global[node2_dof_ind1:node2_dof_ind2,node1_dof_ind1:node1_dof_ind2] += k_eg[6:12,0:6]
        K_global[node2_dof_ind1:node2_dof_ind2,node2_dof_ind1:node2_dof_ind2] += k_eg[6:12,6:12]

    return K_global

def partition_matrices(Nodes,Elements,K_global):

    # partition forces 
    # these correspond to unknown displacement
    # keep track of their index with forceInd

    partF = []
    forceInd = []

    allDOF = np.reshape(Nodes.BC,(Nodes.numNodes*6,1))
    for i, dof_BC in enumerate(allDOF):
        if dof_BC == 1: # free displacement
            partF.append(Nodes.forces[i])
            forceInd.append(i)

    # partition k pt 1
    # need rows + columns associated with unknown displacements

    unknown_dof = np.size(forceInd)
    partK_DOF = np.zeros((unknown_dof,unknown_dof))
    for rows,i in enumerate(forceInd):
        for cols,j in enumerate(forceInd):
            partK_DOF[rows][cols] = K_global[i,j]

    # partition k pt 2
    # need rows associated with unknown displacement 
    # and columns associated with known displacement

    dofInd = []
    known_disp = np.where(allDOF == 0)[0]
    partK_forces = np.zeros((unknown_dof,unknown_dof))
    for rows,i in enumerate(known_disp):
        for cols, j in enumerate(forceInd):
            partK_forces[rows,cols] = K_global[i,j]
        dofInd.append(i)

    # # assemble partitioned k_global based on which DOF are known (0) or unknown (1)
    # # # should have the same indexing as the forceInd
    # # unknown_disp = np.where(allDOF != 0)[0]
    # known_disp = np.where(allDOF == 0)[0]

    # # #partition_forces = np.vstack((Nodes.forces[unknown_disp],Nodes.forces[known_disp]))
    # # #partition_K_global = np.vstack((K_global[unknown_disp],K_global[known_disp]))
    # # partK_DOF = np.vstack((K_global[unknown_disp]))
    # partK_forces = np.vstack((K_global[known_disp]))

    return partK_DOF,partK_forces, partF, forceInd, dofInd