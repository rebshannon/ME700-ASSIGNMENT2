import numpy as np
import scipy as sp
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
    def __init__(self,connections,E,nu,A,Iz,Iy,I_rho,J,local_z=None):
        self.connections = connections
        self.E = E
        self.nu = nu
        self.A = A
        self.Iz = Iz
        self.Iy = Iy
        self.I_rho = I_rho
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

def assemble_allDisp_array(Nodes, disp):
    # put all dispalcements and forces in a single displacement and force array (WIP)
    all_disp = np.zeros(Nodes.numDOF*Nodes.numNodes)
    for row in disp:
        index = int(row[0])
        value = row[1]
        all_disp[index] = value
    return all_disp

def assemble_allForce_array(Nodes,force):
    all_force = np.reshape(Nodes.load,(Nodes.numDOF*Nodes.numNodes,1))
    for row in force:
        index = int(row[0])
        value = row[1]
        all_force[index] += value
    return all_force

def find_local_loads(Nodes,Elements,forces,elm_ind):
    '''forces should include all reactions and applied loads'''

    # find coords and dof index of end points
    x1,y1,z1,x2,y2,z2,node1_dof,node2_dof,node0,node1 = find_element_endPoints_dof(Nodes, Elements, elm_ind)

    # total applied load on element
    
    # CHECK THIS - FORCES NOT DEFINED YET
    #F_tot = forces[node0,:] + forces[node1,:]

    # tranformation matrix
    gamma = find_gamma(x1,y1,z1,x2,y2,z2,Elements.local_z, elm_ind)
    Gamma = MSA_math.transformation_matrix_3D(gamma)

    forces = np.array(forces)

    # force in local coordinates
    F_tot_local = Gamma @ forces

    return F_tot_local

def run_buckling_part(Nodes, Elements):

    displacement, forces = run_MSA_solver(Nodes,Elements)
    #move these to run_MSA
    all_displacement = assemble_allDisp_array(Nodes, displacement)
    all_force = assemble_allForce_array(Nodes,forces)

    # find geometric stiffness - for all in global coordinates
    # find elastic stiffnesses
    # assemble to global matricies
    # partition them
    # then solve eigan problem

    eigenValues = []
    eigenVectors = []

    for elm_ind in range(Elements.numElements):

        x1,y1,z1,x2,y2,z2,node1_dof,node2_dof, node0,node1 = find_element_endPoints_dof(Nodes, Elements, elm_ind)

        # material properties and force
        matProps = np.array([Elements.A[elm_ind], Elements.L[elm_ind], Elements.I_rho[elm_ind]])
        F_applied = all_force[node2_dof[0]:node2_dof[1]]

        # local force and stiffness
        F_local = find_local_loads(Nodes,Elements,all_force,elm_ind)
        k_g = MSA_math.local_geometric_stiffness_matrix_3D_beam(*matProps,*F_applied)

        matProps = np.array([Elements.E[elm_ind], Elements.nu[elm_ind], Elements.A[elm_ind], Elements.L[elm_ind], Elements.Iy[elm_ind], Elements.Iz[elm_ind], Elements.J[elm_ind]])

        # local elastic stiffness 
        k_e = MSA_math.local_elastic_stiffness_matrix_3D_beam(*matProps)

        # eigen problem to solve [k_e + x*k_g]delta = 0
        # NEED JUST FREE-FREE COMPONENTS
        eigenVal,eigenVect = solve_eigen_problem(k_e,-k_g)

        eigenValues.append(eigenVal)
        eigenVectors.append(eigenVect)

    # find the smallest eigenValue = critical load
    P_crit = min(eigenValues)
    return P_crit

def solve_eigen_problem(k_e,k_g):

        eigenvalues, eigenvectors = sp.linalg.eig(k_e, -k_g)

        # Extract the real part of the eigenvalues (ignoring the imaginary part)
        eigenvalues = np.real(eigenvalues)

        # Filter for positive eigenvalues
        positive_indices = eigenvalues > 0
        positive_eigenvalues = eigenvalues[positive_indices]
        positive_eigenvectors = eigenvectors[:, positive_indices]

        # Sort the positive eigenvalues in ascending order
        sorted_indices = np.argsort(positive_eigenvalues)
        sorted_eigenvalues = positive_eigenvalues[sorted_indices]
        sorted_eigenvectors = positive_eigenvectors[:, sorted_indices]

        # only return the first index since that is the smallest eigen value
        min_eigenValue = sorted_eigenvalues[0]
        min_eigenVector = sorted_eigenvectors[0]

        return min_eigenValue, min_eigenVector


def find_global_frame_stiffness(Nodes,Elements):

    K_global=np.zeros((Nodes.numNodes*6,Nodes.numNodes*6))
    for elm_ind in range(Elements.numElements):

        # find element endpoints and associated DOF
        x1,y1,z1,x2,y2,z2,node1_dof,node2_dof,node0,node1 = find_element_endPoints_dof(Nodes, Elements, elm_ind)

        # material properties
        matProps = np.array([Elements.E[elm_ind], Elements.nu[elm_ind], Elements.A[elm_ind], Elements.L[elm_ind], Elements.Iy[elm_ind], Elements.Iz[elm_ind], Elements.J[elm_ind]])

        # find k_e = element stiffness local coordinates
        k_e = MSA_math.local_elastic_stiffness_matrix_3D_beam(*matProps)

        # transformation matrix
        # checks for local z axis      
        gamma = find_gamma(x1,y1,z1,x2,y2,z2,Elements.local_z, elm_ind)
        Gamma = MSA_math.transformation_matrix_3D(gamma)
    
        # transform each element stiffness matrix to global coordinates
        k_eg = np.transpose(Gamma) @ k_e @ Gamma
                
        # assemble frame stiffness matrix

        K_global[node1_dof[0]:node1_dof[1],node1_dof[0]:node1_dof[1]] += k_eg[0:6,0:6]
        K_global[node1_dof[0]:node1_dof[1],node2_dof[0]:node2_dof[1]] += k_eg[0:6,6:12]
        K_global[node2_dof[0]:node2_dof[1],node1_dof[0]:node1_dof[1]] += k_eg[6:12,0:6]
        K_global[node2_dof[0]:node2_dof[1],node2_dof[0]:node2_dof[1]] += k_eg[6:12,6:12]

    return K_global

def find_element_endPoints_dof(Nodes, Elements, elm_ind):

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
    node1_dof = np.array([6 * node0, 6 * (node0 + 1)])
    node2_dof = np.array([6 * node1, 6 * (node1 + 1)])
    
    return x1,y1,z1,x2,y2,z2,node1_dof, node2_dof, node0, node1

def find_gamma(x1,y1,z1,x2,y2,z2,local_z, elm_ind):
    if local_z is None: gamma = MSA_math.rotation_matrix_3D(x1,y1,z1,x2,y2,z2)
    else: gamma = MSA_math.rotation_matrix_3D(x1,y1,z1,x2,y2,z2,local_z[elm_ind])
    return gamma

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

    return partK_DOF,partK_forces, partF, forceInd, dofInd