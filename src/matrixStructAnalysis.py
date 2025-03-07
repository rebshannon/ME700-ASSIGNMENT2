import numpy as np
import scipy as sp
import math
import math_utils as MSA_math
import matplotlib.pyplot as plt

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


def run_linear_solver(Nodes,Elements):
    
    K_global = find_global_frame_stiffness(Nodes,Elements)
    partK_DOF,partK_forces, partF, forceInd, dofInd = partition_matrices(Nodes,Elements,K_global)

    forceInd = np.reshape(np.array(forceInd),(np.size(forceInd),1))
    dofInd = np.reshape(np.array(dofInd),(np.size(dofInd),1))

    # solve for unknown displacement and forces
    Delta_f = np.linalg.solve(partK_DOF,partF)
    F_rxn = partK_forces @ Delta_f

    # just the unknowns
    dof_displacement = np.hstack((forceInd, Delta_f))
    dof_force = np.hstack((dofInd,F_rxn[:np.size(dofInd)]))

    # put all displacements and forces into arrays
    all_displacement = assemble_allDisp_array(Nodes, dof_displacement)
    all_force = assemble_allForce_array(Nodes, dof_force)

    return all_displacement, all_force

def allElm_endPointForce_local(Nodes,Elements,displacements):
    allEndPointForces_local = []

    for elm_ind in range(Elements.numElements):

        element_force = find_endPoint_force_local(Nodes,Elements,displacements,elm_ind)
        allEndPointForces_local.append(element_force)
  
    return allEndPointForces_local

def find_endPoint_force_local(Nodes,Elements,displacements,elm_ind):
    
    # find original coordinates and DOF
    x1,y1,z1,x2,y2,z2,node1_dof,node2_dof,node0,node1 = find_element_endPoints_dof(Nodes, Elements, elm_ind)

    # displacements for this element
    elm_displacement = np.hstack((displacements[node1_dof[0]:node1_dof[1]],displacements[node2_dof[0]:node2_dof[1]]))
    elm_displacement = np.reshape(elm_displacement,(12,1))

    # transformation matrix
    gamma = find_gamma(x1,y1,z1,x2,y2,z2,Elements.local_z, elm_ind)
    Gamma = MSA_math.transformation_matrix_3D(gamma)

    # displacement in local coordinates
    local_disp = Gamma @ elm_displacement

    # find k_e = element stiffness local coordinates
    matProps = np.array([Elements.E[elm_ind], Elements.nu[elm_ind], Elements.A[elm_ind], Elements.L[elm_ind], Elements.Iy[elm_ind], Elements.Iz[elm_ind], Elements.J[elm_ind]])
    k_e = MSA_math.local_elastic_stiffness_matrix_3D_beam(*matProps)

    # endpoint force in local coords
    local_force = k_e @ local_disp

    return local_force


def run_elasticCriticalLoad_analysis(Nodes, Elements):

    # linear part
    displacement, forces = run_linear_solver(Nodes,Elements)

    # local forces
    local_forces = allElm_endPointForce_local(Nodes,Elements,displacement)
    local_forces = np.transpose(local_forces[0])

    # find frame geometric stiffness in global coordinates
    K_geo_global = find_global_frame_geometricStiffness(Nodes,Elements,local_forces)
    partKgeo_DOF, partKgeo_forces, partGeoF, forceGeoInd, dofGeoInd = partition_matrices(Nodes,Elements,K_geo_global)

    # frame elastic stiffness in global coordinates
    K_global = find_global_frame_stiffness(Nodes,Elements)
    partK_DOF,partK_forces, partF, forceInd, dofInd = partition_matrices(Nodes,Elements,K_global)

    #eigenVal,eigenVect = solve_eigen_problem(partK_DOF,-partKgeo_DOF)

    eigenvalues, eigenvectors = sp.linalg.eig(partK_DOF,-partKgeo_DOF)    
    # Extract the real part of the eigenvalues (ignoring the imaginary part)
    eigenvalues = np.real(eigenvalues)

    # find the smallest eigenValue = critical load
    P_crit = np.min(np.abs(eigenvalues))
    
    return P_crit,eigenvectors

def find_global_frame_geometricStiffness(Nodes,Elements,F_local):

    K_geo_global = np.zeros((Nodes.numNodes*6,Nodes.numNodes*6))

    for elm_ind in range(Elements.numElements):

        x1,y1,z1,x2,y2,z2,node1_dof,node2_dof,node0,node1 = find_element_endPoints_dof(Nodes, Elements, elm_ind)

        # material properties
        matProps = np.array([Elements.L[elm_ind], Elements.A[elm_ind],Elements.I_rho[elm_ind]])
        
        # force on element (may need to adjust)
        F_local_element = F_local[elm_ind]
        F_indexForStiffness = [6,9,4,5,10,11]
        F_forStiffness = F_local_element[F_indexForStiffness]
               
        k_g = MSA_math.local_geometric_stiffness_matrix_3D_beam(*matProps,*F_forStiffness)

        # transformation matrix
        # checks for local z axis      
        gamma = find_gamma(x1,y1,z1,x2,y2,z2,Elements.local_z, elm_ind)
        Gamma = MSA_math.transformation_matrix_3D(gamma)
    
        # transform each element stiffness matrix to global coordinates
        k_g = np.transpose(Gamma) @ k_g @ Gamma
                
        # assemble frame stiffness matrix

        K_geo_global[node1_dof[0]:node1_dof[1],node1_dof[0]:node1_dof[1]] += k_g[0:6,0:6]
        K_geo_global[node1_dof[0]:node1_dof[1],node2_dof[0]:node2_dof[1]] += k_g[0:6,6:12]
        K_geo_global[node2_dof[0]:node2_dof[1],node1_dof[0]:node1_dof[1]] += k_g[6:12,0:6]
        K_geo_global[node2_dof[0]:node2_dof[1],node2_dof[0]:node2_dof[1]] += k_g[6:12,6:12]

    return K_geo_global

def solve_eigen_problem(k_e,k_g):

    eigenvalues, eigenvectors = sp.linalg.eig(k_e, -k_g)

    # Extract the real part of the eigenvalues (ignoring the imaginary part)
    eigenvalues = np.real(eigenvalues)

    real_pos = np.isreal(eigenvalues) 

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

def assemble_allDisp_array(Nodes, disp):
    # put all displacements and forces in a single displacement and force array (WIP)
    all_disp = np.zeros(Nodes.numDOF*Nodes.numNodes)
    for row in disp:
        index = int(row[0])
        value = row[1]
        all_disp[index] = value
    return all_disp

def assemble_allForce_array(Nodes,force):
    all_force = np.reshape(Nodes.load,(Nodes.numDOF*Nodes.numNodes,1))
    print(force)
    for row in force:
        index = int(row[0])
        value = row[1]
        all_force[index] += value
    return all_force

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
    if None in local_z[elm_ind]: gamma = MSA_math.rotation_matrix_3D(x1,y1,z1,x2,y2,z2)
    else: 
        local_z_axis = np.array(local_z[elm_ind],dtype=float)
        gamma = MSA_math.rotation_matrix_3D(x1,y1,z1,x2,y2,z2,local_z_axis)
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