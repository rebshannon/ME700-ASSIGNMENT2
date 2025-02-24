import numpy as np
import math
import math_utils as MSA_math

class Nodes:
    def __init__(self,nodes,nodeType,load):
        self.nodes = nodes
        self.nodeType = nodeType
        self.load = load
        
        self.numNodes = np.size(nodes,0)
        self.numDOF = 6
        self.forces = np.reshape(load,(self.numNodes*6,1))

class Elements:
    def __init__(self,connections,E,nu,A,Iz,Iy,Ip,J):
        self.connections = connections
        self.E = E
        self.nu = nu
        self.A = A
        self.Iz = Iz
        self.Iy = Iy
        self.Ip = Ip
        self.J = J

        self.numElements = np.size(self.connections,0)
        self.elem_load = np.zeros((12,self.numElements))
        self.L = np.zeros(self.numElements)+1  


def run_MSA_solver(Nodes,Elements):
    
    K_global = find_global_frame_stiffness(Nodes,Elements)
    partition_K_global, partition_forces, num_unknown_disp = partition_matrices(Nodes,Elements,K_global)

    # pull out what you need for forces and stiffness to solve for unknown displacement
    known_forces = partition_forces[0:num_unknown_disp]
    K_known_forces = partition_K_global[0:num_unknown_disp,0:num_unknown_disp]

    Delta_f = np.linalg.solve(K_known_forces,known_forces)

    # find rxn forces
    K_unknown_forces = partition_K_global[0:num_unknown_disp,num_unknown_disp:]
    F_rxn = K_unknown_forces @ Delta_f

    # assemble displacement and forces in arrays
    #all_forces = np.reshape(np.vstack((Nodes.forces,F_rxn)),(Nodes.numNodes,6))
    # all_disp = np.reshape(Delta_f,(Nodes.numNodes,6)))

    #return np.size(Delta_f),np.size(F_rxn)
    return F_rxn


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

        ##DEFINE V_TEMP

        # transformation matrix
        gamma = MSA_math.rotation_matrix_3D(x1,y1,z1,x2,y2,z2)
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
    # make DOF matrix
    DOF = np.zeros((Nodes.numNodes,Nodes.numDOF))

    # 0 = known DOF
    # 1 = unknown DOF
    for node_ind in range(Nodes.numNodes):
        if Nodes.nodeType[node_ind] == 0:
            DOF[node_ind][:] = 1 
        elif Nodes.nodeType[node_ind] == 2:
            DOF[node_ind][3:6] = 1
    
    DOF = np.reshape(DOF,(Nodes.numNodes*6,1))

    # now have F = KX where F,X are 18x1, K is 18x18
    # rearrange the system of equations so that the unknown DOF are at the top
    # unknown forces != 0
    unknown_disp = np.where(DOF != 0)[0]
    known_disp = np.where(DOF == 0)[0]

    known_forces = np.array(Nodes.forces[unknown_disp])
    unknown_forces = np.array(Nodes.forces[known_disp])

    partition_forces = np.vstack((Nodes.forces[unknown_disp],Nodes.forces[known_disp]))
    partition_K_global = np.vstack((K_global[unknown_disp],K_global[known_disp]))

    num_unknown_disp = np.size(unknown_disp)

    return partition_K_global, partition_forces, num_unknown_disp
    #return DOF, partition_forces, partition_K_global, num_unknown_disp
