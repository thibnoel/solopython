from ctypes import *
import numpy as np

######## Legs collisions 
# Init C types and store function call in a vector y of dim (nb_motors + 1)*nb_pairs
def getLegsCollisionsResults(q, cdll_func, nb_motors, nb_pairs, witnessPoints=False):
    n_wp = 6 if witnessPoints else 0

    ny = (nb_motors+1+n_wp)*nb_pairs

    DoubleArrayIn = c_double*nb_motors
    DoubleArrayOut = c_double*ny

    y = np.zeros(ny).tolist()

    q = DoubleArrayIn(*q)
    y = DoubleArrayOut(*y)
    cdll_func.solo_autocollision_legs_legs_forward_zero(q, y)

    return y


# Extract distances from results vector
def getLegsDistances(legsCollResults, nb_motors, nb_pairs, witnessPoints=False):
    n_wp = 6 if witnessPoints else 0
    return np.array([legsCollResults[i*(1+nb_motors+n_wp)] for i in range(nb_pairs)])


# Extract jacobians from results vector
def getLegsJacobians(legsCollResults, nb_motors, nb_pairs, witnessPoints=False):
    n_wp = 6 if witnessPoints else 0
    return np.vstack([legsCollResults[i*(1+nb_motors+n_wp) + 1 : i*(1+nb_motors+n_wp) + 1 + nb_motors] for i in range(nb_pairs)])


def getLegsWitnessPoints(legsCollResults, nb_motors, nb_pairs):
    wPoints = []
    for i in range(nb_pairs):
        ind_offset = i*(1+nb_motors+6) + 1 + nb_motors 
        wpoint1, wpoint2 = legsCollResults[ind_offset:ind_offset+3], legsCollResults[ind_offset+3:ind_offset+6]
        wPoints.append([wpoint1, wpoint2])
    return wPoints


######## Shoulders collisions 
# Init C types and store function call in a vector y of dim q_dim+1
def getShoulderCollisionsResults(q, cdll_func, q_dim):
    DoubleArrayIn = c_double*(2*q_dim)
    DoubleArrayOut = c_double*(1 + q_dim)

    x = np.concatenate((np.cos(q), np.sin(q))).tolist()
    y = np.zeros(1 + q_dim).tolist()
    
    x = DoubleArrayIn(*x)
    y = DoubleArrayOut(*y)
    cdll_func.solo_autocollision_nn_shoulder_forward_zero(x,y)

    return y


# Extract distance from results vector
def getShoulderDistance(shoulderCollResult, offset=0):
    return np.array(shoulderCollResult[0]) - offset


# Extract jacobian from results vector
def getShoulderJacobian(shoulderCollResult):
    return np.array(shoulderCollResult[1:])


def getAllShouldersCollisionsResults(q, cdll_func, q_dim=2, offset=0):
    distances = []
    jacobians = []
    shoulder_syms = [[1,1], [-1,1], [1,-1], [-1,-1]]

    for shoulder_ind in range(4):
        q_ind = [k for k in range(3*shoulder_ind,3*shoulder_ind + q_dim)]
        q_val = q[q_ind]
        sym = shoulder_syms[shoulder_ind]

        sym_q_val = np.array(q_val.copy())
        sym_q_val[0] = sym[0]*sym_q_val[0]
        sym_q_val[1] = sym[1]*sym_q_val[1]

        shoulder_result = getShoulderCollisionsResults(sym_q_val, cdll_func, q_dim)
        
        J = np.array(getShoulderJacobian(shoulder_result))
        J[0] = sym[0]*J[0]
        J[1] = sym[1]*J[1]

        formatted_J = np.zeros(len(q))
        formatted_J[q_ind] = J
        
        distances.append(getShoulderDistance(shoulder_result, offset=offset))
        jacobians.append(formatted_J)

    return np.array(distances), np.vstack(jacobians)
