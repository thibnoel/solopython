from ctypes import *
import numpy as np

def getLegsCollisionsResults12(q, cdll_func):
    DoubleArray12 = c_double*12
    DoubleArray260 = c_double*260
    
    y = np.zeros(20*13).tolist()

    q = DoubleArray12(*q)
    y = DoubleArray260(*y)
    cdll_func.solo_autocollision_legs_legs_forward_zero(q,y)

    return y


def getDistances12(collResults):
    return np.array([collResults[i*13] for i in range(20)])


def getJacobians12(collResults):
    return np.vstack([collResults[i*13 + 1 : (i+1)*13] for i in range(20)])



def getLegsCollisionsResults8(q, cdll_func):
    DoubleArray8 = c_double*8
    DoubleArray54 = c_double*54
    
    y = np.zeros(6*9).tolist()

    q = DoubleArray8(*q)
    y = DoubleArray54(*y)
    cdll_func.solo_autocollision_legs_legs_forward_zero(q,y)

    return y


def getDistances8(collResults):
    return np.array([collResults[i*9] for i in range(6)])


def getJacobians8(collResults):
    return np.vstack([collResults[i*9 + 1 : (i+1)*9] for i in range(6)])
