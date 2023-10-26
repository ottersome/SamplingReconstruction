import numpy as np
from scipy.linalg import null_space
from scipy.sparse import diags
from scipy.linalg import null_space

def generate_true_birth(lam, mu):
    # Given Lambda Mu 
    # Generate Generator Matrix
    pass
def generate_true_gmatrix(rates, shape):
    shape += 1
    gen_mat = np.eye(shape)
    k = [rates['mu']*np.ones(shape-1), -(rates['mu']+rates['lam'])*np.ones(shape), rates['lam']*np.ones(shape-1)]
    offset = [-1,0,1]
    tridiag = diags(k, offset).toarray()
    tridiag[0,0] = - rates['lam']
    tridiag[-1,-1] = -rates['mu']
    return tridiag
    
def one_step_matrx(Q):
    stat_dist = []
    dqm1 = np.linalg.inv(np.diag(np.diag(Q)))
    W = np.eye(Q.shape[0]) - dqm1@Q
    W = W / np.sum(W,axis=1)[0:np.newaxis]

    return W
def get_stat_dist(Q,eigen_method =False):
    pi = null_space(np.transpose(Q)).flatten()
    return  pi/np.sum(pi)
