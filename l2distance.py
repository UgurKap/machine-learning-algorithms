import numpy as np
import innerproduct

def l2distance(X,Z=None):
    
    """
    function D=l2distance(X,Z)
    
    Computes the Euclidean distance matrix.
    Syntax:
    D=l2distance(X,Z)
    Input:
    X: nxd data matrix with n vectors (rows) of dimensionality d
    Z: mxd data matrix with m vectors (rows) of dimensionality d
    
    Output:
    Matrix D of size nxm
    D(i,j) is the Euclidean distance of X(i,:) and Z(j,:)
    
    call with only one input:
    l2distance(X)=l2distance(X,X)
    
    """
    
    if Z is None:
        Z = X
    
    n, d = X.shape
    m = Z.shape[0]
    
    G = innerproduct(X, Z)
    S = np.broadcast_to(innerproduct(X).diagonal().reshape(n, 1), (n,m))
    R = np.broadcast_to(innerproduct(Z).diagonal().reshape(m, 1).T, (n,m))
    
    D = np.sqrt(S - (2 * G) + R)
    
    assert(D.shape == (n, m))
    
    return D
