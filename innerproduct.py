import numpy as np

def innerproduct(X,Z=None):
    
    """
    Computes the inner-product matrix.
    
    Syntax:
    D=innerproduct(X,Z)
    
    Input:
    X: nxd data matrix with n vectors (rows) of dimensionality d
    Z: mxd data matrix with m vectors (rows) of dimensionality d
    
    Output:
    Matrix G of size nxm
    G[i,j] is the inner-product between vectors X[i,:] and Z[j,:]
    
    Call with only one input:
    innerproduct(X)=innerproduct(X,X)
    """
    
    if Z is None:
        Z = X
    
    n, d = X.shape[0], X.shape[1]
    m = Z.shape[0]
    
    D = np.matmul(X, Z.T)
    
    assert(D.shape == (n, m))
    
    return D
