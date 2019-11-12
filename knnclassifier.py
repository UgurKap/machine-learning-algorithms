import numpy as np
from scipy import stats
import innerproduct
import l2distance

def findknn(xTr,xTe,k):
    """
    function [indices,dists]=findknn(xTr,xTe,k);
    
    Finds the k nearest neighbors of xTe in xTr.
    
    Input:
    xTr = nxd input matrix with n row-vectors of dimensionality d
    xTe = mxd input matrix with m row-vectors of dimensionality d
    k = number of nearest neighbors to be found
    
    Output:
    
    I = kxm matrix, where indices(i,j) is the i^th nearest neighbor of xTe(j,:)
    D = Euclidean distances to the respective nearest neighbors
    """
    
    D = l2distance(xTe, xTr)
    I = np.argsort(D, axis=1)[:, :k].T
    D = np.sort(D, axis=1)[:, :k].T
    return I, D

def knnclassifier(xTr,yTr,xTe,k):
    """
    function preds=knnclassifier(xTr,yTr,xTe,k);
    
    k-nn classifier 
    
    Input:
    xTr = nxd input matrix with n row-vectors of dimensionality d
    xTe = mxd input matrix with m row-vectors of dimensionality d
    k = number of nearest neighbors to be found
    
    Output:
    
    preds = predicted labels, ie preds(i) is the predicted label of xTe(i,:)
    """
    
    m = xTe.shape[0]
    
    indices, _ = findknn(xTr, xTe, k)
    
    labels = yTr[indices]
    preds = stats.mode(labels)[0].reshape((m, 1))
    
    return preds
