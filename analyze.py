import numpy as np

def analyze(kind,truth,preds):
    """
    function output=analyze(kind,truth,preds)         
    Analyses the accuracy of a prediction

    Input:
    kind='acc' classification error
    kind='abs' absolute loss
    """
    err = 0
    
    if kind == "acc":
        accurate = (truth == preds)
        err = np.sum(accurate) / accurate.size
    elif kind == "abs":
        loss = np.abs(truth - preds)
        err = np.sum(loss)
        
    return err
