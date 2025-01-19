# compare the results of syobolic regression, return the V_error.

import numpy as np

def compare(loss_comp0,loss_comp1):
    comp = list(set(list(loss_comp0[:,1].astype('int16'))).intersection(set(list(loss_comp1[:,1].astype('int16')))))
    loss = np.zeros((len(comp),2))
    for i in range(len(comp)):
        complexity = comp[i]
        loss[i,0] = (loss_comp0[np.abs(loss_comp0[:,1]-complexity)<0.01])[0,0]
        loss[i,1] = (loss_comp1[np.abs(loss_comp1[:,1]-complexity)<0.01])[0,0]
    loss0 = loss[:,0]
    loss1 = loss[:,1]  
    rmse = (loss[:,1]-loss[:,0])/loss[:,0]
    try:
        rmse = rmse[1:7]
    except:
        pass
    print(rmse)
    return rmse