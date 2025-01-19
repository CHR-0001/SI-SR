# select the optimal model on the Pareto front.

import numpy as np

def select(loss_comp,per=0.1):
    lc = loss_comp.copy()
    k = 0
    while k < len(lc) - 1:
        cache = lc[k + 1:, :]
        dec_per = (lc[k, 0] - cache[:, 0]) /lc[k, 0] /  (cache[:, 1] - lc[k, 1])
        dec_per = np.where(dec_per > per, 1, -1)
        print(dec_per)
        if np.max(dec_per) < 0:
            break
        k = k + 1
    return k
