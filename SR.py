# define the symbolic regression model.

import numpy as np
import sympy
from Random_choice import random_choice

def sr(X, y,
       model_selection="score",
       data_points=500,
       niterations=100,
       binary_operators=["+", "*", "-", "/"],
       unary_operators=["exp", "log", "sin"],
       complexity_of_operators={"exp":3, "log": 3, "sin":3},
       verbosity=0,
       progress=False,
       print_precision=2):
    from pysr import PySRRegressor
    model = PySRRegressor(
        maxsize=20,
        niterations=niterations,  # < Increase me for better results
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        complexity_of_operators=complexity_of_operators,
        loss="loss(prediction, target) = (prediction - target)^2",
    )
    cache = np.hstack((X, y.reshape(-1, 1)))
    cache = random_choice(cache, data_points)
    model.fit(cache[:, :-1], cache[:, -1])
    loss_comp = np.array([model.get_best(0)[1], model.get_best(0)[0]])
    sym_exp = [model.get_best(0)[4]]
    for i in range(1, 30):
        try:
            loss_comp = np.vstack((loss_comp, np.array([model.get_best(i)[1], model.get_best(i)[0]])))
            sym_exp.append(model.get_best(i)[4])
        except:
            pass
    return loss_comp, sym_exp, model.get_best()[4]



