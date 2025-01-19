from Rmse_loss import rmse_loss 
from Random_choice import random_choice
from Train_fun import train_fun
from Select import select
from SR import sr
from Symmetry import symmetry
from Compare import compare
from select import select
from Var_inplace import var_inplace
import numpy as np
def run_all(filename,feature,loss_comp,exp_list,exp,data_points,niterations,tol,alpha,state,
            binary_operators,unary_operators,complexity_of_operators,threshold,deg):
    if feature.shape[1]>2:
        data = feature
        loss_comp0,exp_list0,exp0 = loss_comp,exp_list,exp
        label_list = symmetry(data,alpha=alpha,state=state,deg=deg,tol=tol)
        new_data,i,j,k = var_inplace(data,label_list)
        if i<state:
            state = state-1
        with open(filename,'a') as f:
            f.write('\n')
            f.write('{}\t{}\t{}\n'.format(i,j,k))
        loss_comp1,exp_list1,exp1 = sr(new_data[:,:-1],new_data[:,-1],
                                       data_points=data_points,
                                       niterations=niterations,
                                       binary_operators=binary_operators,
                                       unary_operators=unary_operators,
                                       complexity_of_operators=complexity_of_operators)
        rmse = compare(loss_comp0,loss_comp1)
        with open(filename,'a') as f:
            f.write(str(exp1)+'\n')
            f.write(str(exp_list1)+'\n')
            f.write(str(loss_comp1)+'\n')
            f.write(str(label_list)+'\n')
            f.write(str(rmse)+'\n')    
        if np.mean(rmse)<threshold:
            with open(filename,'a') as f:
                f.write('variable inplace right \n')
            loss_compf, exp_listf, expf=run_all(filename,new_data,loss_comp1,exp_list1,exp1,
                                                data_points = data_points,
                                                niterations = niterations,
                                                binary_operators = binary_operators,
                                                unary_operators = unary_operators,
                                                complexity_of_operators = complexity_of_operators,
                                                threshold = threshold,
                                                tol = tol,
                                                alpha = alpha,
                                                deg = deg,
                                                state =state)
            return loss_compf, exp_listf, expf
        else:
            with open(filename,'a') as f:
                f.write('variable inplace false \n')
            return loss_comp0, exp_list0, exp0
        
    else:
        print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n',
              'there is only one variable',
              '\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
        return loss_comp,exp_list,exp
def all0(filename,
        data,
        data_points=2000,
        niterations=500,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=["exp", "log", "sin", "cube"],
        complexity_of_operators={"exp":3 , "log": 3, "sin": 3, "cube":3},
        compare_threshold=1,
        tol = 0.001,
        deg=2,
        state = 1,
        alpha = None):
    loss_comp0,exp_list0,exp0 = sr(data[:,:-1],data[:,-1],
                                   data_points=data_points,
                                   niterations=niterations,
                                   binary_operators=binary_operators,
                                   unary_operators=unary_operators,
                                   complexity_of_operators=complexity_of_operators)
    with open(filename,'w') as f:
        f.write(str(exp0)+'\n')
        f.write(str(exp_list0)+'\n')
        f.write(str(loss_comp0)+'\n')
    loss_comp, exp_list, exp = run_all(filename,data,loss_comp0,exp_list0,exp0,
                                       data_points=data_points,
                                       niterations=niterations,
                                       binary_operators=binary_operators,
                                       unary_operators=unary_operators,
                                       complexity_of_operators=complexity_of_operators,
                                       threshold=compare_threshold,
                                       deg=deg,
                                       state=state,
                                       tol = tol,
                                       alpha=alpha)
    return loss_comp, exp_list, exp