# make variable replacement. For two variable x_i and x_j (j<k), the i-th column will be delected and the new variable will be placed at j-th column.

import numpy as np

def var_inplace(data,label_list):
    index = np.argmin(label_list[:,3])
    label = label_list[index]
    i,j,k = int(label[0]),int(label[1]),int(label[4])
    if k==0:
        print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nfound plus symmetry between {} and {}\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n'.format(i,j))
        new_data = data.copy()
        new_data[:,j] = data[:,i]+data[:,j]
        new_data = np.delete(new_data,obj=i,axis=1)
        return new_data,i,j,k
    elif k==1:
        print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nfound minus symmetry between {} and {}\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n'.format(i,j))
        new_data = data.copy()
        new_data[:,j] = data[:,i]-data[:,j]
        new_data = np.delete(new_data,obj=i,axis=1)
        return new_data,i,j,k
    elif k==2:
        print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nfound multiply symmetry between {} and {}\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n'.format(i,j))
        new_data = data.copy()
        new_data[:,j] = data[:,i]*data[:,j]
        new_data = np.delete(new_data,obj=i,axis=1)
        return new_data,i,j,k
    elif k==3:
        print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nfound divide symmetry between {} and {}\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n'.format(i,j))
        new_data = data.copy()
        new_data[:,j] = data[:,i]/data[:,j]
        new_data = np.delete(new_data,obj=i,axis=1)
        return new_data,i,j,k