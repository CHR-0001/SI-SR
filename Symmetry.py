# calculate the symmetry error for all input variable pairs.

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import numpy as np
from Rmse_loss import rmse_loss

# return the extented function library that includs non-linear terms
def extension(feature,state=1,deg=2):
    if state == 0:
        poly_reg = PolynomialFeatures(degree=1)
        return poly_reg.fit_transform(feature)
    else:
        x = (feature[:,:state])
        poly_reg = PolynomialFeatures(degree=deg)
        x = poly_reg.fit_transform(x)
        cache = np.hstack((x,feature[:,state:]))
        for k in range(state,feature.shape[1]):
            cache = np.hstack((cache,x[:,1:]*feature[:,k].reshape(-1,1)))
        return cache


def sparse_reg(data,deg,alpha,tol,state=1,index=[]):
    variables = data[:,:-1]
    x = extension(variables,state=state,deg=deg)
    x[:,index] = 0
    y = data[:,-1]
    lin_reg = linear_model.Lasso(alpha=alpha)
    lin_reg.fit(x,y)
    index = 0
    label0 = np.ones(x.shape[1])
    label1 = np.where(np.abs(lin_reg.coef_)<tol,0,1)
    while not np.array_equal(label0,label1):
            label0 = label1.copy()
            x = x * label0
            lin_reg.fit(x,y)
            label1 = np.where(np.abs(lin_reg.coef_)<tol,0,1)
    return lin_reg

# define STLasso
def sparse_reg1(data,deg,alpha,tol,state=1):
    variables = data[:,:-1]
    x = extension(variables,deg=deg,state=state)
    loss_mini = 1000
    index = -1
    for i in range(data.shape[1]-1):
        lin_reg = sparse_reg(data=data,tol=tol,deg=deg,state=state,alpha=10**-6,index=[i])
        loss = rmse_loss(np.sum(lin_reg.coef_*x,axis=1),data[:,-1].astype('float32').reshape(-1,1))
        print(i,loss)
        if loss<loss_mini:
            loss_mini = loss
            index = i
    return sparse_reg(data=data,tol=tol,deg=deg,alpha=alpha,state=state,index=[index])

# calculate plus error
def check_translational_symmetry_plus(data, state, deg, lin_reg):
    n_variables = data.shape[1]-1
    variables = data[:, :-1]
    x = extension(variables,state=state,deg=deg)
    y = data[:,-1]
    result = np.zeros((int(n_variables * (n_variables - 1) / 2), 4))
    k = 0
    for i in range(0, n_variables):
        for j in range(0, n_variables):
            if i < j:
                fact_translate = variables.copy()
                a = 0.25 * (np.std(fact_translate[:, i]) + np.std(fact_translate[:, j]))
                fact_translate[:, i] = fact_translate[:, i] + a
                fact_translate[:, j] = fact_translate[:, j] - a
                feature = extension(fact_translate,state=state,deg=deg)
                list_errs = np.abs(lin_reg.predict(x) - lin_reg.predict(feature)) 
                fact_translate1 = variables.copy()
                fact_translate1[:, i] = fact_translate1[:, i] - a
                fact_translate1[:, j] = fact_translate1[:, j] + a
                feature1 = extension(fact_translate1,state=state,deg=deg)
                list_errs1 = np.abs(lin_reg.predict(x) - lin_reg.predict(feature1)) 
                error = np.median(list_errs + list_errs1)
                result[k] = np.array([i, j, error, 0])
                k = k + 1
    return result


# calculate minus error
def check_translational_symmetry_minus(data, state, deg, lin_reg):
    n_variables = data.shape[1]-1
    variables = data[:, :-1]
    x = extension(variables,state=state,deg=deg)
    y = data[:,-1]
    result = np.zeros((int(n_variables * (n_variables - 1) / 2), 4))
    k = 0
    for i in range(0, n_variables):
        for j in range(0, n_variables):
            if i < j:
                fact_translate = variables.copy()
                a = 0.25 * (np.std(fact_translate[:, i]) + np.std(fact_translate[:, j]))
                fact_translate[:, i] = fact_translate[:, i] + a
                fact_translate[:, j] = fact_translate[:, j] + a
                feature = extension(fact_translate,state=state,deg=deg)
                list_errs = np.abs(lin_reg.predict(x) - lin_reg.predict(feature)) 
                fact_translate1 = variables.copy()
                fact_translate1[:, i] = fact_translate1[:, i] - a
                fact_translate1[:, j] = fact_translate1[:, j] - a
                feature1 = extension(fact_translate1,state=state,deg=deg)
                list_errs1 = np.abs(lin_reg.predict(x) - lin_reg.predict(feature1)) 
                error = np.median(list_errs + list_errs1)
                result[k] = np.array([i, j, error, 1])
                k = k + 1
    return result


# calculate multiplication error
def check_translational_symmetry_multiply(data, state, deg, lin_reg):
    n_variables = data.shape[1]-1
    variables = data[:, :-1]
    x = extension(variables,state=state,deg=deg)
    y = data[:,-1]
    result = np.zeros((int(n_variables * (n_variables - 1) / 2), 4))
    k = 0
    a = 1.5
    for i in range(0, n_variables):
        for j in range(0, n_variables):
            if i < j:
                fact_translate = variables.copy()
                fact_translate[:, i] = fact_translate[:, i] * a
                fact_translate[:, j] = fact_translate[:, j] / a
                feature = extension(fact_translate,state=state,deg=deg)
                list_errs = np.abs(lin_reg.predict(x) - lin_reg.predict(feature)) 
                fact_translate1 = variables.copy()
                fact_translate1[:, i] = fact_translate1[:, i] / a
                fact_translate1[:, j] = fact_translate1[:, j] * a
                feature1 = extension(fact_translate1,state=state,deg=deg)
                list_errs1 = np.abs(lin_reg.predict(x) - lin_reg.predict(feature1)) 
                error = np.median(list_errs + list_errs1)
                result[k] = np.array([i, j, error, 2])
                k = k + 1
    return result


# calculate division error
def check_translational_symmetry_divide(data, state ,deg, lin_reg):
    n_variables = data.shape[1]-1
    variables = data[:, :-1]
    x = extension(variables,state=state,deg=deg)
    y = data[:,-1]
    result = np.zeros((int(n_variables * (n_variables - 1) / 2), 4))
    k = 0
    a = 1.5
    for i in range(0, n_variables):
        for j in range(0, n_variables):
            if i < j:
                fact_translate = variables.copy()
                fact_translate[:, i] = fact_translate[:, i] * a
                fact_translate[:, j] = fact_translate[:, j] * a
                feature = extension(fact_translate,state=state,deg=deg)
                list_errs = np.abs(lin_reg.predict(x) - lin_reg.predict(feature)) 
                fact_translate1 = variables.copy()
                fact_translate1[:, i] = fact_translate1[:, i] / a
                fact_translate1[:, j] = fact_translate1[:, j] / a
                feature1 = extension(fact_translate1,state=state,deg=deg)
                list_errs1 = np.abs(lin_reg.predict(x) - lin_reg.predict(feature1)) 
                error = np.median(list_errs + list_errs1)
                result[k] = np.array([i, j, error, 3])
                k = k + 1
    return result

def choose_alpha(data,deg,state):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn import linear_model
    import numpy as np
    from Rmse_loss import rmse_loss
    n_variables = data.shape[1] - 1
    variables = data[:, :-1]
    x = x = extension(variables,state=state,deg=deg)
    y = data[:,-1]
    alpha = 10
    lin_reg = linear_model.Lasso(alpha=alpha)
    lin_reg.fit(x, y)
    error = rmse_loss(lin_reg.predict(x), y)
    while error>0.05 and alpha>0.000001:
        alpha = alpha/10
        lin_reg = linear_model.Lasso(alpha=alpha)
        lin_reg.fit(x, y)
        error = rmse_loss(lin_reg.predict(x), y)
    print(error)
    return alpha

# calculate the relative error.
def symmetry(data, deg=2, state=1 ,tol=0.001, alpha=None):
    cache = data
    if alpha==None:
        alpha = choose_alpha(cache,deg,state)
    lin_reg = sparse_reg1(data, deg=deg, state=state,alpha=alpha, tol=tol)
    print(lin_reg.coef_)
    result0 = check_translational_symmetry_plus(cache, state, deg, lin_reg)
    result1 = check_translational_symmetry_minus(cache, state, deg, lin_reg)
    result2 = check_translational_symmetry_multiply(cache, state, deg, lin_reg)
    result3 = check_translational_symmetry_divide(cache, state, deg, lin_reg)
    results = np.zeros((len(result0), 6))
    for i in range(len(result0)):
        results[i] = [result0[i, 0], result0[i, 1], result0[i, 2], result1[i, 2], result2[i, 2], result3[i,2]]
    print(results)
    min_error = np.sort(results[:, 2:])[:, 0].reshape(-1, 1)
    results[:, 2:] = results[:, 2:] / (np.sort(results[:, 2:])[:, 1].reshape(-1, 1))
    r_errors = np.min(results[:, 2:], axis=1).reshape(-1, 1)
    index = np.argmin(results[:, 2:], axis=1).reshape(-1, 1)
    label = np.hstack((results[:, :2], min_error, r_errors, index))
    label[np.isnan(label)]=np.inf
    print(label)
    return label