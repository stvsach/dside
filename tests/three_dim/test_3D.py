from dside import DSI
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

def Sobol_sequence(lbd, ubd, power_no):
    """
    Create 2^power_no of inputs for sampling based on the lists of lbd (lower
    bound) and ubd (upper bound).
    """
    sampler = qmc.Sobol(d = len(lbd), scramble = False)
    inputs = sampler.random_base2(m = power_no)
    inputs = qmc.scale(inputs, lbd, ubd)
    return inputs

def qcp4con(x1, x2, x3):
    # qcp4con problem from Metta et al (2020)
    # https://doi.org/10.1002/aic.17095
    
    A = np.array([[0, 0, 1], [0, -1, 0], [-2, 1, -1]])
    b = np.array([[3], [0], [4]])
    y = np.array([[1.5], [-0.5], [-5]])
    z = np.array([[0], [-1], [-6]])

    if len(x1.shape) != 0:
        con_res = []
        X = np.array([x1, x2, x3])
        for i in range(X.shape[1]):
            x1 = X[:, i][0]
            x2 = X[:, i][1]
            x3 = X[:, i][2]
            x = np.array([[x1], [x2], [x3]])
            g1 = x1 + x2 + x3 - 4
            g2 = 3*x2 + x3 - 6
            g3 = np.matmul(np.matmul(np.matmul(x.T, A.T), A), x) - 2*np.matmul(np.matmul(y.T, A), x) + y**2 - 0.25*(b - z)**2
            con_res.append(np.sum([g1, g2, np.sum(g3)]))
        con_res = np.array(con_res)
    else:
        x = np.array([[x1], [x2], [x3]])
        g1 = x1 + x2 + x3 - 4
        g2 = 3*x2 + x3 - 6
        g3 = np.matmul(np.matmul(np.matmul(x.T, A.T), A), x) - 2*np.matmul(np.matmul(y.T, A), x) + y**2 - 0.25*(b - z)**2
        con_res = np.sum([g1, g2, np.sum(g3)])
    return con_res

# Preparing example dataset
x1 = [0, 2]
x2 = [0, 3]
x3 = [0, 3]

bounds_array = np.array([x1, x2, x3])
lbd = bounds_array[:, 0]
ubd = bounds_array[:, 1]

pwr = 12
inputs = Sobol_sequence(lbd, ubd, pwr)

vn = ['x1', 'x2', 'x3']
df = pd.DataFrame(inputs, columns = vn)

x1 = df['x1'].to_numpy()
x2 = df['x2'].to_numpy()
x3 = df['x3'].to_numpy()
df['con_res'] = qcp4con(x1, x2, x3)
constraints = {'con_res': [0, 1e20]}
# Design space identification
ds = DSI(df) # initialise design space class using the dataset

ds.screen(constraints) # screen the data using constraints
ds.find_DSp(vn, opt = {'printF': True}) # form the alpha shape
ds.plot()
ds.send_output('DSI_3D_without_hmv')

opt = {'hmv': 'con_res', 'hmvlabel': 'con_res'}
ds.plot(opt = opt)
ds.find_AOR([0.5, 2.5, 0.5])
ds.send_output('DSI_3D_with_hmv')