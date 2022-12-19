from src.dside import DSI
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

def ex3(x1, x2):
    ## 4.1. Nonlinear nonconvex test problem\
    ## https://doi.org/10.1016/j.ces.2015.06.014
    g1 = -2*x1 + x2 - 15
    g2 = x1**2/2 + 4*x1 - x2 - 5
    g3 = -(x1 - 4)**2/5 - x2**2/0.5 + 10
    return g1, g2, g3

# Preparing example dataset
x1    = [-10, 15]
x2    = [-15, 15]

bounds_array = np.array([x1, x2])
lbd = bounds_array[:, 0]
ubd = bounds_array[:, 1]

pwr = 12
inputs = Sobol_sequence(lbd, ubd, pwr)

vn = ['x1', 'x2']
df = pd.DataFrame(inputs, columns = vn)
df['g1'], df['g2'], df['g3'] = ex3(df['x1'], df['x2'])
u = -0 # upper bound of the KPIs: g1, g2, and g3
constraints = {'g1': [-1e20, u], 'g2': [-1e20, u], 'g3': [-1e20, u]}
# Design space identification
ds = DSI(df) # initialise design space class using the dataset

ds.screen(constraints) # screen the data using constraints
ds.find_DSp(vn) # form the alpha shape
ds.plot()
ds.find_AOR([-5, -5])
ds.send_output('wihtout_hmv')

opt = {'hmv': 'g1', 'hmvlabel': 'g2'}
ds.plot(opt = opt)
ds.find_AOR([-5, -5])
ds.send_output('wiht_hmv')