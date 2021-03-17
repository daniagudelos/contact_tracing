#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:48:11 2021

@author: saitel
"""
from parameters.parameters import TestParameters1
from periodic.reproduction_number import ReproductionNumberCalculator
from scipy.optimize import differential_evolution
from scipy.optimize import NonlinearConstraint, Bounds
import numpy as np


T = 7
beta0 = np.array([1, 1, 1, 1, 3, 3, 3, 3, 3.5, 3.5, 3.5, 3.5, 4, 4, 4, 4,
                  3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1])
parameter = par = TestParameters1(beta0, p=1/3, h=0.25, period_time=T)

rnc_nct = ReproductionNumberCalculator(par, a_max=T, t_0_max=2 * T,
                                       trunc=1)
rnc_ot_bct = ReproductionNumberCalculator(par, a_max=T, t_0_max=2 * T,
                                          trunc=1)

R_0 = rnc_nct.calculateReproductionNumber
R_p = rnc_ot_bct.calculateReproductionNumber

# Ro constraint
r_0 = rnc_nct.calculateReproductionNumber(beta0, 0)


def constrain(beta):
    return R_0(beta, 0)


cons = NonlinearConstraint(constrain, r_0, r_0)

# Positive boundary
bounds = Bounds([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [10, 10, 10, 10, 10, 10, 10,
                 10, 10, 10, 10, 10, 10, 10,
                 10, 10, 10, 10, 10, 10, 10,
                 10, 10, 10, 10, 10, 10, 10])

# bounds = [(0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf),
#           (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf),
#           (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf),
#           (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf),
#           (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf)]

# res = linprog(, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds])

# res = minimize(R_p, beta0, bounds=bnds,
#                        constraints=cons, tol=1e-3, args=(1,),
#                        options={'maxiter': 10, 'eps':1})

result = differential_evolution(R_p, bounds=bounds, constraints=(cons),
                                seed=1, workers=14)