#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:48:11 2021

@author: saitel
"""
from parameters.parameters import TestParameters1
from periodic.reproduction_number import ReproductionNumberCalculator
from scipy.optimize import minimize  # differential_evolution
from scipy.optimize import NonlinearConstraint, Bounds
from multiprocessing import Pool
import logging
import numpy as np


def solve_optim1():
    logger = logging.getLogger('trust-constr')
    formatter = logging.Formatter('%(asctime)s %(message)s',
                                  '%m/%d/%Y %I:%M:%S %p')
    fh = logging.FileHandler('/home/saitel/TUM/Thesis/Code/trust-constr.log')
    # fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # period
    T = 7
    # initial beta
    beta0 = np.array([8.35605740e-13, 6.66426290e-12, 2.29316288e-11,
                      7.42019869e-01, 3.35762692e+00, 3.17140304e+00,
                      3.15382815e+00, 3.13113382e+00, 3.60125754e+00,
                      3.59210071e+00, 3.60207457e+00, 3.57093952e+00,
                      4.06754615e+00, 4.09562978e+00, 4.08613250e+00,
                      4.04624675e+00, 3.01995612e+00, 3.04343909e+00,
                      3.00555974e+00, 2.98062361e+00, 1.94936812e+00,
                      1.96435729e+00, 1.86778192e+00, 1.83965838e+00,
                      5.98393006e-01, 5.44390370e-02, 2.27910361e-11,
                      9.15493380e-12])

    # initial parameters
    par = TestParameters1(beta0, p=1/3, h=0.25, period_time=T)

    # Calculators
    rnc_nct = ReproductionNumberCalculator(logger, par, a_max=T, t_0_max=2 * T,
                                           trunc=1)
    rnc_ot_bct = ReproductionNumberCalculator(logger, par, a_max=T,
                                              t_0_max=2 * T, trunc=1)

    R_0 = rnc_nct.calculateReproductionNumber
    R_p = rnc_ot_bct.calculateReproductionNumber

    # Ro constraint
    def constrain(beta):
        return R_0(beta, 0)

    cons = NonlinearConstraint(constrain, 3, 3)

    # Positive boundary
    bounds = Bounds([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                     10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])

    logger.info('START trust-constr')

    result = minimize(R_p, beta0, bounds=bounds, method='trust-constr',
                      constraints=cons, args=(1,),
                      options={'maxiter': 1000, 'verbose': 2})
    logger.info('Success ', result.success)
    logger.info('STOP trust-constr')

    return result


def solve_optim2():
    logger = logging.getLogger('solve_optim2')
    formatter = logging.Formatter('%(asctime)s %(message)s',
                                  '%m/%d/%Y %I:%M:%S %p')
    fh = logging.FileHandler('/home/saitel/TUM/Thesis/Code/SLSQP.log')
    # fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # period
    T = 7
    # initial beta
    beta0 = np.array([8.35605740e-13, 6.66426290e-12, 2.29316288e-11,
                      7.42019869e-01, 3.35762692e+00, 3.17140304e+00,
                      3.15382815e+00, 3.13113382e+00, 3.60125754e+00,
                      3.59210071e+00, 3.60207457e+00, 3.57093952e+00,
                      4.06754615e+00, 4.09562978e+00, 4.08613250e+00,
                      4.04624675e+00, 3.01995612e+00, 3.04343909e+00,
                      3.00555974e+00, 2.98062361e+00, 1.94936812e+00,
                      1.96435729e+00, 1.86778192e+00, 1.83965838e+00,
                      5.98393006e-01, 5.44390370e-02, 2.27910361e-11,
                      9.15493380e-12])

    # initial parameters
    par = TestParameters1(beta0, p=1/3, h=0.25, period_time=T)

    # Calculators
    rnc_nct = ReproductionNumberCalculator(logger, par, a_max=T,
                                           t_0_max=2 * T, trunc=1)
    rnc_ot_bct = ReproductionNumberCalculator(logger, par, a_max=T,
                                              t_0_max=2 * T, trunc=1)

    R_0 = rnc_nct.calculateReproductionNumber
    R_p = rnc_ot_bct.calculateReproductionNumber

    # Ro constraint
    def constrain(beta):
        return R_0(beta, 0)

    cons = NonlinearConstraint(constrain, 3, 3)

    # Positive boundary
    bounds = Bounds([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                     10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
    logger.info('START SLSQP')

    result = minimize(R_p, beta0, bounds=bounds, method='SLSQP',
                      constraints=cons, args=(1,),
                      options={'maxiter': 1000})
    logger.info('Success ', result.success)
    logger.info('STOP SLSQP')

    return result


def main():
    pool = Pool()
    result1 = pool.apply_async(solve_optim1)
    result2 = pool.apply_async(solve_optim2)
    answer1 = result1.get(timeout=14400)
    answer2 = result2.get(timeout=14400)

    return result1, result2, answer1, answer2


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    result1, result2, answer1, answer2 = main()

# res = linprog(, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds])

# result = differential_evolution(R_p, bounds=bounds, args=(1,),
#                                constraints=(cons),
#                                seed=1, workers=-1)
# beta0 = np.array([1, 1, 1, 1, 3, 3, 3, 3, 3.5, 3.5, 3.5, 3.5, 4, 4, 4, 4,
    #                  3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1])