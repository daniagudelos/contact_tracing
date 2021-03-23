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
from helper.exporter import Exporter
from multiprocessing import Pool
import logging
import numpy as np


def solve_re_bct():
    logger = logging.getLogger('solve_optim1')
    formatter = logging.Formatter('%(asctime)s %(message)s',
                                  '%m/%d/%Y %I:%M:%S %p')
    fh = logging.FileHandler('/home/saitel/TUM/Thesis/Code/re_bct_SLSQP.log')
    # fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # period
    T = 7
    # initial beta
    beta0 = np.array([9.08611232e-23, 2.68520261e-18, 1.37888902e-04,
                      8.62416099e-01, 3.45384645e+00, 2.90048590e+00,
                      3.10142458e+00, 2.56611650e+00, 3.69617468e+00,
                      3.53392347e+00, 3.93938788e+00, 3.83235544e+00,
                      4.10657378e+00, 4.12117064e+00, 3.75965060e+00,
                      3.88962677e+00, 3.58391512e+00, 3.23374595e+00,
                      3.55333169e+00, 3.17617161e+00, 2.17393546e+00,
                      2.09780564e+00, 2.00315860e+00, 2.27245852e+00,
                      9.45962165e-01, 3.93843186e-01, 1.73737956e-04,
                      6.19287167e-04])

    # initial parameters
    par = TestParameters1(beta0, p=1/3, h=0.25, period_time=T)

    # Calculators
    rnc_nct = ReproductionNumberCalculator(logger, par, a_max=T,
                                           t_0_max=2 * T, trunc=1)
    rnc_re_bct = ReproductionNumberCalculator(logger, par, a_max=T,
                                              t_0_max=2 * T, trunc=1)

    R_0 = rnc_nct.calculateReproductionNumber
    R_p = rnc_re_bct.calculateReproductionNumber

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
                      constraints=cons, args=(4,),
                      options={'maxiter': 1000})
    Exporter.save_variable(result, 're_bct_result')
    logger.info('Success %s', str(result.success))
    logger.info('STOP SLSQP')

    return result


def solve_ot_bct():
    logger = logging.getLogger('solve_optim2')
    formatter = logging.Formatter('%(asctime)s %(message)s',
                                  '%m/%d/%Y %I:%M:%S %p')
    fh = logging.FileHandler('/home/saitel/TUM/Thesis/Code/ot_bct_SLSQP.log')
    # fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # period
    T = 7
    # initial beta
    beta0 = np.array([9.08611232e-23, 2.68520261e-18, 1.37888902e-04,
                      8.62416099e-01, 3.45384645e+00, 2.90048590e+00,
                      3.10142458e+00, 2.56611650e+00, 3.69617468e+00,
                      3.53392347e+00, 3.93938788e+00, 3.83235544e+00,
                      4.10657378e+00, 4.12117064e+00, 3.75965060e+00,
                      3.88962677e+00, 3.58391512e+00, 3.23374595e+00,
                      3.55333169e+00, 3.17617161e+00, 2.17393546e+00,
                      2.09780564e+00, 2.00315860e+00, 2.27245852e+00,
                      9.45962165e-01, 3.93843186e-01, 1.73737956e-04,
                      6.19287167e-04])

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
    Exporter.save_variable(result, 'ot_bct_result')
    logger.info('Success %s', str(result.success))
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
    # result1, result2, answer1, answer2 = main()
    result = solve_optim2()

# res = linprog(, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds])

# result = differential_evolution(R_p, bounds=bounds, args=(1,),
#                                constraints=(cons),
#                                seed=1, workers=-1)
# beta0 = np.array([1, 1, 1, 1, 3, 3, 3, 3, 3.5, 3.5, 3.5, 3.5, 4, 4, 4, 4,
    #                  3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1])