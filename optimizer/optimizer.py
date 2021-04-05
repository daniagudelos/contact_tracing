#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:48:11 2021

@author: saitel
"""
from parameters.parameters import TestParameters1
from parameters.parameters import TestParameters2
from periodic.reproduction_number_case1 import ReproductionNumberCalculator as rn_case1
from periodic.reproduction_number_case2 import ReproductionNumberCalculator as rn_case2
from scipy.optimize import minimize  # differential_evolution
from scipy.optimize import NonlinearConstraint, Bounds
from helper.exporter import Exporter
from multiprocessing import Pool
import logging
import numpy as np


def min_case1():
    logger = logging.getLogger('min_case1')
    formatter = logging.Formatter('%(asctime)s %(message)s',
                                  '%m/%d/%Y %I:%M:%S %p')
    fh = logging.FileHandler(
        '/home/saitel/TUM/Thesis/Code/min_case1_SLSQP.log')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # period
    T = 7
    # initial beta
    # beta0 = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    #                 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1])
    # beta0 = np.array([1.00170228, 0.99516993, 1.00423168, 1.00270967,
    #                   1.99963177, 2.00046247, 2.00304428, 1.99900241,
    #                   2.99744607, 2.99884238, 3.00049076, 2.99970097,
    #                   2.99532557, 2.99588381, 3.0027815,  2.99779152,
    #                   2.00008628, 1.99801713, 1.99809348, 2.00272219,
    #                   1.00303413, 0.99985534, 0.99921461, 0.99919501,
    #                   0.99768312, 0.99884077, 0.99937266, 0.99952812])
    beta0 = np.array([1e-60, 1e-60, 1e-60, 1e-60, 1e-60, 1e-60, 1e-60, 1e-60,
                      1e-60, 1e-60, 1e-60, 1e-60, 1e-60, 1e-60, 1e-60, 1e-60,
                      1e-60, 1e-60, 1e-60, 1e-60, 25.0489, 1e-60, 1e-60, 1e-60,
                      1e-60, 1e-60, 1e-60, 1e-60])
    # initial parameters
    par = TestParameters1(beta0, p=1/3, h=0.25, period_time=T)

    # Calculators
    rnc_nct = rn_case1(logger, par, a_max=T, t_0_max=2 * T, trunc=1)
    rnc_ot_bct = rn_case1(logger, par, a_max=T, t_0_max=2 * T, trunc=1)

    R_0 = rnc_nct.calculateReproductionNumber
    R_p = rnc_ot_bct.calculateReproductionNumber

    # Ro constraint
    def constrain(beta):
        return R_0(beta, 0)

    cons = NonlinearConstraint(constrain, 3, 3)

    # Positive boundary
    bounds = Bounds([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                     np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                     np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                     np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
    logger.info('START SLSQP')

    result = minimize(R_p, beta0, bounds=bounds, method='SLSQP',
                      constraints=cons, args=(1,),
                      options={'maxiter': 1000, 'ftol': 1.0e-8, 'eps': 1e-06})
    Exporter.save_variable(result, 'min_case1_result')
    logger.info('Success %s', str(result.success))
    logger.info('STOP SLSQP')

    return result


def max_case1():
    logger = logging.getLogger('max_case1')
    formatter = logging.Formatter('%(asctime)s %(message)s',
                                  '%m/%d/%Y %I:%M:%S %p')
    fh = logging.FileHandler(
        '/home/saitel/TUM/Thesis/Code/max_case1_SLSQP.log')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # period
    T = 7
    # initial beta
    beta0 = np.array([1.65799, 1.65799, 1.65799, 1.65799, 1.65799, 1.65799,
                      1.65799, 1.65799, 1.65799, 1.65799, 1.65799, 1.65799,
                      1.65799, 1.65799, 1.65799, 1.65799, 1.65799, 1.65799,
                      1.65799, 1.65799, 1.65799, 1.65799, 1.65799, 1.65799,
                      1.65799, 1.65799, 1.65799, 1.65799])

    # beta0 = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    #                 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1])

    # beta0 = np.array([1.65546, 1.65546, 1.65546, 1.65546, 1.65546, 1.65546,
    #                   1.65546, 1.65546, 1.65546, 1.65546, 1.65546, 1.65546,
    #                   1.65546, 1.65546, 1.65546, 1.65546, 1.65546, 1.65546,
    #                   1.65546, 1.65546, 1.65546, 1.65546, 1.65546, 1.65546,
    #                   1.65546, 1.65546, 1.65546, 1.65546])

    # initial parameters
    par = TestParameters1(beta0, p=1/3, h=0.25, period_time=T)

    # Calculators
    rnc_nct = rn_case1(logger, par, a_max=T, t_0_max=2 * T, trunc=1)
    rnc_ot_bct = rn_case1(logger, par, a_max=T, t_0_max=2 * T, trunc=1)

    R_0 = rnc_nct.calculateReproductionNumber

    def R_p(beta, tracing_type):
        rp = -1 * rnc_ot_bct.calculateReproductionNumber(beta, tracing_type)
        logger.info('rp %s', str(rp))
        return rp

    # Ro constraint
    def constrain(beta):
        return R_0(beta, 0)

    cons = NonlinearConstraint(constrain, 3, 3)

    # Positive boundary
    bounds = Bounds([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
                     20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20])
    logger.info('START SLSQP')

    result = minimize(R_p, beta0, bounds=bounds, method='SLSQP',
                      constraints=cons, args=(1,),
                      options={'maxiter': 1000, 'ftol': 1.0e-8, 'eps': 1e-06})
    Exporter.save_variable(result, 'max_case1_result')
    logger.info('Success %s', str(result.success))
    logger.info('STOP SLSQP')

    return result


def min_case2():
    logger = logging.getLogger('min_case2')
    formatter = logging.Formatter('%(asctime)s %(message)s',
                                  '%m/%d/%Y %I:%M:%S %p')
    fh = logging.FileHandler(
        '/home/saitel/TUM/Thesis/Code/min_case2_SLSQP.log')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # period
    T = 7
    # initial beta
    beta0 = np.array([1e-60, 1e-60, 1e-60, 1e-60, 1e-60, 1e-60, 1e-60, 1e-60,
                      1e-60, 1e-60, 1e-60, 1e-60, 11.877, 1e-60, 1e-60, 1e-60,
                      1e-60, 1e-60, 1e-60, 1e-60, 1e-60, 1e-60, 1e-60, 1e-60,
                      1e-60, 1e-60, 1e-60, 1e-60])
    # beta0 = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    #                 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1])
    # beta0 = np.array([1.00170228, 0.99516993, 1.00423168, 1.00270967,
    #                   1.99963177, 2.00046247, 2.00304428, 1.99900241,
    #                   2.99744607, 2.99884238, 3.00049076, 2.99970097,
    #                   2.99532557, 2.99588381, 3.0027815,  2.99779152,
    #                   2.00008628, 1.99801713, 1.99809348, 2.00272219,
    #                   1.00303413, 0.99985534, 0.99921461, 0.99919501,
    #                   0.99768312, 0.99884077, 0.99937266, 0.99952812])
    # initial parameters
    par = TestParameters2(beta0, p=1/3, h=0.25, period_time=T)

    # Calculators
    rnc_nct = rn_case2(logger, par, a_max=T, t_0_max=2 * T, trunc=1)
    rnc_ot_bct = rn_case2(logger, par, a_max=T, t_0_max=2 * T, trunc=1)

    R_0 = rnc_nct.calculateReproductionNumber
    R_p = rnc_ot_bct.calculateReproductionNumber

    # Ro constraint
    def constrain(beta):
        return R_0(beta, 0)

    cons = NonlinearConstraint(constrain, 3, 3)

    # Positive boundary
    bounds = Bounds([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                     np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                     np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                     np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
    logger.info('START SLSQP')

    result = minimize(R_p, beta0, bounds=bounds, method='SLSQP',
                      constraints=cons, args=(1,),
                      options={'maxiter': 1000, 'ftol': 1.0e-8, 'eps': 1e-06})
    Exporter.save_variable(result, 'min_case2_result')
    logger.info('Success %s', str(result.success))
    logger.info('STOP SLSQP')

    return result


def max_case2():
    logger = logging.getLogger('max_case2')
    formatter = logging.Formatter('%(asctime)s %(message)s',
                                  '%m/%d/%Y %I:%M:%S %p')
    fh = logging.FileHandler(
        '/home/saitel/TUM/Thesis/Code/max_case2_SLSQP.log')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # period
    T = 7
    # initial beta
    beta0 = np.array([1.00859, 1.00859, 1.00859, 1.00859, 1.00859, 1.00859,
                      1.00859, 1.00859, 1.00859, 1.00859, 1.00859, 1.00859,
                      1.00859, 1.00859, 1.00859, 1.00859, 1.00859, 1.00859,
                      1.00859, 1.00859, 1.00859, 1.00859, 1.00859, 1.00859,
                      1.00859, 1.00859, 1.00859, 1.00859])
    # beta0 = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    #                 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1])

    # beta0 = np.array([1.65546, 1.65546, 1.65546, 1.65546, 1.65546, 1.65546,
    #                   1.65546, 1.65546, 1.65546, 1.65546, 1.65546, 1.65546,
    #                   1.65546, 1.65546, 1.65546, 1.65546, 1.65546, 1.65546,
    #                   1.65546, 1.65546, 1.65546, 1.65546, 1.65546, 1.65546,
    #                   1.65546, 1.65546, 1.65546, 1.65546])

    # initial parameters
    par = TestParameters2(beta0, p=1/3, h=0.25, period_time=T)

    # Calculators
    rnc_nct = rn_case2(logger, par, a_max=T, t_0_max=2 * T, trunc=1)
    rnc_ot_bct = rn_case2(logger, par, a_max=T, t_0_max=2 * T, trunc=1)

    R_0 = rnc_nct.calculateReproductionNumber

    def R_p(beta, tracing_type):
        rp = -1 * rnc_ot_bct.calculateReproductionNumber(beta, tracing_type)
        logger.info('rp %s', str(rp))
        return rp

    # Ro constraint
    def constrain(beta):
        return R_0(beta, 0)

    cons = NonlinearConstraint(constrain, 3, 3)

    # Positive boundary
    bounds = Bounds([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
                     20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20])
    logger.info('START SLSQP')

    result = minimize(R_p, beta0, bounds=bounds, method='SLSQP',
                      constraints=cons, args=(1,),
                      options={'maxiter': 1000, 'ftol': 1.0e-8, 'eps': 1e-06})
    Exporter.save_variable(result, 'max_case2_result')
    logger.info('Success %s', str(result.success))
    logger.info('STOP SLSQP')

    return result


def main():
    pool = Pool(4)
    result1 = pool.apply_async(min_case1)
    result2 = pool.apply_async(max_case1)
    result3 = pool.apply_async(min_case2)
    result4 = pool.apply_async(max_case2)

    pool.close()
    pool.join()

    return result1, result2, result3, result4


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    result1, result2, result3, result4 = main()
