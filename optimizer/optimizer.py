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
    beta0 = np.array([0.2, 0.2, 0.2, 0.24741959,
                      7.28E-01, 1.07816417, 1.0607325, 0.60621096,
                      8.82E-01, 0.99107515, 0.65395751, 7,
                      0.50522205, 0.81943302, 0.71081963, 0.88830103,
                      1.13303859, 0.67266571, 0.93446271, 0.56359306,
                      0.98098441, 1.14759829, 1.07E+00, 0.74946083,
                      1.064594, 1.26587296, 0.4, 0.2])
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
    beta0 = np.array([0.85, 0.85, 0.85, 0.85, 0.85,
                      0.820518, 0.85, 0.85, 0.831422, 0.85,
                      0.85, 0.856831, 0.749512, 0.774931, 0.85,
                      0.799953, 0.887688, 0.913373, 0.807308, 0.900797,
                      0.784698, 0.810171, 0.820771, 0.747022, 0.918913,
                      0.854015, 0.85, 0.85])

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
    beta0 = np.array([0.1, 1.31E-01, 2.93E-01, 4.10E-02,
                      0.4, 1.9, 2.20E-01, 5.93E-01,
                      5.93E-01, 2.41E-04, 2.16E-01, 2.64E-01,
                      8.41E-02, 1.9, 4.62E-01, 7.25E-21,
                      2.38E-01, 4.13E-01, 2.62E-01, 3.27E-01,
                      0.5, 5.11E-01, 0.5, 7.2,
                      8.37E-01, 2.50E-01, 5.30E-02, 7.47E-01])

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
    beta0 = np.array([0.486, 0.48, 0.48, 0.49232964,
                      0.03659941, 0.45581489, 0.39924676, 0.65134535,
                      0.718, 0.77183116, 0.45682268, 0.90388365,
                      0.40701978, 0.71598858, 0.47772997, 0.48,
                      0.49250512, 0.56, 0.56777387, 0.46635623,
                      8.24E-01, 0.42467745, 0.82718676, 0.35211435,
                      5.26E-01, 0.26156585, 0.31122063, 0.69608544])

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
