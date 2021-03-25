#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:43:24 2021

@author: saitel
"""
import numpy as np
import logging
from helper.exporter import Exporter
from parameters.parameters import TestParameters1
from periodic.no_contact_tracing import NoCT
from periodic.backward_tracing.one_time_bct import OneTimeBCT
from periodic.forward_tracing.one_time_fct import OneTimeFCT
from periodic.full_tracing.one_time_lct import OneTimeLCT
from periodic.backward_tracing.recursive_bct import RecursiveBCT
from periodic.forward_tracing.recursive_fct import RecursiveFCT
from periodic.full_tracing.recursive_lct import RecursiveLCT


class ReproductionNumberCalculator:
    def __init__(self, logger, parameters, a_max, t_0_max, trunc=2):
        """
        Parameters
        ----------
        logger : Logger
            a logger.
        parameters : Parameters
            Initial parameters. beta2 is replaced later.
        a_max : INTEGER
            Number of a periods to calculate.
        t_0_max : INTEGER
            Number of t_0 periods to calculate
        trunc : INTEGER, optional
            Number of extra periods to approximate infinity. The default is 4.

        Returns
        -------
        None.

        """
        self.logger = logger
        self.optimizer_iteration = 0
        self.parameters = parameters
        self.h = self.parameters.get_h()
        self.p = self.parameters.get_p()
        self.period = self.parameters.get_period()
        self.period_length = self.parameters.get_period_length()
        self.beta = self.parameters.get_beta
        self.beta_array = None
        # Number of extra periods to approximate infinity
        self.trunc = trunc
        self.a_max = (a_max + self.trunc) * self.period
        self.a_length = int(round(self.a_max / self.h, 1))
        self.t_0_max = t_0_max * self.period
        self.t_0_length = int(round(self.t_0_max / self.h, 1))
        self.t_0_array = np.linspace(0.0, self.t_0_max, self.t_0_length + 1)
        self.t_array = np.linspace(0.0, self.t_0_max, self.t_0_length + 1)
        self.t_length = int(round(self.t_0_max / self.h, 1))
        self.a_array = np.linspace(0.0, self.a_max,
                                   self.a_length + 1)

        self.switcher = {
            0: self.calculate_kappa_nct,
            1: self.calculate_kappa_ot_bct,
            2: self.calculate_kappa_ot_fct,
            3: self.calculate_kappa_ot_lct,
            4: self.calculate_kappa_re_bct,
            5: self.calculate_kappa_re_fct,
            6: self.calculate_kappa_re_lct
        }

        self.name_switcher = {
            0: 'kappa_nct',
            1: 'kappa_ot_bct',
            2: 'kappa_ot_fct',
            3: 'kappa_ot_lct',
            4: 'kappa_re_bct',
            5: 'kappa_re_fct',
            6: 'kappa_re_lct'
        }

        self.kappa = None

    def get_kappa(self, tracing_type):
        try:
            name = self.name_switcher.get(tracing_type)
            kappa = Exporter.load_variable(name)
        except (FileNotFoundError):
            print('Calculating kappa - type: ', tracing_type)
            func = self.switcher.get(tracing_type)
            kappa = func()
        return kappa

    def calculate_kappa_nct(self):
        nct = NoCT(parameters=self.parameters, a_max=self.a_max,
                   t_0_max=self.t_0_max)
        _, _, kappa = nct.calculate_kappa_hat()
        name = self.name_switcher.get(0)
        Exporter.save_variable(kappa, name)
        return kappa

    def calculate_kappa_ot_bct(self):
        nct = OneTimeBCT(parameters=self.parameters, a_max=self.a_max,
                         t_0_max=self.t_0_max)
        _, _, kappa = nct.calculate_kappa_minus()
        name = self.name_switcher.get(1)
        Exporter.save_variable(kappa, name)
        return kappa

    def calculate_kappa_re_bct(self):
        nct = RecursiveBCT(parameters=self.parameters, a_max=self.a_max,
                           t_0_max=self.t_0_max)
        _, _, kappa = nct.calculate_kappa_minus()
        name = self.name_switcher.get(4)
        Exporter.save_variable(kappa, name)
        return kappa

    def calculate_kappa_ot_fct(self):
        nct = OneTimeFCT(parameters=self.parameters, n_gen=6, trunc=10,
                         a_max=self.a_max, t_0_max=self.t_0_max)
        _, _, kappa = nct.calculate_kappa_plus()
        name = self.name_switcher.get(2)
        Exporter.save_variable(kappa, name)
        return kappa

    def calculate_kappa_re_fct(self):
        nct = RecursiveFCT(parameters=self.parameters, n_gen=4, trunc=10,
                           a_max=self.a_max, t_0_max=self.t_0_max)
        _, _, kappa = nct.calculate_kappa_plus()
        name = self.name_switcher.get(5)
        Exporter.save_variable(kappa, name)
        return kappa

    def calculate_kappa_ot_lct(self):
        nct = OneTimeLCT(parameters=self.parameters, n_gen=4, trunc=10,
                         a_max=self.a_max, t_0_max=self.t_0_max)
        _, _, kappa = nct.calculate_kappa()
        name = self.name_switcher.get(3)
        Exporter.save_variable(kappa, name)
        return kappa

    def calculate_kappa_re_lct(self):
        nct = RecursiveLCT(parameters=self.parameters, n_gen=4, trunc=10,
                           a_max=self.a_max, t_0_max=self.t_0_max)
        _, _, kappa = nct.calculate_kappa()
        name = self.name_switcher.get(6)
        Exporter.save_variable(kappa, name)
        return kappa

    def build_vector(self, w):

        F_sum = np.zeros(self.period_length + 1)
        for i in range(0, self.period_length + 1):
            F = np.zeros(self.period_length + 1)

            # First part: from 0 to t (from 0 to i-1)
            for j in range(0, max(0, i - 2)):
                H_sum = 0
                temp = np.zeros((self.trunc + 1))
                a_index = i - j  # This is always +

                # sums from 0 to an infinite number of periods
                for n in range(0, self.trunc + 1):
                    temp[n] = (self.beta(self.a_array[a_index] +
                                         n * self.period,
                                         self.t_array[i]) *
                               self.kappa[i, a_index + n * self.period_length])
                H_sum = np.sum(temp)
                F[j] = H_sum * w[j]

            #  Second part: from t to T (from i-1 to N-1)
            for j in range(max(0, i - 1), self.period_length + 1):
                H_sum = 0
                temp = np.zeros((self.trunc + 1))
                index = i - j
                while(index < 0):
                    index = index + self.period_length
                for n in range(0, self.trunc + 1):
                    temp[n] = (self.beta(self.a_array[index] +
                                         (n + 1) * self.period,
                                         self.t_array[i]) *
                               self.kappa[i, index +
                                          (n + 1) * self.period_length])
                H_sum = np.sum(temp)
                F[j] = H_sum * w[j]
            F_sum[i] = np.sum(F)
        u = self.period * F_sum * self.h
        return u

    def calculateReproductionNumber(self, beta, tracing_type):
        self.beta_array = beta
        self.parameters = TestParameters1(beta)
        self.beta = self.parameters.get_beta
        self.kappa = self.get_kappa(tracing_type)

        ew, ev = self.get_eigenpair()

        self.logger.info('Iteration %s, beta: %s', self.optimizer_iteration,
                         np.array2string(self.beta_array))
        self.logger.info('Iteration %s, ew: %s', self.optimizer_iteration,
                         str(ew))

        if tracing_type != 0:
            self.optimizer_iteration += 1

        return ew

    def get_eigenpair(self):
        ev = np.zeros_like(self.t_0_array)
        ev[1] = 1
        ev[3] = 1
        ev = ev / np.linalg.norm(ev)
        ew = 0

        error = 10
        while error > 1e-5:
            z = self.build_vector(ev)  # np.matmul(A, ev)  # A  * ev
            ev = z / np.linalg.norm(z)
            Aev = self.build_vector(ev)
            ew = np.transpose(ev).dot(Aev)
            error = np.linalg.norm(Aev - ew * ev)
        return ew, ev


def bct_test(par, T):
    rnc = ReproductionNumberCalculator(par, a_max=2 * T, t_0_max=2 * T,
                                       tracing_type=0, trunc=1)
    rnc2 = ReproductionNumberCalculator(par, a_max=2 * T, t_0_max=2 * T,
                                        tracing_type=1, trunc=1)
    rnc3 = ReproductionNumberCalculator(par, a_max=2 * T, t_0_max=2 * T,
                                        tracing_type=4, trunc=1)
    ew1, _ = rnc.calculateReproductionNumber()
    print('nct ', ew1)
    ew2, _ = rnc2.calculateReproductionNumber()
    print('ot_bct ', ew2)
    ew3, _ = rnc3.calculateReproductionNumber()
    print('re_bct ', ew3)
    return ew1, ew2, ew3


def fct_test(par, T):
    #rnc = ReproductionNumberCalculator(par, a_max=2 * T, t_0_max=2 * T,
     #                                  tracing_type=0, trunc=1)
    rnc2 = ReproductionNumberCalculator(par, a_max=2 * T, t_0_max=2 * T,
                                        tracing_type=2, trunc=1)
    rnc3 = ReproductionNumberCalculator(par, a_max=2 * T, t_0_max=2 * T,
                                        tracing_type=5, trunc=1)
    # ew1, _ = rnc.calculateReproductionNumber()
    # print('nct ', ew1)
    ew2, _ = rnc2.calculateReproductionNumber()
    print('ot_lct ', ew2)
    ew3, _ = rnc3.calculateReproductionNumber()
    print('re_fct ', ew3)
    return ew2, ew3  # ew1, ew2, ew3

def lct_test(par, T):
    # rnc2 = ReproductionNumberCalculator(par, a_max=2 * T, t_0_max=2 * T,
    #                                    tracing_type=3, trunc=1)
    rnc3 = ReproductionNumberCalculator(par, a_max=T, t_0_max=2 * T,
                                        tracing_type=6, trunc=1)
    # ew1, _ = rnc.calculateReproductionNumber()
    # print('nct ', ew1)
    # ew2, _ = rnc2.calculateReproductionNumber()
    # print('ot_lct ', ew2)
    ew3, _ = rnc3.calculateReproductionNumber()
    print('re_lct ', ew3)
    return ew3  # ew1, ew2, ew3


def main():
    T = 7

    # beta0 = np.array([1, 1, 1, 1, 3, 3, 3, 3, 3.5, 3.5, 3.5, 3.5, 4, 4, 4, 4,
    #                  3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1])

    beta0 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    par = TestParameters1(beta0, p=1/3, h=0.25, period_time=T)

    logger = logging.getLogger('rep_num_test')
    formatter = logging.Formatter('%(asctime)s %(message)s',
                                  '%m/%d/%Y %I:%M:%S %p')
    fh = logging.FileHandler('/home/saitel/TUM/Thesis/Code/rep_num_test.log')
    # fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    rnc = ReproductionNumberCalculator(logger, par, a_max=2, t_0_max=2,
                                       trunc=0)
    ew1 = rnc.calculateReproductionNumber(beta0, 2)
    return ew1


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    ew1 = main()
