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
            Number of extra periods to approximate infinity.

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
        self.a_max = (a_max + self.trunc)
        self.a_length = (a_max + self.trunc) * self.period_length
        self.t_0_max = t_0_max
        self.t_0_length = t_0_max * self.period_length
        self.t_0_array = np.linspace(0.0, t_0_max * self.period,
                                     self.t_0_length + 1)
        self.t_array = np.array(self.t_0_array)
        self.t_length = t_0_max * self.period_length
        self.a_array = np.linspace(0.0, (a_max + self.trunc) * self.period,
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
        # name = self.name_switcher.get(0)
        # Exporter.save_variable(kappa, name)
        return kappa

    def calculate_kappa_ot_bct(self):
        nct = OneTimeBCT(parameters=self.parameters, a_max=self.a_max,
                         t_0_max=self.t_0_max)
        _, _, kappa = nct.calculate_kappa_minus()
        # name = self.name_switcher.get(1)
        # Exporter.save_variable(kappa, name)
        return kappa

    def calculate_kappa_re_bct(self):
        nct = RecursiveBCT(parameters=self.parameters, a_max=self.a_max,
                           t_0_max=self.t_0_max)
        _, _, kappa = nct.calculate_kappa_minus()
        name = self.name_switcher.get(4)
        Exporter.save_variable(kappa, name)
        return kappa

    def calculate_kappa_ot_fct(self):
        nct = OneTimeFCT(parameters=self.parameters, n_gen=3, trunc=3,
                         a_max=self.a_max, t_0_max=self.t_0_max)
        _, _, kappa = nct.calculate_kappa_plus()
        name = self.name_switcher.get(2)
        Exporter.save_variable(kappa, name)
        return kappa

    def calculate_kappa_re_fct(self):
        nct = RecursiveFCT(parameters=self.parameters, n_gen=3, trunc=3,
                           a_max=self.a_max, t_0_max=self.t_0_max)
        _, _, kappa = nct.calculate_kappa_plus()
        name = self.name_switcher.get(5)
        Exporter.save_variable(kappa, name)
        return kappa

    def calculate_kappa_ot_lct(self):
        nct = OneTimeLCT(parameters=self.parameters, n_gen=3, trunc=3,
                         a_max=self.a_max, t_0_max=self.t_0_max)
        _, _, kappa = nct.calculate_kappa()
        name = self.name_switcher.get(3)
        Exporter.save_variable(kappa, name)
        return kappa

    def calculate_kappa_re_lct(self):
        nct = RecursiveLCT(parameters=self.parameters, n_gen=3, trunc=3,
                           a_max=self.a_max, t_0_max=self.t_0_max)
        _, _, kappa = nct.calculate_kappa()
        name = self.name_switcher.get(6)
        Exporter.save_variable(kappa, name)
        return kappa

    def build_vector(self, w):

        F_sum = np.zeros(self.period_length)
        for i in range(0, self.period_length):
            F = np.zeros(self.period_length)

            # First part: from 0 to t (from 0 to i-1)
            for j in range(0, max(0, i)):
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
            for j in range(max(0, i), self.period_length):
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
        self.logger.info('Iteration %s, ev: %s', self.optimizer_iteration,
                         str(ev))

        if tracing_type != 0:
            self.optimizer_iteration += 1

        return ew, ev

    def get_eigenpair(self):
        # ev = np.zeros_like(self.t_0_array)
        # ev[1] = 1
        # ev[3] = 1
        # ev = ev / np.linalg.norm(ev)
        ev = np.random.rand(self.period_length)
        ew = 0

        error = 10
        iterations = 0
        while error > 1e-8 or iterations < 5:
            z = self.build_vector(ev)  # np.matmul(A, ev)  # A  * ev
            ev = z / np.linalg.norm(z)
            Aev = self.build_vector(ev)
            ew = np.transpose(ev).dot(Aev)
            error = np.linalg.norm(Aev - ew * ev)
            iterations += 1
        return ew, ev


def main():
    T = 7
    # min case 1
    #                   1      2      3      4      5      6      7      8
    # beta0 = np.array([1e-60, 1e-60, 1e-60, 1e-60, 1e-60, 1e-60, 1e-60, 1e-60,
    #                    1e-60, 1e-60, 1e-60, 1e-60, 1e-60, 1e-60, 1e-60, 1e-60,
    #                    1e-60, 1e-60, 1e-60, 1e-60,  1e-60, 1e-60, 1e-60,
    #                    1e-60, 1e-60, 1e-60,25.04890, 1e-60])

    # max case 1
    beta0 = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
                      0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
                      0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
                      0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
                      0.8, 0.8, 0.8, 0.8])

    # min case 2
    #                   1      2      3      4      5      6      7      8
    # beta0 = np.array([1e-60, 1e-60, 1e-60, 1e-60, 1e-60, 1e-60, 1e-60, 1e-60,
    #                   1e-60, 1e-60, 1e-60, 1e-60, 100, 1e-60,  1e-60,  1e-60,
    #                   1e-60, 1e-60,  1e-60, 1e-60, 1e-60, 1e-60, 1e-60, 1e-60,
    #                   1e-60, 1e-60, 1e-60, 1e-60])

    # max case 2
    # beta0 = np.array([1.00859, 1.00859, 1.00859, 1.00859, 1.00859, 1.00859,
    #                   1.00859, 1.00859, 1.00859, 1.00859, 1.00859, 1.00859,
    #                   1.00859, 1.00859, 1.00859, 1.00859, 1.00859, 1.00859,
    #                   1.00859, 1.00859, 1.00859, 1.00859, 1.00859, 1.00859,
    #                   1.00859, 1.00859, 1.00859, 1.00859])

    par = TestParameters1(beta0, p=1/3, h=0.25, period_time=T)

    logger = logging.getLogger('rep_num_test')
    formatter = logging.Formatter('%(asctime)s %(message)s',
                                  '%m/%d/%Y %I:%M:%S %p')
    fh = logging.FileHandler('/home/saitel/TUM/Thesis/Code/rep_num_test.log')
    # fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    rnc = ReproductionNumberCalculator(logger, par, a_max=2, t_0_max=2)
    ew, ev = rnc.calculateReproductionNumber(beta0, 0)
    return ew, ev


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    ew, ev = main()
