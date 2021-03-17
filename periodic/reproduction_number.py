#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:43:24 2021

@author: saitel
"""
import numpy as np
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
    def __init__(self, parameters, a_max, t_0_max, trunc=2):
        # Initial parameters. They are replaced later.
        self.optimizer_iteration = 0
        self.parameters = parameters
        self.h = self.parameters.get_h()
        self.p = self.parameters.get_p()
        self.period = self.parameters.get_period()
        self.period_length = self.parameters.get_period_length()
        self.beta = self.parameters.get_beta
        self.trunc = trunc
        self.trunc_length = int(round(self.trunc / self.h, 1))
        self.a_max = a_max + (self.trunc + 2) * self.period
        self.a_length = int(round(self.a_max / self.h, 1))
        self.t_0_max = t_0_max
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
        print('Calculating kappa - type: ', tracing_type)
        func = self.switcher.get(tracing_type)
        name = (self.name_switcher.get(tracing_type) + '_it' +
                str(self.optimizer_iteration))
        kappa = func()

        if tracing_type != 0:
            Exporter.save_variable(kappa, name)
        return kappa

    def calculate_kappa_nct(self):
        nct = NoCT(parameters=self.parameters, a_max=self.a_max,
                   t_0_max=self.t_0_max)
        _, _, kappa = nct.calculate_kappa_hat()
        return kappa

    def calculate_kappa_ot_bct(self):
        nct = OneTimeBCT(parameters=self.parameters, a_max=self.a_max,
                         t_0_max=self.t_0_max)
        _, _, kappa = nct.calculate_kappa_minus()
        return kappa

    def calculate_kappa_re_bct(self):
        nct = RecursiveBCT(parameters=self.parameters, a_max=self.a_max,
                           t_0_max=self.t_0_max)
        _, _, kappa = nct.calculate_kappa_minus()
        return kappa

    def calculate_kappa_ot_fct(self):
        nct = OneTimeFCT(parameters=self.parameters, n_gen=4, trunc=10,
                         a_max=self.a_max, t_0_max=self.t_0_max)
        _, _, kappa = nct.calculate_kappa_plus()
        return kappa

    def calculate_kappa_re_fct(self):
        nct = RecursiveFCT(parameters=self.parameters, n_gen=4, trunc=10,
                           a_max=self.a_max, t_0_max=self.t_0_max)
        _, _, kappa = nct.calculate_kappa_plus()
        return kappa

    def calculate_kappa_ot_lct(self):
        nct = OneTimeLCT(parameters=self.parameters, n_gen=4, trunc=10,
                         a_max=self.a_max, t_0_max=self.t_0_max)
        _, _, kappa = nct.calculate_kappa()
        return kappa

    def calculate_kappa_re_lct(self):
        nct = RecursiveLCT(parameters=self.parameters, n_gen=4, trunc=10,
                           a_max=self.a_max, t_0_max=self.t_0_max)
        _, _, kappa = nct.calculate_kappa()
        return kappa

    def build_vector(self, w):
        F_sum = np.zeros(self.period_length + 1)
        for i in range(0, self.period_length + 1):
            F = np.zeros(self.period_length + 1)

            # First sum: from 0 to i-1
            for j in range(0, max(0, i - 2)):
                H_sum = 0
                H = np.zeros((self.trunc + 1))
                index = i - j  # This is always +
                for n in range(0, self.trunc + 1):
                    H[n] = (self.beta(self.a_array[index] + n * self.period,
                                      self.t_array[i]) *
                            self.kappa[i, index + n * self.period_length])
                H_sum = np.sum(H)
                F[j] = H_sum * w[j]

            #  Second sum: from i-1 to N-1
            for j in range(max(0, i - 1), self.period_length + 1):
                H_sum = 0
                H = np.zeros((self.trunc + 1))
                index = i - j
                while(index < 0):
                    index = index + self.period_length
                for n in range(0, self.trunc + 1):
                    H[n] = (self.beta(self.a_array[index] +
                                      (n + 1) * self.period, self.t_array[i]) *
                            self.kappa[i,
                                       index + (n + 1) * self.period_length])
                H_sum = np.sum(H)
                F[j] = H_sum * w[j]
            F_sum[i] = np.sum(F)
        u = self.period * F_sum * self.h
        return u

    def calculateReproductionNumber(self, beta, tracing_type):
        # Exporter.save_variable(beta, 'beta_it' + str(self.optimizer_iteration))
        print(beta)
        self.parameters = TestParameters1(beta)
        self.beta = self.parameters.get_beta
        self.kappa = self.get_kappa(tracing_type)

        ew, ev = self.get_eigenpair()
        print('Itetation ', self.optimizer_iteration, ', ew: ', ew)

        if tracing_type != 0:
            self.optimizer_iteration += 1

        return ew

    def get_eigenpair(self):
        print('Calculating reproduction number!')
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


def main():
    T = 7

    beta2 = np.array([4.03180775, 3.32875495, 5.61445947, 3.27064162,
                        4.12945533, 3.56094193,
                        2.34989042, 8.170785,   4.79047693, 7.93459138, 
                        5.41256494, 6.58981984,
                        6.46465104, 8.09681395, 6.97391276, 0.95409398,
                        5.5625838,  2.77441899,
                        7.63343152, 2.99323211, 6.32643309, 0.91895196,
                        8.72991048, 8.46562939,
                        2.53669335, 1.63826452, 1.02829687, 0.45548419])
    beta1 = np.array([1, 1, 1, 1, 3, 3, 3, 3, 3.5, 3.5, 3.5, 3.5, 4, 4, 4, 4,
                      3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1])
    par = TestParameters1(beta1, p=1/3, h=0.25, period_time=T)
    rnc = ReproductionNumberCalculator(par, a_max=2 * T, t_0_max=2 * T,
                                       trunc=4)
    ew1 = rnc.calculateReproductionNumber(beta2, 0)
    return ew1


if __name__ == '__main__':
    ew1 = main()
    print('ew1 ', ew1)
