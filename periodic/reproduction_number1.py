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
    def __init__(self, parameters, a_max, t_0_max, tracing_type, trunc=10):
        self.parameters = parameters
        self.h = parameters.get_h
        self.p = parameters.get_p
        self.period = parameters.get_period()
        self.period_length = parameters.get_period_length()
        self.beta = parameters.get_beta
        self.trunc = trunc
        self.trunc_length = int(round(self.trunc / self.h(), 1))
        self.a_max = a_max + (self.trunc + 2) * self.period
        self.a_length = int(round(self.a_max / self.h(), 1))
        self.t_0_max = t_0_max
        self.t_0_length = int(round(self.t_0_max / self.h(), 1))
        self.t_0_array = np.linspace(0.0, self.t_0_max, self.t_0_length + 1)
        self.t_array = np.linspace(0.0, self.t_0_max, self.t_0_length + 1)
        self.t_length = int(round(self.t_0_max / self.h(), 1))
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

        self.kappa = self.get_kappa(tracing_type)

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
        nct = OneTimeFCT(parameters=self.parameters, n_gen=4, trunc=10,
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
            for j in range(0, max(0, i - 2)):  # from 0 to i-1
                H_sum = 0
                H = np.zeros((self.trunc + 1))
                index = i - j  # This is always +
                for n in range(0, self.trunc + 1):
                    H[n] = (self.beta(self.a_array[index] + n * self.period,
                                      self.t_array[i]) *
                            self.kappa[i, index + n * self.period_length])
                H_sum = np.sum(H)
                F[j] = H_sum * w[j]

            #  Second part
            for j in range(max(0, i - 1), self.period_length + 1):  # from i-1 to N-1
                H_sum = 0
                H = np.zeros((self.trunc + 1))
                index = i - j
                while(index < 0):
                    index = index + self.period_length
                for n in range(0, self.trunc + 1):
                    H[n] = (self.beta(self.a_array[index] +
                                      (n + 1) * self.period, self.t_array[i]) *
                            self.kappa[i, index + (n + 1) * self.period_length])
                H_sum = np.sum(H)
                F[j] = H_sum * w[j]
            F_sum[i] = np.sum(F)
        u = self.period * F_sum * self.h()
        return u

    def calculateReproductionNumber(self):
        print('Calculating!')
        q = np.zeros_like(self.t_0_array)
        q[1] = 1
        q[3] = 1
        q = q / np.linalg.norm(q)
        v = 0

        error = 10
        while error > 1e-5:
            z = self.build_vector(q)  # np.matmul(A, q)  # A  * q
            q = z / np.linalg.norm(z)
            Aq = self.build_vector(q)
            v = np.transpose(q).dot(Aq)
            error = np.linalg.norm(Aq - v * q)
        return v, q


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


if __name__ == '__main__':
    beta2 = np.array([1, 1, 1, 1, 3, 3, 3, 3, 3.5, 3.5, 3.5, 3.5, 4, 4, 4, 4,
                      3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1])
    #beta1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     #                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    par = TestParameters1(beta2, p=1/3, h=0.25, period_time=7)
    ew6 = lct_test(par, 7)
    # ew4, ew5 = fct_test(par, 7)
    # ew1, ew2, ew3 = bct_test(par, 7)
