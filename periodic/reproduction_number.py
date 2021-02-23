#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:43:24 2021

@author: saitel
"""
import numpy as np
from parameters.parameters import TestParameters1
from periodic.no_contact_tracing import NoCT
from periodic.backward_tracing.one_time_bct import OneTimeBCT
from periodic.forward_tracing.one_time_fct import OneTimeFCT
from periodic.full_tracing.one_time_lct import OneTimeLCT


class ReproductionNumberCalculator:
    def __init__(self, parameters, a_max, t_0_max, tracing_type, trunc=10):
        # tracing_type: 0: no contact tracing, 1: bct, 2: fct, 3: lct
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

        switcher = {
            0: self.get_kappa_hat,
            1: self.get_kappa_minus,
            2: self.get_kappa_plus,
            3: self.get_kappa_full
        }

        func = switcher.get(tracing_type)
        self.kappa = func()

    def get_kappa_hat(self):
        nct = NoCT(parameters=self.parameters, a_max=self.a_max,
                   t_0_max=self.t_0_max)
        _, _, kappa_hat = nct.calculate_kappa_hat()
        return kappa_hat

    def get_kappa_minus(self):
        nct = OneTimeBCT(parameters=self.parameters, a_max=self.a_max,
                         t_0_max=self.t_0_max)
        _, _, kappa_minus = nct.calculate_kappa_minus()
        return kappa_minus

    def get_kappa_plus(self):
        nct = OneTimeFCT(parameters=self.parameters, n_gen=4, a_max=self.a_max,
                         t_0_max=self.t_0_max)
        _, _, kappa_plus = nct.calculate_kappa_plus()
        return kappa_plus

    def get_kappa_full(self):
        nct = OneTimeLCT(parameters=self.parameters, n_gen=4, a_max=self.a_max,
                         t_0_max=self.t_0_max)
        _, _, kappa_full = nct.calculate_kappa()
        return kappa_full

    def build_vector(self, w):

        F_sum = np.zeros(self.period_length + 1)
        for i in range(0, self.period_length):
            F = np.zeros(self.period_length)
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
            for j in range(max(0, i - 1), self.period_length):  # from i-1 to N-1
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

    def calculateRo(self):
        q = np.zeros_like(self.t_0_array)
        q[1] = 1
        q[3] = 1
        q = q / np.linalg.norm(q)
        v = 0

        for k in range(10):
            #  print('it: ', k, ' - q: ', q)
            z = self.build_vector(q)  # np.matmul(A, q)  # A  * q
            #  print('it: ', k, ' - z: ', z)
            q = z / np.linalg.norm(z)
            #  print('it: ', k, ' - new q: ', q)
            v = np.transpose(q).dot(self.build_vector(q))
            #  print('it: ', k, ' - v: ', v)
        return v


def main():
    T = 1
    beta2 = np.array([4, 3, 2, 1])
    par = TestParameters1(beta2, p=1/3, h=0.25, period_time=T)
    rnc = ReproductionNumberCalculator(par, a_max=2 * T, t_0_max=2 * T,
                                       tracing_type=0, trunc=3)
    ew1 = rnc.calculateRo()
    return ew1


if __name__ == '__main__':
    ew1 = main()
    print('ew1 ', ew1)
