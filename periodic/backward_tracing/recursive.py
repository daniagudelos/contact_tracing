#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 23:54:35 2020

@author: saitel
"""
import numpy as np
import threading
import time


class RecursiveBCT:
    def __init__(self, parameters):
        self.beta = parameters.get_beta
        self.dbeta = parameters.get_dbeta
        self.mu = parameters.get_mu
        self.sigma = parameters.get_sigma
        self.p = parameters.get_p
        self.h = parameters.get_h
        self.cycle = 0
        self.cycles = 0
        self.progress = 0
        self.interrupted = False

    def update_progress(self):
        while self.cycle != self.cycles and self.interrupted is False:
            time.sleep(5)
            print('Cycle: {} / {}, Progress: {:.2f}%'.format(self.cycle,
                                                             self.cycles,
                                                             self.progress))

    def integral(self, dkappa, kappa, a, a_index, t_0, t_0_index):

        if a_index == 0:
            raise Exception('integral should not be called before a[1].')

        result = 0.5 * (kappa[0, t_0_index + a_index] *
                        self.dbeta(a[a_index], t_0[t_0_index] + a[a_index]) -
                        self.beta(a[a_index], t_0[t_0_index] + a[a_index]) *
                        kappa[0, t_0_index + a_index] *
                        self.mu(a[0]) +
                        kappa[a_index, t_0_index] *
                        self.dbeta(0, t_0[t_0_index]) -
                        self.beta(0, t_0[t_0_index]) *
                        kappa[a_index, t_0_index] *
                        self.mu(a[a_index]))

        # [1, a_index - 1]
        for i in range(1, a_index, 1):
            beta_v = self.beta(a[a_index - i], t_0[t_0_index] + a[a_index - i])
            dbeta_v = self.dbeta(a[a_index - i], t_0[t_0_index] + a[a_index - i])
            kappa_v = kappa[i, t_0_index + a_index - i]
            mu_v = self.mu(a[i])
            result = result + kappa_v * dbeta_v - beta_v * kappa_v * mu_v

        return self.h() * result

    def calculate_kappa(self, a_max, t_0_max):

        t_0 = np.arange(0, t_0_max + a_max + self.h(), self.h())
        a = np.arange(0, a_max + self.h(), self.h())

        kappa = np.zeros((len(a), len(t_0)))
        dkappa = np.zeros((len(a), len(t_0)))

        # initialize cycle for a = 0
        for j in range(0, len(t_0)):
            kappa[0, j] = 1
            dkappa[0, j] = -(self.mu(a[0]) + self.sigma(a[0], t_0[j] + a[0]))

        self.cycles = len(a) - 1

        try:
            x = threading.Thread(target=self.update_progress)
            x.start()


            for i in range(1, len(a)):
                self.cycle = i
                count = 0
                total = len(t_0) - i
                for j in range(0, len(t_0) - i):
                    count = count + 1
                    self.progress = count / total * 100

                    # kappa using euler method
                    kappa[i, j] = kappa[i-1, j] + self.h() * dkappa[i-1, j]

                    # dkappa using kappa[i,j]
                    temp = - self.beta(0, t_0[j]) * kappa[i, j] + \
                        self.beta(a[i], t_0[j] + a[i]) * kappa[0, j + i] + \
                            self.integral(dkappa, kappa, a, i, t_0, j)
                    dkappa[i, j] = - kappa[i, j] * (
                        self.mu(a[i]) + self.sigma(a[i], t_0[j] + a[i]) +
                        self.p() * temp)

            t_0_max_index = np.where(t_0 == t_0_max)[0][0]

            x.join(2)

        except KeyboardInterrupt:
            self.interrupted = True
            print("Error: stopping the program")

        return t_0[0:(t_0_max_index + 1)], a, kappa[:, 0:(t_0_max_index + 1)],\
            dkappa[:, 0:(t_0_max_index + 1)]