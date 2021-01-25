#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:43:24 2021

@author: saitel
"""
import scipy.integrate as integrate
import numpy as np
from parameters import parameters
from periodic.no_contact_tracing import NoCT


class ReproductionNumberCalculator:
    def __init__(self):
        pass

    def test_funct(self, a, t_0):
        print('required: ', a)
        return - a * a + t_0

    def calculateRo(self):
        t_0 = 0
        nct = NoCT(parameters.VariableParameters(h=0.05), 1, 1)
        integral = integrate.quad(lambda a: self.beta(a, t_0 + a) * \
                                  nct.calculate_kappa_hat_point(a, t_0) *
                                  nct.calculate_kappa_hat_point(0, t_0)
        w_1 = nct.calculate_kappa_hat_point(a, t_0)
        ro = integral / w_1

        return integral


rnc = ReproductionNumberCalculator()
rnc.calculateRo()
