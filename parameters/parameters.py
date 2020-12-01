#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:49:44 2020

@author: saitel
"""
from math import sin


class Parameters:
    pass


class ConstantParameters(Parameters):

    def beta(self, a, t):
        return 2.2

    def mu(self, a):
        return 1 / 3.47

    def sigma(self, a, t):
        return 1 / 3.47

    def p(self):
        return 0.5

    def h(self):
        return 0.005


class VariableParameters(Parameters):
    def beta1(self, a):
        f = a/4 * (6-a)
        if f >= 0:
            return f
        return 0

    def beta2(self, a, t):
        return sin(5*t) + 1

    def beta(self, a, t):
        return self.beta1(a) * self.beta2(a, t)

    def mu(self, a):
        f = a/5 * (7-a)
        if f >= 0:
            return f
        return 0

    def sigma1(self, a):
        f = a/4 * (7-a)
        if f >= 0:
            return f
        return 0

    def sigma2(self, a, t):
        return sin(3 * t) + 1

    def sigma(self, a, t):
        return self.sigma1(a) * self.sigma2(a, t)

    def p(self):
        return 0.5

    def h(self):
        return 0.005
