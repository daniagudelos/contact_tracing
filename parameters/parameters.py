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

    def __init__(self, beta=2.2, mu=1/3.47, sigma=1/3.47, p=0.5, h=0.005):
        self.beta = beta
        self.mu = mu
        self.sigma = sigma
        self.p = p
        self.h = h

    def get_beta(self, a, t):
        return self.beta

    def get_mu(self, a):
        return self.mu

    def get_sigma(self, a, t):
        return self.sigma

    def get_p(self):
        return self.p

    def get_h(self):
        return self.h


class VariableParameters(Parameters):
    def __init__(self, p=0.5, h=0.005):
        self.p = p
        self.h = h

    def get_beta1(self, a):
        f = a/4 * (6-a)
        if f >= 0:
            return f
        return 0

    def get_beta2(self, a, t):
        return sin(5*t) + 1

    def get_beta(self, a, t):
        return self.get_beta1(a) * self.get_beta2(a, t)

    def get_mu(self, a):
        f = a/5 * (7-a)
        if f >= 0:
            return f
        return 0

    def get_sigma1(self, a):
        f = a/4 * (7-a)
        if f >= 0:
            return f
        return 0

    def get_sigma2(self, a, t):
        return sin(3 * t) + 1

    def get_sigma(self, a, t):
        return self.get_sigma1(a) * self.get_sigma2(a, t)

    def get_p(self):
        return self.p

    def get_h(self):
        return self.h
