#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 19:05:06 2021

@author: saitel
"""
from periodic.backward_tracing.recursive_bct import RecursiveBCT
from periodic.forward_tracing.recursive_fct import RecursiveFCT
from parameters.parameters import TestParameters1, TestParameters2
import numpy as np
from helper.plotter import Plotter


class RecursiveLCT():
    def __init__(self, parameters, n_gen, trunc, a_max, t_0_max):
        self.parameters = parameters
        self.bct = RecursiveBCT(self.parameters, a_max, t_0_max)
        self.fct = RecursiveFCT(parameters, n_gen, a_max, t_0_max)
        self.t_0_array = None
        self.a_array = None
        self.f = []

    def calculate_kappa(self):
        """
            Returns an array with kappa_plus
        """
        self.t_0_array, self.a_array, kappa_minus = self.bct.calculate_kappa_minus()
        f_plus = self.fct.calculate_f_plus()
        return (self.t_0_array, self.a_array, kappa_minus * f_plus)


def recursive_lct_test(pars, filename, a_max=2, t_0_max=6):
    otlct = RecursiveLCT(pars, n_gen=5, trunc=10, a_max=a_max, t_0_max=t_0_max)
    t_0_array, a_array, kappa_re_lct = otlct.calculate_kappa()
    a, t_0 = np.meshgrid(a_array, t_0_array)
    mx = round(t_0_max * pars.get_period() / 10)
    my = round(a_max * pars.get_period() / 10)
    Plotter.plot_3D(t_0, a, kappa_re_lct, filename + '_60_10', mx=mx, my=my)
    Plotter.plot_3D(t_0, a, kappa_re_lct, filename + '_n60_10', azim=-60,
                    mx=mx, my=my)
    return t_0_array, a_array, kappa_re_lct


def main():
    T = 7  # days
    beta1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    par = TestParameters1(beta1, p=1/3, h=0.25, period_time=T)
    t_0_array, a_array, kappa_re_lct = recursive_lct_test(
        par, '../../figures/periodic/lct_re_test1_p03', a_max=2,
        t_0_max=2)
    return t_0_array, a_array, kappa_re_lct


def main2():
    T = 7  # days
    beta2 = np.array([1, 1, 1, 1, 3, 3, 3, 3, 3.5, 3.5, 3.5, 3.5, 4, 4, 4, 4,
                      3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1])
    par = TestParameters2(beta2, p=1/3, h=0.25, period_time=T)
    t_0_array, a_array, kappa_re_lct = recursive_lct_test(
        par, '../../figures/periodic/lct_re_test2_p03', a_max=2,
        t_0_max=2)
    return t_0_array, a_array, kappa_re_lct


if __name__ == '__main__':
    t_0_array, a_array, kappa_re_lct = main2()
