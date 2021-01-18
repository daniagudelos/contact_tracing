#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 11:10:09 2021

@author: saitel
"""
import numpy as np
from periodic.no_contact_tracing import NoCT
from parameters import parameters
from helper.plotter import Plotter


def kappa_hat_test(pars, filename, a_max=2, t_0_max=6):
    nct = NoCT(pars)
    t_0, a, kappa_hat = nct.get_kappa_hat(4, 4)
    t_0, a = np.meshgrid(t_0, a)
    Plotter.plot_3D(t_0, a, kappa_hat,
                    '../figures/no_ct/' + filename + '_60_10', my=0.5)
    Plotter.plot_3D(t_0, a, kappa_hat,
                    '../figures/no_ct/' + filename + '_n60_10', azim=-60,
                    my=0.5)
    return t_0, a, kappa_hat


def main():
    print('Running test NCT with variable parameters')
    t_0, a, kappa_hat = kappa_hat_test(parameters.VariableParameters(h=0.05),
                                       'nct_variable')
    return t_0, a, kappa_hat


if __name__ == '__main__':
    t_0, a, kappa_hat = main()
