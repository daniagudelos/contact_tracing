#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 17:38:14 2020

@author: saitel
"""
import numpy as np
from parameters import parameters
from periodic.backward_tracing import one_time as ot
from helper.plotter import Plotter


def constant_parameters_test():
    """
    Test one time backward tracing using constant parameters

    Returns
    -------
    None.

    """
    pars = parameters.ConstantParameters()
    bct = ot.OneTimeBCT(pars)

    a_max = 2
    t_0_max = 6
    h = pars.get_h

    t_0 = np.arange(0, t_0_max + h(), h())
    a = np.arange(0, a_max + h(), h())

    kappa = bct.calculate_kappa(a, t_0)
    a, t_0 = np.meshgrid(a, t_0)

    # Plot results
    Plotter.plot_3D(t_0, a, kappa,
                    '../figures/periodicity/backward_ct_ot_constant_pars')

    return t_0, a, kappa


def constant_parameters_p0_test():
    """
    Test one time backward tracing using constant parameters

    Returns
    -------
    None.

    """
    pars = parameters.ConstantParameters(p=0)
    bct = ot.OneTimeBCT(pars)

    a_max = 2
    t_0_max = 6
    h = pars.get_h

    t_0 = np.arange(0, t_0_max + h(), h())
    a = np.arange(0, a_max + h(), h())

    kappa = bct.calculate_kappa(a, t_0)
    a, t_0 = np.meshgrid(a, t_0)

    # Plot results
    Plotter.plot_3D(t_0, a, kappa,
                    '../figures/periodicity/backward_ct_ot_constant_pars_p0')

    return t_0, a, kappa


def constant_parameters_p1_test():
    """
    Test one time backward tracing using constant parameters

    Returns
    -------
    None.

    """
    pars = parameters.ConstantParameters(p=1)
    bct = ot.OneTimeBCT(pars)

    a_max = 2
    t_0_max = 6
    h = pars.get_h

    t_0 = np.arange(0, t_0_max + h(), h())
    a = np.arange(0, a_max + h(), h())

    kappa = bct.calculate_kappa(a, t_0)
    a, t_0 = np.meshgrid(a, t_0)

    # Plot results
    Plotter.plot_3D(t_0, a, kappa,
                    '../figures/periodicity/backward_ct_ot_constant_pars_p1')

    return t_0, a, kappa


#constant_parameters_p0_test()
#constant_parameters_test()
constant_parameters_p1_test()