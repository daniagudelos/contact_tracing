#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:33:24 2020

@author: saitel
"""
from spyder_kernels.utils.iofuncs import load_dictionary
from spyder_kernels.utils.iofuncs import save_dictionary


class Exporter():
    @staticmethod
    def load(filepath):
        data = load_dictionary(filepath + '.spydata')
        return data

    @staticmethod
    def save(t_0, a, kappa, dkappa, filepath):
        data = {'t_0': t_0,
                'a': a,
                'kappa': kappa,
                'dkappa': dkappa}
        save_dictionary(data, filepath + '.spydata')
