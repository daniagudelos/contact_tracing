#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:33:24 2020

@author: saitel
"""
import os
from spyder_kernels.utils.iofuncs import load_dictionary
from spyder_kernels.utils.iofuncs import save_dictionary


class Exporter():
    @staticmethod
    def load_dictionary(data_name):
        filepath = os.path.join(os.path.expanduser('~'), 'TUM', 'Thesis',
                                'Code', 'data', data_name)
        data, _ = load_dictionary(filepath + '.spydata')
        return data

    @staticmethod
    def save_dictionary(data, data_name):
        filepath = os.path.join(os.path.expanduser('~'), 'TUM', 'Thesis',
                                'Code', 'data', data_name)
        save_dictionary(data, filepath + '.spydata')

    @staticmethod
    def save_variable(variable, variable_name):
        data = {variable_name: variable}
        filepath = os.path.join(os.path.expanduser('~'), 'TUM', 'Thesis',
                                'Code', 'data', variable_name)
        save_dictionary(data, filepath + '.spydata')

    @staticmethod
    def load_variable(variable_name):
        filepath = os.path.join(os.path.expanduser('~'), 'TUM', 'Thesis',
                                'Code', 'data', variable_name)
        data, _ = load_dictionary(filepath + '.spydata')
        return data.get(variable_name)
