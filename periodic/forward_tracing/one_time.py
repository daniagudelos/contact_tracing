#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 15:13:28 2021

@author: saitel
"""
from periodic.no_contact_tracing import NoCT
import numpy as np
import threading
import time


class OneTimeFCT:
    def __init__(self, parameters):
        self.beta = parameters.get_beta
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

    def calculate_f_zero_point(self, a_index, t_0_index):
        count = 5
        f0 = 1
        while count > 0:
            f0 = ...
                
    def calculate_kappa_plus(self, a_max, t_0_max):
        pass
