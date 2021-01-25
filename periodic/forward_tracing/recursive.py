#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 19:05:06 2021

@author: saitel
"""


class RecursiveFCT:
    def __init__(self, parameters):
        self.beta = parameters.get_beta
        self.mu = parameters.get_mu
        self.sigma = parameters.get_sigma
        self.p = parameters.get_p
        self.h = parameters.get_h
    
    
