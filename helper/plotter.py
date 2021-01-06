#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 09:39:08 2020

@author: saitel
"""
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.font_manager import FontProperties


class Plotter:
    @staticmethod
    def plot_3D(t_0, a, kappa, file_name, elev=10, azim=60, mx=1, my=1,
                mz=0.2):
        """
        Plot the probability of infection vs age of infection and time of
        infection. Save the image.

        Parameters
        ----------
        t_0 : array.
        a : array.
        kappa : array.
        file_name: filename without extension
        elev: elevation
        azim: rotation in degrees
        mx: multiple for ticks in the x axis
        my: multiple for ticks in the y axis
        mz: multiple for ticks in the z axis

        Returns
        ----------
        None.
        """

        font = FontProperties()
        font.set_family('serif')
        font.set_name('Palatino')
        font.set_size(12)

        fig = plt.figure(figsize=(9.8, 9.6))
        ax = fig.gca(projection='3d')
        ax.view_init(elev, azim)
        ax.grid(False)
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')
        ax.set_ylabel('age of infection', fontproperties=font)
        ax.set_xlabel('time of infection', fontproperties=font)
        ax.set_zlabel('probability of infection', fontproperties=font)
        ax.xaxis.set_major_locator(MultipleLocator(mx))
        ax.yaxis.set_major_locator(MultipleLocator(my))
        ax.zaxis.set_major_locator(MultipleLocator(mz))
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.plot_wireframe(X=t_0, Y=a, Z=kappa, color='black', linewidth=1,
                          cstride=20, rstride=20)

        plt.savefig(file_name + '.pdf', bbox_inches='tight')

    def plot_2D(v, kappa, xlabel, file_name):
        plt.figure
        plt.plot(v, kappa, marker='*')
        plt.xlabel(xlabel)
        plt.ylabel('probability of infection')
        plt.title('probability of infection vs ' + xlabel)
        plt.show()
        plt.savefig(file_name + '.pdf', bbox_inches='tight', frameon=False)
