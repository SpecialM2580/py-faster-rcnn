#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import re

if sys.platform.startswith('linux'):
    import matplotlib
    # matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import rcParams


def draw_loss_curve(fname, lim_min=None):

    loss_iter = []
    loss = []
    test_iter = []
    temp_test = []
    iter_value = 0

    for line in open(fname):

        if 'Iteration' in line and 'loss' in line:
            iteration_txt = re.search(ur'Iteration\s([0-9]+)', line)
            loss_txt = re.search(ur'loss\s=\s([0-9\.e-]+)\n', line)
            iter_value = int(iteration_txt.groups()[0])
            loss_value = float(loss_txt.groups()[0])

            if lim_min is not None and iter_value > lim_min:
                loss.append(loss_value)
                loss_iter.append(iter_value)
            #print line 

        if 'Testing net' in line:
            iteration_txt = re.search(ur'Iteration\s([0-9]+)', line)
            iter_value = int(iteration_txt.groups()[0])
            if lim_min is not None and iter_value > lim_min:
                test_iter.append(iter_value)

        if 'Test net output' in line and 'loss = ' in line:
            txt = re.search(ur'=\s*([0-9\.]+)\s*loss\)', line)
            if lim_min is not None and txt and iter_value > lim_min:
                temp_test.append(float(txt.groups()[0]))
        #print line
    # Set font family to serif
    rcParams['font.family'] = 'serif'
    rcParams['figure.figsize'] = (8.0,6.0)
    
    fig, ax = plt.subplots()

    ax.set_ylabel('Loss')
    ax.set_xlabel('Iterations')

    ax.plot(loss_iter, loss, 'k', label='Training', color='blue')
    #print loss_iter,loss
    if len(temp_test) == len(test_iter):
        ax.plot(test_iter, temp_test, 'k', label='Validation', color='green')
 
    legend = ax.legend(loc='upper right', shadow=True, fontsize = 'small')
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    
    #print fname 
    plt.show()

import os
import sys
if __name__ == '__main__':
    draw_loss_curve(sys.argv[1],int(sys.argv[2]))
