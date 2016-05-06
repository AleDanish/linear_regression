#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np

def getResults(p_acc):
    error = ''
    correlation = ''
    for i, var in  enumerate(p_acc):
        if i == len(p_acc) - 1:
            correlation = var
        if i == len(p_acc) - 2:
            error = var
    return error, correlation

def getFunction(t):
    if t == 0:
        return 'LINEAR'
    elif t == 1:
        return 'POLYNOMIAL'
    elif t == 2:
        return 'GAUSSIAN'
    elif t == 3:
        return 'SIGMOID'

def plot(labels, real_values, *values_list):
    values = list(values_list)
    one_step_ahead = []
    one_step_ahead.append(real_values)
    k_step_ahead = []
    k_step_ahead.append(real_values)
    for i in range(0, len(values)/2):
        one_step_ahead.append(values[i])
        k_step_ahead.append(values[len(values)/2 + i])
    observations = []
    for i in np.arange(1, len(real_values) + 1):
        observations.append(i)
    plt.subplot(211)
    plt.grid(True)
    plt.xlabel('observations')
    plt.ylabel('temperature (C)')
    plt.title('One-step-ahead Data Prediction')
    plt.ylim(0, 31)
    lines = []
    for lab, val in zip(labels, one_step_ahead):
        lines.extend(plt.plot(observations, val, label=lab))
    plt.subplot(212)
    plt.grid(True)
    plt.xlabel('observations')
    plt.ylabel('temperature (C)')
    plt.title('K-step-ahead Data Prediction')
    plt.ylim(0, 31)
    for lab, val in zip(labels, k_step_ahead):
        plt.plot(observations, val, label=lab)
    plt.figlegend(lines, labels, loc = 'lower center', ncol=len(labels), labelspacing=0.)
    plt.show()
