#!/usr/bin/env python
import numpy as np
from math import sqrt

def calculateMAE(targets_list, predictions_list):
    if len(predictions_list) != len(targets_list):
        raise Exception('Error: number of elements do not match.')
    return sum(map(lambda t:float(t[0]-t[1]),zip(predictions_list, targets_list)))/len(targets_list)

def calculateRMSE(targets_list, predictions_list):
    if len(predictions_list) != len(targets_list):
        raise Exception('Error: number of elements do not match.')
    predictions = np.array(predictions_list)
    targets = np.array(targets_list)
    n = len(predictions)
    return np.linalg.norm(predictions - targets) / np.sqrt(n)
    #return np.sqrt(((predictions - targets) ** 2).mean())
