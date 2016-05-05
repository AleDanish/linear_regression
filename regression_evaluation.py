#!/usr/bin/env python
import numpy as np

def calculateMAE(targets_list, predictions_list):
    if len(predictions_list) != len(targets_list):
        raise Exception('Error: number of elements do not match.')
    return sum(map(lambda t:float(t[0]-t[1]),zip(predictions_list, targets_list)))/len(targets_list)

def calculateRMSE(targets_list, predictions_list):
    if len(predictions_list) != len(targets_list):
        raise Exception('Error: number of elements do not match.')
    n = len(predictions_list)
    return np.linalg.norm(predictions_list - targets_list) / np.sqrt(n)
