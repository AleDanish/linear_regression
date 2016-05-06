#!/usr/bin/env python
import numpy as np

def calculateMAE(targets_list, predictions_list):
    if len(predictions_list) != len(targets_list):
        raise Exception('Error: number of elements do not match.')
    error = 0
    for target, pred in zip(targets_list, predictions_list):
        error += abs(target - pred)
    return error/len(targets_list)

def calculateRMSE(targets_list, predictions_list):
    if len(predictions_list) != len(targets_list):
        raise Exception('Error: number of elements do not match.')
    predictions = np.array(predictions_list)
    targets = np.array(targets_list)
    n = len(predictions)
    return np.linalg.norm(predictions - targets) / np.sqrt(n)
