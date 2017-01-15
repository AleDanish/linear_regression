#!/usr/bin/env python
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import numpy as np
from svmutil import *
import subprocess
import math

def getResults(p_acc):
    error = ''
    correlation = ''
    for i, var in  enumerate(p_acc):
        if i == len(p_acc) - 1:
            correlation = math.fabs(var)
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

def main(filename):
    filename_scale = filename + '.scale'
    subprocess.call('svm-scale -l -1 -u 1 ' + filename + ' > ' + filename_scale, shell=True) 
    print ('Created file' + filename_scale + ' with normalized data')
    y, x = svm_read_problem(filename_scale)
    #prob  = svm_problem(y, x)

    max_correlation = 0
    max_correlation_error = 0
    best_conditions_corr = ''
    best_results_corr = [['0' for i in range(3)] for j in range(2)]
    prediction_corr = [[] for i in range(2)]

    min_error = 1000000
    min_error_corr = 0
    best_condition_error = ''
    best_results_error = [['1000000' for i in range(3)] for j in range(2)]
    prediction_error = [[] for i in range(2)]

    line_number = sum(1 for line in open(filename))
    sample_num = int(line_number * 0.9)
    print('lines: ',line_number)
    print('samples: ',sample_num)

    for t in range(1, 3):
        name = getFunction(t)
        print('Started training and prevision for ' + name)
        file = open(path.dirname(path.abspath(__file__)) + '/output/' + name, 'w')
        file.write('### EPSILON-SRV ' + name + ' ###\n')
        e_arr = np.arange(0.1, 1, 0.5)
        p_arr = np.arange(0.1, 1, 0.5)
        if t ==3:
            g_arr = [1]
            r_arr = np.arange(1, 10, 5)
            d_arr = np.arange(1, 10, 5)
        else:
            g_arr = np.arange(1, 10, 5)
            r_arr = np.arange(1, 10, 5)
            d_arr = np.arange(1, 10, 5)
        for c in range (1, 100, 30):
            for e in e_arr:
                for p in p_arr:
                    for r in r_arr:
                        for d in d_arr:
                            for g in g_arr:
                                conditions = '-t ' + str(t) + ' -c ' + str(c) + ' -e ' + str(e) + ' -p ' + str(p) + ' -r ' + str(r)
                                if t != 3:
                                    conditions += ' -g ' + str(g)
                                #param = svm_parameter('-q -s 3 ' + conditions)
                                param = '-q -s 3 ' + conditions
                                m = svm_train(y[:sample_num], x[:sample_num], param+' -q')
                                p_label, p_acc, p_val = svm_predict(y[sample_num:], x[sample_num:], m)
                                ##print('label:', p_label, ' p_acc:', p_acc, ' p_val:', p_val)
                                error, correlation = getResults(p_acc)
                                results = 'mean squared error:' + str(error) + ' - correlation:' + str(correlation) + '\n'
                                file.write(conditions + ": " + results)
                                if float(best_results_corr[t-1][2]) < correlation:
                                    best_results_corr[t-1][0] = conditions
                                    best_results_corr[t-1][1] = str(error)
                                    best_results_corr[t-1][2] = str(correlation)
                                    prediction_corr[t-1] = p_label
                                if float(best_results_error[t-1][1]) > error:
                                    best_results_error[t-1][0] = conditions
                                    best_results_error[t-1][1] = str(error)
                                    best_results_error[t-1][2] = str(correlation)
                                    prediction_error[t-1] = p_label
                                if max_correlation < correlation:
                                    max_correlation = correlation
                                    max_correlation_error = error
                                    best_condition_corr = conditions
                                if min_error > error:
                                    min_error = error
                                    min_error_corr = correlation
                                    best_condition_error = conditions
                                if error >= 1:
                                    break;
        print('Finished training and prevision for' +  name)
        file.close()
    print('----------------------------------------------------------------------------------------------------------------------------')
    for t in range(1, 3):
        print('EPSILON-SRV', getFunction(t))
        print('BEST CORRELATION - parameters:', best_results_corr[t-1][0] , '- error:', best_results_corr[t-1][1], '- correlation:', best_results_corr[t-1][2])
        print('BEST ERROR - parameters: ', best_results_error[t-1][0], '- error: ', best_results_error[t-1][1], '- correlation: ', best_results_error[t-1][2], '\n')
        print('Best correlation conditions:', best_condition_corr)
        print('Max correlation:', max_correlation)
        print('Best error conditions:', best_condition_error)
        print('Min erorr:', min_error)
        print('----------------------------------------------------------------------------------------------------------------------------')

    # K-steps ahead prevision
    real_values = y[sample_num:]
    print('Real values:', real_values)

    print('PREDICTION ONE-STEP AHEAD - Polynomial')
    param = best_results_corr[0][0]
    m = svm_train(y[:sample_num], x[:sample_num], param + ' -q')
    p_label, p_acc, p_val = svm_predict(y[sample_num:], x[sample_num:], m)
    print('results: ', p_label)

    print('PREDICTION ONE-STEP AHEAD - Gaussian')
    param = best_results_corr[1][0]
    m = svm_train(y[:sample_num], x[:sample_num], param + ' -q')
    p_label, p_acc, p_val = svm_predict(y[sample_num:], x[sample_num:], m)
    print('results: ', p_label)

    return p_label[-1]
