#!/usr/bin/env python
if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

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

import numpy as np
from svmutil import *
import subprocess
filename = 'temperature_data'
filename_scale = filename + '.scale'
subprocess.call('svm-scale -l -1 -u 1 ' + filename + ' > ' + filename_scale, shell=True) 
print 'Created file', filename_scale, 'with normalized data'

y, x = svm_read_problem(filename_scale)
prob  = svm_problem(y, x)

max_correlation = 0
max_correlation_error = 0
best_conditions_corr = ''
best_results_corr = [['0' for i in range(3)] for j in range(4)]
prediction_corr = [[] for i in range(4)]

min_error = 1000000
min_error_corr = 0
best_condition_error = ''
best_results_error = [['1000000' for i in range(3)] for j in range(4)]
prediction_error = [[] for i in range(4)]

sample_num = 500

for t in range(0, 1):
    name = getFunction(t)
    print 'Started training and prevision for', name
    file = open('output/' + name, 'w')
    file.write('### EPSILON-SRV ' + name + ' ###\n')
    e_arr = np.arange(0.1, 1, 0.3)
    p_arr = np.arange(0.1, 1, 0.3)
    if t == 0:
        g_arr = [1]
        r_arr = [1]
        d_arr = [1]
    elif t ==3:
        g_arr = [1]
        r_arr = np.arange(1, 10, 3)
        d_arr = np.arange(1, 10, 3)
    else:
        g_arr = np.arange(1, 10, 3)
        r_arr = np.arange(1, 10, 3)
        d_arr = np.arange(1, 10, 3)
    for c in range (1, 100, 10):
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
                            m = svm_train(y[:sample_num], x[:sample_num], param)
                            p_label, p_acc, p_val = svm_predict(y[sample_num:], x[sample_num:], m)
                            error, correlation = getResults(p_acc)
                            results = 'mean squared error:' + str(error) + ' - correlation:' + str(correlation) + '\n'
                            file.write(conditions + ": " + results)
                            if float(best_results_corr[t][2]) < correlation:
                                best_results_corr[t][0] = conditions
                                best_results_corr[t][1] = str(error)
                                best_results_corr[t][2] = str(correlation)
                                prediction_corr[t] = p_label
                            if float(best_results_error[t][1]) > error:
                                best_results_error[t][0] = conditions
                                best_results_error[t][1] = str(error)
                                best_results_error[t][2] = str(correlation)
                                prediction_error[t] = p_label
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
    print 'Finished training and prevision for: ', name
    file.close()

# K-step ahead prevision
rw = []
for i in range(sample_num, len(y), 1):
    rw.append(y[sample_num])
print 'Predict k-step with Random-Walk: ', rw

for t in range(0, 1):
    print getFunction(t) + ' - best correlation prediction'
    print 'parameters: ', best_results_corr[t][0]
    print 'error: ', best_results_corr[t][1]
    print 'correlation: ', best_results_corr[t][2]
    print 'prediction: ', prediction_corr[t]
    print getFunction(t) + ' - best error prediction'
    print 'parameters: ', best_results_error[t][0]
    print 'error: ', best_results_error[t][1]
    print 'correlation: ', best_results_error[t][2]
    print 'prediction: ', prediction_error[t]
    print '\n'

# One-step ahead prevision
file = open('output/results', 'w')
for num in range(sample_num, len(y), 1):
    num_rw = y[sample_num]
    print '### Random Walk ###'
    print 'next value: ', num_rw
    file.write('### RANDOM WALK ###\n')
    file.write('next value: ' + str(num_rw) + '\n')
    for t in range(0, 1):
        param = '-q -s 3 ' + best_results_error[t][0]
        m = svm_train(y[:num], x[:num], param)
        X=[x[num]]
        Y=[y[num]]
        p_label, p_acc, p_val = svm_predict(Y, X, m)
        error, correlation = getResults(p_acc)
        print '### EPSILON-SRV ' + getFunction(t) + ' ###\n'
        print 'error: ', error, ' - correlation: ', correlation, ' - next value: ', p_label
        file.write('### EPSILON-SRV ' + getFunction(t) + ' ###\n')
        file.write('error: ' + str(error) + ' - correlation: ' + str(correlation) + ' - next value: ' + str(p_label))
file.close()

print "best correlation conditions: ", best_condition_corr
print "max correlation: ", max_correlation
print "best error conditions: ", best_condition_error
print "min erorr: ", min_error
