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
# Read data in LIBSVM format
y, x = svm_read_problem('input_data')
prob  = svm_problem(y, x)

max_correlation = 0
best_conditions = ''
best_results = [['0' for i in range(3)] for j in range(4)]

for t in range(0, 4):
    name = getFunction(t)
    print 'Inizio training e previsione per ', name
    file = open('output/' + name, 'w')
    file.write('### EPSILON-SRV ' + name + ' ###\n')
    e_arr = np.arange(0.1, 1, 0.2)
    p_arr = np.arange(0.1, 1, 0.2)
    if t == 0:
        g_arr = [1]
        r_arr = [1]
        d_arr = [1]
    else:
        g_arr = np.arange(1, 10, 2)
        r_arr = np.arange(1, 10, 2)
        d_arr = np.arange(1, 10, 2)
    for c in range (1, 100, 10):
        for e in e_arr:
            for p in p_arr:
                for g in g_arr:
                    for r in r_arr:
                        for d in d_arr:
                            conditions = 't=' + str(t) + ' c=' + str(c) + ' e=' + str(e) + ' p=' + str(p) + ' g=' + str(g) + ' r=' + str(r)
                            param = svm_parameter('-q -s 3 -t '+ str(t) +' -c '+ str(c) +' -e ' + str(e) + ' -p ' + str(p) + ' -g ' + str(g) + ' -r '+ str(r))
                            m = svm_train(prob, param)
                            p_label, p_acc, p_val = svm_predict(y, x, m)
                            error, correlation = getResults(p_acc)
                            results = 'mean squared error:' + str(error) + ' - correlation:' + str(correlation) + '\n'
                            file.write(conditions + ": " + results)
                            if (float(best_results[t][2]) < correlation):
                                best_results[t][0] = conditions
                                best_results[t][1] = str(error)
                                best_results[t][2] = str(correlation)
                            if (max_correlation < correlation):
                                max_correlation = correlation
                                best_condition = conditions
                            if error >= 1:
                                break;
    print 'Finiti training e previsione per ', name
    file.close()

print "Standard Sigmoid:"
param = svm_parameter('-s 3 -t 3')
m = svm_train(prob, param)
p_label, p_acc, p_val = svm_predict(y, x, m)

for t in range(0, 4):
    print getFunction(t) + ' - best prediction'
    print 'parameters: ', best_results[t][0]
    print 'error: ', best_results[t][1]
    print 'correlation: ', best_results[t][2]
    print '\n'
print "best conditions: ", best_condition
print "max correlation: ", max_correlation
