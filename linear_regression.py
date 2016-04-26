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

import numpy as np
from svmutil import *
# Read data in LIBSVM format
y, x = svm_read_problem('input_data')
prob  = svm_problem(y, x)

max_correlation = 0
best_conditions = ''
best_results = [[0 for i in range(2)] for j in range(4)] 
for t in range(0, 4):
    if t == 0:
        name = 'LINEAR'
    elif t == 1:
        name = 'POLYNOMIAL'
    elif t == 2:
        name = 'GAUSSIAN'
    elif t == 3:
        name = 'SIGMOID'
    file = open('output/' + name, 'w')
    file.write('### EPSILON-SRV ' + name + ' ###\n')
    for c in range (1, 100, 10):
        e_arr = np.arange(0.1, 1, 0.1)
        for e in e_arr:
            p_arr = np.arange(0.1, 1, 0.1)
            for p in p_arr:
                conditions = 't=' + str(t) + ' c=' + str(c) + ' e=' + str(e) + ' p=' + str(p)
                param = svm_parameter('-q -s 3 -t '+ str(t) +' -c '+ str(c) +' -e ' + str(e) + ' -p ' + str(p))
                m = svm_train(prob, param)
                p_label, p_acc, p_val = svm_predict(y, x, m)
                error, correlation = getResults(p_acc)
                results = 'mean squared error:' + str(error) + ' - correlation:' + str(correlation) + '\n'
                file.write(conditions + ": " + results)
                if (best_results[t][1] < correlation):
                    best_results[t][0] = error
                    best_results[t][1] = correlation
                if (max_correlation < correlation):
                    max_correlation = correlation
                    best_condition = conditions
                if error >= 1:
                    break;
    file.close()

print "best conditions: ", best_condition
print "max correlation: ", max_correlation
