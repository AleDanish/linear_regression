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

def plot(labels, *values_list):
    values = list(values_list)
    observations = []
    for i in np.arange(1, len(values[0]) + 1):
        observations.append(i)
    print labels
    print values
    for lab, val in zip(labels, values):
        plt.plot(observations, val, label=lab)
    plt.legend()
    plt.grid(True)
    plt.xlabel('observations')
    plt.ylabel('temperature (C)')
    plt.title('Time-series data Prediction')
    plt.ylim(0, 50)
    plt.show()

import numpy as np
from svmutil import *
import subprocess
import matplotlib.pyplot as plt
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
    print 'Finished training and prevision for ', name
    file.close()

print '----------------------------------------------------------------------------------------------------------------------------'
for t in range(0, 4):
    print 'EPSILON-SRV', getFunction(t)
    print 'BEST CORRELATION - parameters:', best_results_corr[t][0] , '- error:', best_results_corr[t][1], '- correlation:', best_results_corr[t][2]
    print 'BEST ERROR - parameters: ', best_results_error[t][0], '- error: ', best_results_error[t][1], '- correlation: ', best_results_error[t][2], '\n'
print '----------------------------------------------------------------------------------------------------------------------------'

# K-steps ahead prevision
real_values = y[sample_num:]
prediction_rw = []
for i in range(sample_num, len(y)):
    prediction_rw.append(y[sample_num-1])
file = open('output/Prevision K-steps ahead', 'w')
file.write('PREDICTION K-STEP AHEAD\n\n')
file.write('\nReal values:' + str(real_values) + '\n')
file.write('\nRandom Walk:' + str(prediction_rw)  + '\n')
print 'PREDICTION K-STEP AHEAD'
print 'Real values:', real_values
print 'Random Walk:', prediction_rw
for t in range(0, 4):
    print '\n', getFunction(t)
    print 'best correlation prediction', prediction_corr[t]
    print 'parameters:', best_results_corr[t][0] , '- error:', best_results_corr[t][1], '- correlation:', best_results_corr[t][2]
    print 'best error prediction', prediction_error[t]
    print 'parameters: ', best_results_error[t][0], '- error: ', best_results_error[t][1], '- correlation: ', best_results_error[t][2], '\n'
    file.write('\n' + getFunction(t))
    file.write('\nbest correlation prediction' + str(prediction_corr[t]))
    file.write('\nparameters:' + best_results_corr[t][0] + 'error:' + best_results_corr[t][1] + 'correlation:' + best_results_corr[t][2])
    file.write('\n\nbest error prediction' + str(prediction_error[t]))
    file.write('\nparameters: ' + best_results_error[t][0] + 'error: ' + best_results_error[t][1] + 'correlation: ' + best_results_error[t][2])
    file.write('\n')
file.close()
print '----------------------------------------------------------------------------------------------------------------------------'

# One-step ahead prevision
file = open('output/Prevision One-step ahaed', 'w')
file.write('PREDICTION ONE-STEP AHEAD\n')
print 'PREDICTION ONE-STEP AHEAD'
regression_prediction = [[] for i in range(4)]
for num in range(sample_num, len(y), 1):
    print '\nReal value:', y[num]
    file.write('\nReal value: ' + str(y[num]))
    num_rw = y[num - 1]
    print 'Random Walk: ', num_rw
    file.write('\nRandom Walk next value: ' + str(num_rw))
    for t in range(0, 1):
        param = '-q -s 3 ' + best_results_error[t][0]
        m = svm_train(y[:num], x[:num], param)
        p_label, p_acc, p_val = svm_predict([y[num]], [x[num]], m)
        regression_prediction[t].extend(p_label)
        print 'EPSILON-SRV', getFunction(t), '- next predicted value:', p_label
        file.write('\nEPSILON-SRV: ' + str(p_label) + '\n')
file.close()
print '----------------------------------------------------------------------------------------------------------------------------'
print "Best correlation conditions: ", best_condition_corr
print "Max correlation: ", max_correlation
print "Best error conditions: ", best_condition_error
print "Min erorr: ", min_error
print regression_prediction
plot(['real values', 'linear'], y[sample_num:], regression_prediction[0])
