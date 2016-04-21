#!/usr/bin/env python

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from svmutil import *
datafile = 'input_data'
#datafile='heart_scale'

# Read data in LIBSVM format
y, x = svm_read_problem(datafile)
problem = svm_problem(y, x)

##### LINEAR #####
print "y[:5]: ", y[:5]
print "x[:5]: ", x[:5]

m = svm_train(y[:5], x[:5], '-t 0 -c 5 -h 0')


svm_save_model(datafile + '.model', m)
p_label, p_acc, p_val = svm_predict(y[5:], x[5:], m)
print "p_label: ",p_label
print "p_acc: ", p_acc
print "p_val: ",p_val
