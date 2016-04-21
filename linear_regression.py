#!/usr/bin/env python
if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from svmutil import *
#datafile = 'input_data'
datafile='heart_scale'
# Read data in LIBSVM format
y, x = svm_read_problem('../../'+datafile)
m = svm_train(y[:200], x[:200], '-t 0 -c 2 -v 20')
p_label, p_acc, p_val = svm_predict(y[200:], x[200:], m)

print "p_label: ",p_label
print "p_acc: ", p_acc
print "p_val: ",p_val

# Other utility functions
#svm_save_model(datafile + '.model', m)
#m = svm_load_model(datafile + '.model')
#p_label, p_acc, p_val = svm_predict(y, x, m, '-b 1')
#ACC, MSE, SCC = evaluations(y, p_label)
