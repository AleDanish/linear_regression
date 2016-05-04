#!/usr/bin/env python
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
