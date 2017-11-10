import numpy as np
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
from datetime import datetime
import pandas
from sklearn.naive_bayes import MultinomialNB


# Splits the data according to their label.
# Returns Y where 1 represents a chargeback transaction, 0 a settled transaction and -1 a refused transaction.
def generate_Y(data):
    charge_back = np.where(data[:, 9]=='Chargeback')[0]
    refused = np.where(data[:, 9]=='Refused')[0]
    Y = np.zeros(len(data), dtype=int)
    for index in charge_back:
        Y[index] = 1
    for index in refused:
        Y[index] = -1
    return Y


# Reads the input data from an .npy filie if it exists or from the original csv
# otherwise. Returns npArray
def read_data():
    try:
        return np.load('X.npy'), np.load('Y.npy')
    except IOError:
        data = pandas.read_csv('data_for_student_case.csv/data_for_student_case.csv').as_matrix()
        Y = generate_Y(data)
        for i in range (len(data)):
            data[i, 1] = pandas.to_datetime(data[i, 1])
            data[i, 13] = pandas.to_datetime(data[i, 12])
        X = np.delete(data, 9, axis=1)
        X = np.delete(X, 0, axis=1)
        np.save('X.npy', data)
        np.save('Y.npy', Y)
        return X, Y

# Remove all refused transactions
# Returns filtered X and Y
def filter_refused_transactions(X, Y):
    refused = np.where(Y < 0)
    X_filtered = np.delete(X, refused, axis=0)
    Y_filtered = np.delete(Y, refused, axis=0)
    return X_filtered, Y_filtered


# Splits X into discrete and continous variables.
# Returns X_continous and X_discrete.
def split(X):
    continous_indices = [0, 4, 10]
    discrete_indeces = [1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14]
    X_continuous = np.delete(X, discrete_indeces, axis=1)
    X_discrete = np.delete(X, continous_indices, axis=1)
    return X_continuous, X_discrete

# Perform Naive bayes on all discrete variables.
def naive_bayes(X, Y):
    clf = MultinomialNB()
    clf.fit(X, Y)
    return

# Perform simple logistic regression on all continuous variables.
def simple_logistic(X, Y):
    return

# Initial naive analysis. We train two models, one with continous and one with
# Discrete variables.
def naive_analysis(X, Y):
    X, Y = filter_refused_transactions(X, Y)
    X_continuous, X_discrete = split(X)
    naive_bayes(X_discrete, Y)
    simple_logistic(X_continuous, Y)
    return


# Randomly the data into train, dev and test sets of size 'train_size', 'dev_size' and the remainder.
# Returns all six arrays.
def split_train_dev_test(X, Y, train_size, dev_size):
    total_indices = list(range(len(X)))
    no_train_samples = int(len(X) * train_size)
    no_dev_samples = int(len(X) * dev_size)

    train_indices = np.sort(np.random.choice(total_indices, no_train_samples, replace=False))
    temp = np.delete(total_indices, train_indices)
    dev_indices = np.random.choice(temp, no_dev_samples, replace=False)
    test_indices = np.delete(temp, dev_indices)

    X_train = [X[i] for i in train_indices]
    Y_train = [Y[i] for i in train_indices]

    X_dev = [X[i] for i in dev_indices]
    Y_dev = [Y[i] for i in dev_indices]

    X_test = [X[i] for i in test_indices]
    Y_test = [Y[i] for i in test_indices]

    return X_train, Y_train, X_dev, Y_dev, X_test, Y_test




# Main method
def main():
    X, Y = read_data()
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = split_train_dev_test(X, Y, 0.7, 0.15)

    naive_analysis(X_train, Y_train)
    return


main()