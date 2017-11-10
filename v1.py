import numpy as np
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
from datetime import datetime
import pandas
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as sp
from sklearn import preprocessing


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
        data = pandas.read_csv('data_for_student_case.csv/data_for_student_case.csv', keep_default_na=False).as_matrix()
        Y = generate_Y(data)
        for i in range (len(data)):
            data[i, 1] = pandas.to_datetime(data[i, 1])
            data[i, 12] = pandas.to_datetime(data[i, 12])
        X = np.delete(data, 9, axis=1)
        X = np.delete(X, 0, axis=1)
        np.save('X.npy', X)
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

def label(le, X):
    # X[X == 'NA'] = -1
    X = np.transpose(X)
    for i in range(len(X)):
        temp = X[i]
        if (i == 2):
            temp = X[i].astype(str)
        temp = le.fit_transform(temp)
        X[i] = temp

    return np.transpose(X)

def fit_model(X_train, Y_train):
    clf = MultinomialNB()
    clf.fit(X_train, Y_train)
    return clf

def test(clf, X, Y, type):
    prediction = clf.predict(X)
    mistakes = sum([1 for i, j in zip(prediction, Y) if i != j])
    accuracy = (1 - (mistakes / len(Y))) * 100
    fraud_found = (np.dot(np.transpose(prediction), Y) / sum(Y)) * 100

    Y_inv = (Y + 1) % 2
    prediction_inv = (prediction + 1) % 2
    non_fraud_found = (np.dot(np.transpose(prediction_inv), Y_inv) / sum(Y_inv)) * 100

    print(type + ": accuracy: %f%%; fraud detection accuracy: %f%%; non-fraud detection accuracy: %f%%"
          %(accuracy, fraud_found,non_fraud_found))
    return fraud_found

# Perform Naive bayes on all discrete variables.
def naive_bayes(X_train, Y_train, X_dev, Y_dev):
    le = preprocessing.LabelEncoder()
    X_train = label(le, X_train)
    clf = fit_model(X_train, Y_train)
    test(clf, X_train, Y_train, 'Train')
    X_dev = label(le, X_dev)
    test(clf, X_dev, Y_dev, 'Dev')
    return

# Perform simple logistic regression on all continuous variables.
def simple_logistic(X, Y):
    return

# Initial naive analysis. We train two models, one with continous and one with
# Discrete variables.
def naive_analysis(X_train, Y_train, X_dev, Y_dev):
    X_train, Y_train = filter_refused_transactions(X_train, Y_train)
    X_train_continuous, X_train_discrete = split(X_train)
    X_dev_continuous, X_dev_discrete = split(X_dev)

    naive_bayes(X_train_discrete, Y_train, X_dev_discrete, Y_dev)
    simple_logistic(X_train_continuous, Y_train)
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

    X_train = np.array([X[i] for i in train_indices])
    Y_train = np.array([Y[i] for i in train_indices])

    X_dev = np.array([X[i] for i in dev_indices])
    Y_dev = np.array([Y[i] for i in dev_indices])

    X_test = np.array([X[i] for i in test_indices])
    Y_test = np.array([Y[i] for i in test_indices])

    return X_train, Y_train, X_dev, Y_dev, X_test, Y_test




# Main method
def main():
    X, Y = read_data()
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = split_train_dev_test(X, Y, 0.7, 0.15)

    naive_analysis(X_train, Y_train, X_dev, Y_dev)
    return


main()