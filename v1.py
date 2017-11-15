import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing, linear_model, neural_network
from sklearn.preprocessing import StandardScaler

# Splits the data according to their label.
# Returns Y where 1 represents a chargeback transaction, 0 a settled transaction and -1 a refused transaction.
def generate_Y(data):
    charge_back = np.where(data[:, 9]=='Chargeback')[0]
    settled = np.where(data[:, 9]=='Settled')[0]
    Y = np.zeros(len(data), dtype=int)
    for index in charge_back:
        Y[index] = 1
    for index in settled:
        Y[index] = -1
    return Y


# Reads and processes the input data from an .npy filie if it exists or from the original csv
# otherwise. Returns npArray
def read_data():
    try:
        return np.load('X.npy'), np.load('Y.npy')
    except IOError:
        filename = 'data_for_student_case.csv/data_for_student_case.csv'
        data = pandas.read_csv(filename,
                               keep_default_na=False)
        names = np.asarray(list(data))
        data = data.as_matrix()
        Y = generate_Y(data)
        for i in range (len(data)):
            data[i, 1] = pandas.to_datetime(data[i, 1])
            data[i, 12] = pandas.to_datetime(data[i, 12])

        X = np.delete(data, 9, axis=1)
        X = np.delete(X, 0, axis=1)
        names = np.delete(names, 9)
        names = np.delete(names, 0)

        X = np.transpose(X)
        index_offset = 0
        for i in range (len(X)):
            index = i + index_offset
            temp = X[index]
            row_type = type(temp[0])
            if (row_type == float or row_type == int):
                temp[temp == 'NA'] = 0
                temp = temp.astype(float)
                X[index] = temp
            elif (row_type == pandas._libs.tslib.Timestamp):
                temp = pandas.DatetimeIndex(temp)
                name = names[index]

                X = np.delete(X, index, 0)
                names = np.delete(names, index)
                X = np.insert(X, index, temp.second, 0)
                names = np.insert(names, index, name + ' second')
                X = np.insert(X, index, temp.minute, 0)
                names = np.insert(names, index, name + ' minute')
                X = np.insert(X, index, temp.hour, 0)
                names = np.insert(names, index, name + ' hour')
                X = np.insert(X, index, temp.day, 0)
                names = np.insert(names, index, name + ' day')
                X = np.insert(X, index, temp.month, 0)
                names = np.insert(names, index, name + ' month')
                X = np.insert(X, index, temp.year, 0)
                names = np.insert(names, index, name + ' year')

                index_offset += 5
            else:
                X[index] = X[index].astype(str)
        X = np.transpose(X)
        np.save('X.npy', X)
        np.save('Y.npy', Y)
        np.save('Names.npy', names)
        return X, Y

# Remove all refused transactions
# Returns filtered X and Y
def filter_refused_transactions(X, Y):
    refused = np.where(Y == 0)
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


# Replace all categorical values with their corresponding labels in le if they exist. If not
# new labels are added to the encoder.
# Returns labelled array.
def label(le, X):
    X = np.transpose(X)
    for i in range(len(X)):
        temp = X[i]
        if (type(temp[0]) == str):
            temp = le.fit_transform(temp)
            X[i] = temp
    return np.transpose(X)

def label_X(X_train, X_dev, X_test):
    le = preprocessing.LabelEncoder()
    X_train = label(le, X_train)
    X_dev = label(le, X_dev)
    X_test = label(le, X_test)
    return X_train, X_dev, X_test

# Test predicted Y against actual Y of model_type (e.g. logistic) and set_type (e.g. train/dev)
# (Old labelling scheme)
def test(prediction, Y, model_type, set_type):
    mistakes = sum([1 for i, j in zip(prediction, Y) if i != j])

    accuracy = (1 - (mistakes / len(Y))) * 100

    fraud_found = (np.dot(np.transpose(prediction), Y) / sum(Y)) * 100

    Y_inv = (Y + 1) % 2
    prediction_inv = (prediction + 1) % 2
    non_fraud_found = (np.dot(np.transpose(prediction_inv), Y_inv) / sum(Y_inv)) * 100

    print(model_type + ": " + set_type + ": accuracy: %f%%; fraud classification accuracy: %f%%; non-fraud classification accuracy: %f%%"
          %(accuracy, fraud_found,non_fraud_found))
    return


# Test method for new labeling scheme (with settled = -1; chargeback = 1
def test2(prediction, Y, model_type, set_type):
    mistakes = sum([1 for i, j in zip(prediction, Y) if i != j])
    fraud_found = sum([1 for i, j in zip(prediction, Y) if i == j and i == 1])
    total_fraud = sum([1 for i in Y if i == 1])
    non_fraud_found = sum([1 for i, j in zip(prediction, Y) if i == j and i == -1])
    total_non_fraud = sum([1 for i in Y if i == -1])
    accuracy = (1 - (mistakes / len(Y))) * 100
    fraud_found = (fraud_found / total_fraud) * 100

    non_fraud_found = (non_fraud_found / total_non_fraud) * 100

    print(
        model_type + ": " + set_type + ": accuracy: %f%%; fraud classification accuracy: %f%%; non-fraud classification accuracy: %f%%"
        % (accuracy, fraud_found, non_fraud_found))
    return

#plots values against their corresponding names
def plot_against_names(values, plotname, names=np.load('Names.npy')):
    x_cords = range(len(names))
    fig, ax = plt.subplots()
    ax.scatter(x_cords, values)

    for i in x_cords:
        if (i % 2 == 0):
            ax.annotate(names[i],
                        (i, values[i]),
                        xytext=(-10, 10),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle="->"))
        else:
            ax.annotate(names[i],
                        (i, values[i]),
                        xytext=(10, -10),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle="->"))

    fig.savefig(plotname)
    return

# Run naive bayes and verify results agains the dev set
def naive_bayes_classifier(X_train, Y_train, X_dev, Y_dev):
    classifier = simple_classifier(MultinomialNB(), X_train, Y_train, X_dev, Y_dev, 'Naive Bayes')
    posteriors = classifier.feature_log_prob_[1]
    plot_against_names(posteriors, 'naive_bayes_posteriors.png')
    return

# Filters all but specified columns from X
def filter_out_columns(X, columns_to_keep):
    X = np.transpose(X)
    X_filtered = []
    for i in columns_to_keep:
        X_filtered.append(X[i])
    return np.transpose(X_filtered)


# Train a logistic classifier and verify results against the dev set
def logistic_classifier(X_train, Y_train, X_dev, Y_dev):
    # Unweighted (simple) classifier
    classifier = simple_classifier(linear_model.LogisticRegression(), X_train, Y_train, X_dev, Y_dev, 'Logistic')
    plot_against_names(classifier.coef_[0], 'logistic_thetas_unweighted.png')

    # Weighted classifier
    classifier = simple_classifier(linear_model.LogisticRegression(class_weight='balanced'), X_train, Y_train, X_dev, Y_dev, 'Logistic')
    plot_against_names(classifier.coef_[0], 'logistic_thetas_balanced.png')

    # Filter out insignificant parameters
    thetas = classifier.coef_[0]
    names = np.load('Names.npy')

    significant_cols = [i for i in range (len(thetas)) if thetas[i] > 0.1 or thetas[i] < -0.1]
    X_train_filtered = filter_out_columns(X_train, significant_cols)
    X_dev_filtered = filter_out_columns(X_dev, significant_cols)
    names_filtered = [names[i] for i in significant_cols]

    # Classify filtered examples
    classifier = simple_classifier(linear_model.LogisticRegression(class_weight='balanced'),
                                   X_train_filtered, Y_train, X_dev_filtered, Y_dev, 'Logistic')
    plot_against_names(classifier.coef_[0], 'logistic_thetas_weighted_filtered.png', names=names_filtered)
    return classifier, significant_cols

def simple_SVM_classifier(X_train, Y_train, X_dev, Y_dev):
    return simple_classifier(svm.LinearSVC(class_weight='balanced'), X_train, Y_train, X_dev, Y_dev, 'SVM')


def simple_nn_classifier(X_train, Y_train, X_dev, Y_dev):
    simple_classifier(neural_network.MLPClassifier(), X_train, Y_train, X_dev, Y_dev, 'NN')
    return

# Default classifier method that fits the training set to the classifier, generates a prediction
# for dev and tests the prediction against the expected values.
def simple_classifier(classifier, X_train, Y_train, X_dev, Y_dev, classifier_type):
    classifier.fit(X_train, Y_train)
    prediction_training = classifier.predict(X_train)
    test2(prediction_training, Y_train, classifier_type, 'Training')
    prediction_dev = classifier.predict(X_dev)
    # uniques = np.unique(prediction_dev)
    # print('Unique elements found:', uniques)
    test2(prediction_dev, Y_dev, classifier_type, 'Dev')
    return classifier


# Initial naive analysis consisting of Naive Bayes and logistic regression.
def naive_analysis(X, Y):
    X, Y = filter_refused_transactions(X, Y)
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = split_train_dev_test(X, Y, 0.7, 0.15)
    X_train, X_dev, X_test = label_X(X_train, X_dev, X_test)

    naive_bayes_classifier(X_train, Y_train, X_dev, Y_dev)
    classifier, significant_cols = logistic_classifier(X_train, Y_train, X_dev, Y_dev)
    simple_SVM_classifier(X_train, Y_train, X_dev, Y_dev)
    simple_nn_classifier(X_train, Y_train, X_dev, Y_dev)

    # X_test = filter_out_columns(X_test, significant_cols)
    # prediction_test = classifier.predict(X_test)
    # test2(prediction_test, Y_test, 'Logistic', 'Test')
    return

def scale(X_train, X_dev, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_dev = scaler.transform(X_dev)
    X_test = scaler.transform(X_test)
    return X_train, X_dev, X_test

def modified_classifiers(X, Y):
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = split_train_dev_test(X, Y, 0.7, 0.15)

    X_train, X_dev, X_test = label_X(X_train, X_dev, X_test)
    X_train, X_dev, X_test = scale(X_train, X_dev, X_test)
    classifier = simple_classifier(neural_network.MLPClassifier(solver='adam', activation='relu', verbose=True,
                                                   hidden_layer_sizes=(100,)),
                      X_train, Y_train, X_dev, Y_dev, 'NN')
    test2(classifier.predict(X_test), Y_test, 'NN', 'Test')

    classes, counts = np.unique(Y, return_counts=True);
    weights = np.divide(counts, len(Y))

    # SVM with scaled values also gives good results
    classifier = simple_SVM_classifier(X_train, Y_train, X_dev, Y_dev)
    test2(classifier.predict(X_test), Y_test, 'SVM', 'Test')
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
    # naive_analysis(X, Y)
    modified_classifiers(X, Y)
    return

main()