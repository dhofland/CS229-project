import numpy as np
import matplotlib.pyplot as plt
import pandas
import sklearn
import tensorflow as tf
from sklearn import preprocessing, linear_model, neural_network, utils
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
import sklearn
import tensorflow as tf
from scipy.special import expit
from sklearn import svm, metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing, linear_model, neural_network
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.manifold import TSNE

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


# Reads and processes the input data from an .npy file if it exists or from the original csv
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

# Train a logistic classifier and verify results against the dev set
def logistic_classifier(X_train, Y_train, X_dev, Y_dev):
    # Unweighted (simple) classifier
    classifier = simple_classifier(linear_model.LogisticRegression(), X_train, Y_train, X_dev, Y_dev, 'Logistic')

    # Weighted classifier
    classifier = simple_classifier(linear_model.LogisticRegression(class_weight='balanced'), X_train, Y_train,
                                   X_dev, Y_dev, 'Logistic')

    # Filter out insignificant parameters
    thetas = classifier.coef_[0]
    names = np.load('Names.npy')

    significant_cols = [i for i in range(len(thetas)) if thetas[i] > 0.1 or thetas[i] < -0.1]
    X_train_filtered = filter_out_columns(X_train, significant_cols)
    X_dev_filtered = filter_out_columns(X_dev, significant_cols)
    names_filtered = [names[i] for i in significant_cols]

    # Classify filtered examples
    classifier = simple_classifier(linear_model.LogisticRegression(class_weight='balanced'),
                                   X_train_filtered, Y_train, X_dev_filtered, Y_dev, 'Logistic')
    return classifier, significant_cols

def simple_SVM_classifier(X_train, Y_train, X_dev, Y_dev):
    return simple_classifier(svm.LinearSVC(class_weight='balanced'), X_train, Y_train, X_dev, Y_dev, 'SVM')

def simple_nn_classifier(X_train, Y_train, X_dev, Y_dev):
    simple_classifier(neural_network.MLPClassifier(), X_train, Y_train, X_dev, Y_dev, 'NN')
    return

# Run naive bayes and verify results agains the dev set
def naive_bayes_classifier(X_train, Y_train, X_dev, Y_dev):
    classifier = simple_classifier(MultinomialNB(), X_train, Y_train, X_dev, Y_dev, 'Naive Bayes')
    posteriors = classifier.feature_log_prob_[1]
    return

# Default classifier method that fits the training set to the classifier, generates a prediction
# for dev and tests the prediction against the expected values.
def simple_classifier(classifier, X_train, Y_train, X_dev, Y_dev, classifier_type):
    classifier.fit(X_train, Y_train)
    prediction_training = classifier.predict(X_train)
    compute_accuracy(prediction_training, Y_train, classifier_type, 'Training')
    prediction_dev = classifier.predict(X_dev)
    # uniques = np.unique(prediction_dev)
    # print('Unique elements found:', uniques)
    compute_accuracy(prediction_dev, Y_dev, classifier_type, 'Dev')
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

# Method for determining which features have most explanatory value
def feature_select(X, Y):
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = split_train_dev_test(X, Y, 0.7, 0.15)

    X_train, X_dev, X_test = label_X(X_train, X_dev, X_test)
    X_train, X_dev, X_test = scale(X_train, X_dev, X_test)

    for j in list(range(X_train.shape[1])):
        X_train, Y_train, X_dev, Y_dev, X_test, Y_test = split_train_dev_test(X, Y, 0.7, 0.15)

        X_train, X_dev, X_test = label_X(X_train, X_dev, X_test)
        X_train, X_dev, X_test = scale(X_train, X_dev, X_test)

        # n = list(range(X_train.shape[1]))
        # n.remove(j)

        # X_train = filter_out_columns(X_train,n)
        # X_dev = filter_out_columns(X_dev,n)
        # X_test = filter_out_columns(X_test,n)

        NNacc = 0
        NNfraud = 0
        NNnofraud = 0

        SVMacc = 0
        SVMfraud = 0
        SVMnofraud = 0

        maxiter = 10
        for iterate in list(range(maxiter)):
            classifier = simple_classifier(
                neural_network.MLPClassifier(solver='adam', activation='relu', verbose=False,
                                             hidden_layer_sizes=(100,)),
                X_train, Y_train, X_dev, Y_dev, 'NN')
            accuracyNN, fraud_foundNN, non_fraud_foundNN = compute_accuracy(classifier.predict(X_test), Y_test, 'NN', 'Test')

            NNacc = NNacc + accuracyNN
            NNfraud = NNfraud + fraud_foundNN
            NNnofraud = NNnofraud + non_fraud_foundNN

            classes, counts = np.unique(Y, return_counts=True);
            weights = np.divide(counts, len(Y))

            # SVM with scaled values also gives good results
            classifier = simple_SVM_classifier(X_train, Y_train, X_dev, Y_dev)
            accuracySVM, fraud_foundSVM, non_fraud_foundSVM = compute_accuracy(classifier.predict(X_test), Y_test, 'SVM',
                                                                    'Test')

            SVMacc = SVMacc + accuracySVM
            SVMfraud = SVMfraud + fraud_foundSVM
            SVMnofraud = SVMnofraud + non_fraud_foundSVM

        print(
            "NN" + ": " + "Test" + "Column %d: accuracy: %f%%; fraud classification accuracy: %f%%; non-fraud classification accuracy: %f%%"
            % (j, NNacc / maxiter, NNfraud / maxiter, NNnofraud / maxiter))

        print(
            "SVM" + ": " + "Test" + "Column %d: accuracy: %f%%; fraud classification accuracy: %f%%; non-fraud classification accuracy: %f%%"
            % (j, SVMacc / maxiter, SVMfraud / maxiter, SVMnofraud / maxiter))
    return

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

# Filters all but specified columns from X
def filter_out_columns(X, columns_to_keep):
    X = np.transpose(X)
    X_filtered = []
    for i in columns_to_keep:
        X_filtered.append(X[i])
    return np.transpose(X_filtered)


#Scales the input features
def scale(X_train, X_dev, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_dev = scaler.transform(X_dev)
    X_test = scaler.transform(X_test)
    return X_train, X_dev, X_test




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

    #remove bookingdate columns
    X_train = np.delete(X_train, np.s_[0:6], axis=1)
    X_dev = np.delete(X_dev, np.s_[0:6], axis=1)
    X_test = np.delete(X_test, np.s_[0:6], axis=1)

    return X_train, Y_train, X_dev, Y_dev, X_test, Y_test

# Computes the prediction accuracy for overall, frauds and non-frauds
def compute_accuracy(prediction, Y, model_type, set_type):
    fraud_found = 0
    non_fraud_found = 0

    for i in range(len(prediction)):
        if(prediction[i]):
            if(Y[i] == 1):
                fraud_found += 1
            else:
                non_fraud_found += 1

    prediction = prediction*1
    total_correct = sum(prediction)
    accuracy = total_correct / len(Y)

    total_fraud = sum([1 for i in Y if i == 1])
    fraud_found = (fraud_found / total_fraud) * 100
    total_non_fraud = sum([1 for i in Y if i == -1])
    non_fraud_found = (non_fraud_found / total_non_fraud) * 100

    print(
        model_type + ": " + set_type + ": accuracy: %f%%; fraud classification accuracy: %f%%; non-fraud classification accuracy: %f%%"
        % (accuracy, fraud_found, non_fraud_found))
    return accuracy, fraud_found, non_fraud_found

seed = 128
rng = np.random.RandomState(seed)

# Labels the input features
def label_X(X_train, X_dev, X_test):
    le = preprocessing.LabelEncoder()
    X_train = label(le, X_train)
    X_dev = label(le, X_dev)
    X_test = label(le, X_test)
    return X_train, X_dev, X_test


# Calculates output of the NN. Effectively performs the forward computations
# Implementation with a single layer
def multilayer_perceptron_one_layer(x,n_input, n_hidden_1, n_classes):
    name_hidden_layer = "H1_n" + str(n_hidden_1)
    with tf.name_scope (name_hidden_layer) as scope:
        weights_h1 = tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='Weights')
        bias_h1 = tf.Variable(tf.random_normal([n_hidden_1]), name='Bias')
        layer_1 = tf.add(tf.matmul(x, weights_h1), bias_h1)
        layer_1 = tf.nn.relu(layer_1)
    with tf.name_scope('Output_layer') as scope:
        weights_out = tf.Variable(tf.random_normal([n_hidden_1, n_classes]), name='Weights')
        bias_out = tf.Variable(tf.random_normal([n_classes]), name='Bias')
        out_layer = tf.add(tf.matmul(layer_1, weights_out), bias_out)
        out_layer = tf.sigmoid(out_layer)
        return out_layer, weights_h1, weights_out

def encode(series):
  return pandas.get_dummies(series.astype(str))


# Selects batches by making sure that they always contain a fixed number of fraud-cases
# i.e. prob_of_fraud times batch size
def sample_batches(X, Y, batch_size, prob_of_fraud):
    fraud_indices_list = np.where(Y == -1)[0]
    non_fraud_indices_list = np.where(Y == 1)[0]

    num_frauds = int(round(prob_of_fraud * batch_size))
    num_non_frauds = batch_size - num_frauds

    total_batch = int(len(non_fraud_indices_list) / num_non_frauds)
    number_batches = int(len(X) / batch_size)


    X_batches = []
    Y_batches = []

    for i in range(number_batches):
        fraud_indices = np.random.choice(fraud_indices_list, num_frauds)
        non_fraud_indices = np.random.choice(non_fraud_indices_list, num_non_frauds)
        X_batch = np.append(np.take(X, fraud_indices, axis=0), np.take(X, non_fraud_indices, axis=0), axis=0)
        Y_batch = np.append(np.take(Y, fraud_indices), np.take(Y, non_fraud_indices))
        Y_batch = encode(Y_batch)
        X_batches.append(X_batch)
        Y_batches.append(Y_batch)

    return X_batches, Y_batches

# Runs a single training-epoch
def run_epoch(sess, X_train, batch_size, Y_traintemp, optimizer, cost, x, y, freq_frauds, writer, epoch, merged_summary_op):
    avg_cost = 0.0
    total_batch = int(len(X_train) / batch_size)
    x_batches, y_batches = sample_batches(X_train, Y_traintemp, batch_size, freq_frauds)
    for i in range(total_batch):
        batch_x, batch_y = x_batches[i], y_batches[i]
        _, c, summary = sess.run([optimizer, cost, merged_summary_op],
                            feed_dict={
                                x: batch_x,
                                y: batch_y
                            })
        writer.add_summary(summary, epoch * total_batch + i)
        avg_cost += c / total_batch
    return avg_cost

# Filters all but specified columns from X
def filter_out_columns(X, columns_to_keep):
    X = np.transpose(X)
    X_filtered = []
    for i in columns_to_keep:
        X_filtered.append(X[i])
    return np.transpose(X_filtered)

# Focal loss function
def focalLoss(predictions, targets, L2loss, gamma, alpha):
    temp = tf.subtract(tf.constant(1.0), predictions)
    temp = tf.pow(temp, gamma)
    temp = tf.multiply(temp, targets)
    temp = tf.multiply(temp, tf.constant(alpha[1]))
    temp = tf.multiply(temp, tf.log(tf.clip_by_value(predictions,1e-10,1.0)))
    temp = tf.multiply(temp, tf.constant(-1.0))

    temp2 = tf.pow(predictions, gamma)
    temp2 = tf.multiply(temp2, tf.constant(alpha[0]))
    temp2 = tf.multiply(temp2, tf.subtract(tf.constant(1.0), targets))
    temp3 = tf.clip_by_value(tf.subtract(tf.constant(1.0), predictions),1e-10,1.0)
    temp2 = tf.multiply(temp2, tf.log(temp3))
    temp2 = tf.multiply(temp2, tf.constant(-1.0))

    cost = tf.add(temp, temp2)
    cost = tf.add(cost, L2loss)

    return tf.reduce_mean(cost)

# Our tensorflow neural network implementation
def neuralNetwork(X, Y, batch_size, tolerance, learning_rate, alpha_reg, gamma, freq_frauds, n_hidden_1, it):
    X, Y = filter_refused_transactions(X, Y)
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = split_train_dev_test(X, Y, 0.7, 0.15)
    X_train, X_dev, X_test = label_X(X_train, X_dev, X_test)
    X_train, X_dev, X_test = scale(X_train, X_dev, X_test)
    X_train = filter_out_columns(X_train, [2, 3, 14, 16])
    X_dev = filter_out_columns(X_dev, [2,3,14,16])
    Y_traintemp =  encode(Y_train)

    n_input = X_train.shape[1]
    n_classes = Y_traintemp.shape[1]

    if (it == 0):
        with tf.variable_scope('Inputs') as scope:
            global x
            x = tf.placeholder("float", [None, n_input], name="X")

        with tf.variable_scope('Labels') as scope:
            global y
            y = tf.placeholder("float", [None, n_classes])

    predictions, weights_h1, weights_out = multilayer_perceptron_one_layer(x,n_input, n_hidden_1, n_classes)

    weights_fraud = freq_frauds
    weight_settled = (1 - freq_frauds)
    alpha = [np.float32(weights_fraud), np.float32(weight_settled)]

    with tf.name_scope('Focal_loss') as scope:
        L2loss = tf.multiply(alpha_reg, tf.add(tf.nn.l2_loss(weights_h1), tf.nn.l2_loss(weights_out)), name='Reg_term')
        cost = focalLoss(predictions, y, L2loss, tf.constant(gamma), alpha)


    with tf.name_scope('Accuracy') as scope:
        correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
        overall_accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name="Optimizer").minimize(cost)

    tf.summary.scalar("cost", cost)
    tf.summary.scalar("overall_accuracy", overall_accuracy)
    merged_summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph())
        sess.run(tf.global_variables_initializer())

        prev_cost = 0.0
        epoch = 0
        display_step = 1
        while(True):
            avg_cost = run_epoch(sess, X_train, batch_size, Y_train, optimizer, cost, x, y, freq_frauds, summary_writer, epoch, merged_summary_op)
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=",
                      "{:.9f}".format(avg_cost))
            if (abs((avg_cost - prev_cost) / avg_cost) <= tolerance):
                break
            else:
                prev_cost = avg_cost
                epoch += 1
        print("Optimization Finished!")
        correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        summary_writer.close()
        print("Training Accuracy:", accuracy.eval({x: X_train, y: Y_traintemp}))
        results = correct_prediction.eval({x: X_train, y: Y_traintemp})
        acc_train, fraud_train, non_fraud_train = compute_accuracy(results, Y_train, "NN", 'Train')
        results = correct_prediction.eval({x: X_dev, y: encode(Y_dev)})
        acc_dev, fraud_dev, non_fraud_dev = compute_accuracy(results, Y_dev, "NN", 'Dev')

    return acc_train, fraud_train, non_fraud_train, acc_dev, fraud_dev, non_fraud_dev

# Main method
def main():
    X, Y = read_data()
    # Meta-parameters #
    #########################
    batch_size = 1000  # baseline: 1000
    tolerance = 1e-2  # baseline: 1e-2
    learning_rate = 5.  # baseline: 5.
    alpha_reg = 0.001  # baseline: 0.001
    gamma = 2.  # baseline: 2
    freq_frauds = 0.45  # baseline: 0.45
    n_hidden_1 = 500  # baseline: 500
    #########################

    gammarange = np.float32(np.linspace(1.0, 5.0, 5))
    frange = np.float32(np.linspace(0.1, 0.9, 9))

    print(frange)

    iterate = 5

    overalllst = []
    fraudlst = []
    nonfraudlst = []

    proc = 0

    for freq_frauds in frange:
        print(freq_frauds)

        avg_acc_train = 0.0
        avg_fraud_train = 0.0
        avg_nofraud_train = 0.0
        avg_acc_dev = 0.0
        avg_fraud_dev = 0.0
        avg_nofraud_dev = 0.0

        for i in range(iterate):
            print(i)
            acc_train, fraud_train, non_fraud_train, acc_dev, fraud_dev, non_fraud_dev = neuralNetwork(
                X, Y, batch_size, tolerance, learning_rate, alpha_reg, gamma, freq_frauds, n_hidden_1, proc)
            avg_acc_train = avg_acc_train + acc_train
            avg_fraud_train = avg_fraud_train + fraud_train
            avg_nofraud_train = avg_nofraud_train + non_fraud_train
            avg_acc_dev = avg_acc_dev + acc_dev
            avg_fraud_dev = avg_fraud_dev + fraud_dev
            avg_nofraud_dev = avg_nofraud_dev + non_fraud_dev

            proc = proc + 1

        print("")
        print("Average train accuracy:", avg_acc_train / iterate * 100)
        print("Average fraud train accuracy:", avg_fraud_train / iterate)
        print("Average non-fraud train accuracy:", avg_nofraud_train / iterate)
        print("")
        print("Average dev accuracy:", avg_acc_dev / iterate * 100)
        print("Average fraud dev accuracy:", avg_fraud_dev / iterate)
        print("Average non-fraud dev accuracy:", avg_nofraud_dev / iterate)

        overalllst.append(avg_acc_dev / iterate * 100)
        fraudlst.append(avg_fraud_dev / iterate)
        nonfraudlst.append(avg_nofraud_dev / iterate)

    plt.plot(frange, overalllst, label='Overall accuracy')
    plt.plot(frange, fraudlst, label='Fraud accuracy')
    plt.plot(frange, nonfraudlst, label='Non-Fraud accuracy')
    plt.title('Dev accuracy vs. $f$')
    plt.xlabel('$f$')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()
    return

#  Tensorboard:
# 1. cmd
# 2. cd C:\Users\djhof\AppData\Local\Programs\Python\Python35\Lib\site-packages\tensorboard
# 2. python main.py --logdir=C:\Users\djhof\Documents\Studies\Stanford\CS229\Project\CS229-project\logs