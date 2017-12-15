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
    return accuracy, fraud_found, non_fraud_found



# Filters all but specified columns from X
def filter_out_columns(X, columns_to_keep):
    X = np.transpose(X)
    X_filtered = []
    for i in columns_to_keep:
        X_filtered.append(X[i])
    return np.transpose(X_filtered)



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

def test3(prediction, Y, model_type, set_type):
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

def label_X(X_train, X_dev, X_test):
    le = preprocessing.LabelEncoder()
    X_train = label(le, X_train)
    X_dev = label(le, X_dev)
    X_test = label(le, X_test)
    return X_train, X_dev, X_test


# Calculates output of the NN. Effectively performs the forward computations
# Implementation with a single layer
def multilayer_perceptron_one_layer(x, weights, biases, keep_prob):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    out_layer = tf.sigmoid(out_layer)
    return out_layer

def encode(series):
  return pandas.get_dummies(series.astype(str))


# Selects batches by making sure that they always contain a fixed number of fraud-cases
# i.e. prob_of_fraud times batch size
def sample_batches(X, Y, batch_size, prob_of_fraud):
    fraud_indices_list = np.where(Y == -1)[0]
    non_fraud_indices_list = np.where(Y == 1)[0]

    num_frauds = round(prob_of_fraud * batch_size)
    num_non_frauds = batch_size - num_frauds

    total_batch = int(len(non_fraud_indices_list) / num_non_frauds)
    number_batches = int(len(X) / batch_size)


    X_batches = []
    Y_batches = []

    # non_fraud_indices = np.array_split(non_fraud_indices_list, total_batch)

    for i in range(number_batches):
        fraud_indices = np.random.choice(fraud_indices_list, num_frauds)
        non_fraud_indices = np.random.choice(non_fraud_indices_list, num_non_frauds)
        X_batch = np.append(np.take(X, fraud_indices, axis=0), np.take(X, non_fraud_indices, axis=0), axis=0)
        Y_batch = np.append(np.take(Y, fraud_indices), np.take(Y, non_fraud_indices))
        Y_batch = encode(Y_batch)
        X_batches.append(X_batch)
        Y_batches.append(Y_batch)

    return X_batches, Y_batches

def run_epoch4(sess, X_train, batch_size, Y_traintemp, optimizer, cost, x, y, keep_prob, freq_frauds):
    avg_cost = 0.0
    total_batch = int(len(X_train) / batch_size)
    x_batches, y_batches = sample_batches(X_train, Y_traintemp, batch_size, freq_frauds)
    for i in range(total_batch):
        batch_x, batch_y = x_batches[i], y_batches[i]
        _, c = sess.run([optimizer, cost],
                        feed_dict={
                            x: batch_x,
                            y: batch_y,
                            keep_prob: 0.8
                        })
        avg_cost += c / total_batch
    return avg_cost

# Filters all but specified columns from X
def filter_out_columns(X, columns_to_keep):
    X = np.transpose(X)
    X_filtered = []
    for i in columns_to_keep:
        X_filtered.append(X[i])
    return np.transpose(X_filtered)

def FN_cost(predictions, targets, L2loss, gamma, alpha):
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

# https://medium.com/@curiousily/tensorflow-for-hackers-part-ii-building-simple-neural-network-2d6779d2f91b
def tensorFlow(X, Y, batch_size,tolerance,learning_rate,alpha_reg,gamma,freq_frauds,n_hidden_1):
    X, Y = filter_refused_transactions(X, Y)
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = split_train_dev_test(X, Y, 0.7, 0.15)
    X_train, X_dev, X_test = label_X(X_train, X_dev, X_test)
    X_train, X_dev, X_test = scale(X_train, X_dev, X_test)

    X_train = filter_out_columns(X_train, [2, 3, 14, 16])
    X_dev = filter_out_columns(X_dev, [2,3,14,16])

    #poly = PolynomialFeatures(2)
    #X_train = poly.fit_transform(X_train)
    #X_dev = poly.fit_transform(X_dev)

    Y_traintemp =  encode(Y_train)

    (m, n) = X_train.shape
    n_input = X_train.shape[1]
    n_classes = Y_traintemp.shape[1]



    weights_one_layer = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
    }

    biases_one_layer = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    keep_prob = tf.placeholder("float")


    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    predictions = multilayer_perceptron_one_layer(x, weights_one_layer, biases_one_layer, keep_prob)

    weights_fraud = 1/freq_frauds
    weight_settled = 1/(1-freq_frauds)

    alpha = [np.float32(weights_fraud), np.float32(weight_settled)]
    vars = [weights_one_layer['h1'], weights_one_layer['out']]
    L2loss = tf.multiply(alpha_reg, tf.add(tf.nn.l2_loss(weights_one_layer['h1']), tf.nn.l2_loss(weights_one_layer['out'])))
    cost = FN_cost(predictions, y, L2loss, tf.constant(gamma), alpha)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    display_step = 1

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        prev_cost = 0.0
        epoch = 0
        writer = tf.summary.FileWriter('logs', sess.graph)
        while(True):
            avg_cost = run_epoch4(sess, X_train, batch_size, Y_train, optimizer, cost, x, y, keep_prob, freq_frauds)
            if epoch % display_step == 0:
                #print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                #      "{:.9f}".format(avg_cost))
                pass
            if (abs((avg_cost - prev_cost) / avg_cost) <= tolerance):
                break
            else:
                prev_cost = avg_cost
                epoch += 1
        print("Optimization Finished!")
        correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        writer.close()
        print("Training Accuracy:", accuracy.eval({x: X_train, y: Y_traintemp, keep_prob: 1.0}))
        results = correct_prediction.eval({x: X_train, y: Y_traintemp, keep_prob: 1.0})
        acc_train, fraud_train, non_fraud_train = test3(results, Y_train, "NN", 'Train')
        results = correct_prediction.eval({x: X_dev, y: encode(Y_dev), keep_prob: 1.0})
        acc_dev, fraud_dev, non_fraud_dev = test3(results, Y_dev, "NN", 'Dev')
    return acc_train, fraud_train, non_fraud_train, acc_dev, fraud_dev, non_fraud_dev



# Main method
def main():
    X, Y = read_data()
    # Variables to optimise #
    #########################
    batch_size = 10000
    tolerance = 1e-0
    learning_rate = 0.05    #baseline: 0.05
    alpha_reg = 0.001       #baseline: 0.001
    gamma = 2.0             #baseline: 2.0

    freq_frauds = 0.45      #baseline: 0.45
    n_hidden_1 = 300        #baseline: 300
    #########################

    avg_acc_train = 0.0
    avg_fraud_train = 0.0
    avg_nofraud_train = 0.0
    avg_acc_dev = 0.0
    avg_fraud_dev = 0.0
    avg_nofraud_dev = 0.0

    iterate = 5

    for i in range(iterate):
        acc_train, fraud_train, non_fraud_train, acc_dev, fraud_dev, non_fraud_dev = tensorFlow(X, Y, batch_size,tolerance,learning_rate,alpha_reg,gamma,freq_frauds,n_hidden_1)
        avg_acc_train = avg_acc_train + acc_train
        avg_fraud_train = avg_fraud_train + fraud_train
        avg_nofraud_train = avg_nofraud_train + non_fraud_train
        avg_acc_dev = avg_acc_dev + acc_dev
        avg_fraud_dev = avg_fraud_dev + fraud_dev
        avg_nofraud_dev = avg_nofraud_dev + non_fraud_dev

    print("")
    print("Average train accuracy:",avg_acc_train/iterate*100)
    print("Average fraud train accuracy:", avg_fraud_train/iterate)
    print("Average non-fraud train accuracy:", avg_nofraud_train/iterate)
    print("")
    print("Average dev accuracy:",avg_acc_dev/iterate*100)
    print("Average fraud dev accuracy:", avg_fraud_dev/iterate)
    print("Average non-fraud dev accuracy:", avg_nofraud_dev/iterate)
    return

main()