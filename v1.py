import numpy as np
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
from datetime import datetime
import pandas


# Reads the input data from an .npy filie if it exists or from the original csv
# otherwise. Returns npArray

def read_data():
    try:
        return np.load('data.npy')
    except IOError:
        data = pandas.read_csv('data_for_student_case.csv/data_for_student_case.csv').as_matrix()
        for i in range (len(data)):
            data[i, 1] = pandas.to_datetime(data[i, 1])
            data[i, 13] = pandas.to_datetime(data[i, 12])
        np.save('data.npy', data)
        return data

# Splits the data according to their label.
# Returns three sets of indices for Chargeback, Refused and Settled
def generate_Y(data):
    ChargeBack = np.where(data[:, 9]=='Chargeback')[0]
    Refused = np.where(data[:, 9]=='Refused')[0]
    Settled = np.where(data[:, 9]=='Settled')[0]
    return ChargeBack, Refused, Settled

def main():
    data = read_data()
    generate_Y(data)
    return

main()