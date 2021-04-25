import numpy as np


def read_data(path):
    cities = []
    with open(path, 'r') as f:
        for data in f:
            data = data.replace('\n', '')
            data = data.split(' ')
            cities.append([eval(data[1]), eval(data[2])])
    cities = np.array(cities)
    return cities
