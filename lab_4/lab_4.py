import numpy as np
import pandas as pd


def euclidian_distance(a, b):
    return np.sum(np.pow(a - b, 2))


def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))


def cos_distance(a, b):
    return 1 - (np.sum(a * b) / (np.sum(a * a) * np.sum(b * b)))


def get_data(table_name):
    data = pd.read_excel(table_name)
    x = np.array(data.loc[:, x1_name:x3_name])
    y = np.array(data.get(y_name))
    return x, y


def predict(x):
    distances = []
    for x_t in train_x:
        distances.append(euclidian_distance(x, x_t))
    distances = np.array(distances)
    sort_indexes = distances.argsort()
    return sort_indexes


k = int(input('Введите k: '))
x1_name = 'X1-среднее время самостоятельного обучения в день (мин.)'
x2_name = 'X2 - среднее время в день, проведенное в соц сетях. (мин.)'
x3_name = 'X3 – посещаемость занятий за неделю (%)'
y_name = 'Y- наличие задолженностей'

train_x, train_y = get_data('TrainData.xlsx')
test_x, test_y = get_data('TestData.xlsx')

for i in range(len(test_x)):
    y_pred = predict(test_x[i])
    print(y_pred)
