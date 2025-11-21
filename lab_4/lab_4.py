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


def predict(x, neighbors):
    distances = []
    data = []

    for j in range(len(train_x)):
        distances.append(euclidian_distance(x, train_x[j]))
        data.append(train_y[j])
    distances = np.array(distances)
    data = np.array(data)

    sort_indexes = distances.argsort()
    data = data[sort_indexes]
    data = data[:neighbors]

    classes = []
    for d in data:
        classes.append(d)
    predicted = max(classes, key=classes.count)
    return predicted


def calculate_accuracy(y_true, y_predicted):
    count = 0
    for i in range(len(y_true)):
        if y_true[i] == y_predicted[i]:
            count += 1
    return count / len(y_true)


k = int(input('Введите k: '))
x1_name = 'X1-среднее время самостоятельного обучения в день (мин.)'
x2_name = 'X2 - среднее время в день, проведенное в соц сетях. (мин.)'
x3_name = 'X3 – посещаемость занятий за неделю (%)'
y_name = 'Y- наличие задолженностей'

train_x, train_y = get_data('TrainData.xlsx')
test_x, test_y = get_data('TestData.xlsx')

pred_y = []
for i in range(len(test_x)):
    pred_y.append(predict(test_x[i], k))
print("Accuracy:", calculate_accuracy(test_y, pred_y))
