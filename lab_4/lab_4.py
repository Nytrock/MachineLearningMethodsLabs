import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def predict(x, neighbors, distance_function, show_distances=False):
    distances = []
    data = []

    for j in range(len(train_x)):
        distances.append(distance_function(x, train_x[j]))
        data.append(train_y[j])
    distances = np.array(distances)
    if show_distances:
        print('Расстояния:', distances)
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


def show_prediction(x_true, y_true, x_test, y_predicted):
    ax = plt.figure().add_subplot(projection='3d')
    show_data(x_true, y_true, ax, 'green')
    show_data(x_test, y_predicted, ax, 'blue')
    plt.show()


def show_data(x, y, ax, color):
    markers = ['o', 'v']

    x_plt = [[] for _ in markers]
    y_plt = [[] for _ in markers]
    z_plt = [[] for _ in markers]
    for i in range(len(x)):
        object_class = y[i]
        x_plt[object_class].append(x[i][0])
        y_plt[object_class].append(x[i][1])
        z_plt[object_class].append(x[i][2])

    for i in range(len(markers)):
        ax.scatter(x_plt[i], y_plt[i], z_plt[i], color=color, marker=markers[i])


def test_distance_function(function_name, function):
    pred_y = []
    for i in range(len(test_x)):
        pred_y.append(predict(test_x[i], k, function))
    print(function_name)
    print('Точность:', calculate_accuracy(test_y, pred_y))
    show_prediction(train_x, train_y, test_x, pred_y)
    print()


k = int(input('Введите k: '))
x1_name = 'X1-среднее время самостоятельного обучения в день (мин.)'
x2_name = 'X2 - среднее время в день, проведенное в соц сетях. (мин.)'
x3_name = 'X3 – посещаемость занятий за неделю (%)'
y_name = 'Y- наличие задолженностей'

train_x, train_y = get_data('TrainData.xlsx')
test_x, test_y = get_data('TestData.xlsx')

test_distance_function('Евклидово расстояние', euclidian_distance)
test_distance_function('Манхэттенское расстояние', manhattan_distance)
test_distance_function('Косинусное расстояние', cos_distance)
