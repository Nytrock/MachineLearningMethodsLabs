import csv
import numpy as np


def sigmoid(x_f, b_f):
    return 1 / (1 + np.exp(-(x_f @ b_f)))


def logistic_regression(x_f, y_f, lr=0.01, epochs=10000):
    n_samples, n_features = x_f.shape
    x_f = np.hstack([np.ones((n_samples, 1)), x_f])
    w = np.zeros(x_f.shape[1])

    for _ in range(epochs):
        y_pred = sigmoid(x_f, w)
        gradient = (1 / n_samples) * (x_f.T @ (y_pred - y_f))
        w -= lr * gradient
    return w


features_names = ['Возраст', 'Среднемес. доход', 'Площадь квартиры', 'Сумма кредита',
                  'Стоимость кредита', 'Срок кредита']
features_indexes = []
result_index = -1
x = []
y = []

with open('Credit.txt', newline='', encoding='windows-1251') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    for i, row in enumerate(reader):
        if i == 0:
            for name in features_names:
                features_indexes.append(row.index(name))
            result_index = row.index('Давать кредит(число)')
        else:
            needed_features = [float(row[i]) for i in range(len(row)) if i in features_indexes]
            x.append(needed_features)
            y.append(int(row[result_index]))

print(x)
x = np.array(x)
y = np.array(y)

x = x.transpose()
for i in range(len(x)):
    f_min = np.min(x[i])
    f_max = np.max(x[i])
    x[i] = (x[i] - f_min) / (f_max - f_min)
x = x.transpose()

train_data_part = 0.8
x_train_data_part = int(len(x) * train_data_part)
x_train, x_test = x[:x_train_data_part], x[x_train_data_part:]
y_train_data_part = int(len(y) * train_data_part)
y_train, y_test = y[:y_train_data_part], y[y_train_data_part:]

b = logistic_regression(x_train, y_train)
print("Коэффициенты логической регрессии:", b)

ones = np.ones((len(b), 1))
for i in range(len(x_test)):
    print(x_test[i])
    x_t = np.insert(x_test[i], 0, 1)
    y_pred = sigmoid(x_t, b)
    print(y_test[i], y_pred)
