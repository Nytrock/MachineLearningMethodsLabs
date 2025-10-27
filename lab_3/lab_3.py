import csv
import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x_f, b_f):
    return 1 / (1 + np.exp(-(x_f @ b_f)))


def logistic_regression(x_f, y_f, lr=0.01, epochs=1500):
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
print('Коэффициенты логистической регрессии:', b)

y_pred = []
for i in range(len(x_test)):
    x_t = np.insert(x_test[i], 0, 1)
    prediction = sigmoid(x_t, b)
    y_pred.append(prediction)
y_pred = np.array(y_pred)

print()
print('Реальные результаты:     ', y_test)
print('Предсказанные результаты:', np.round(y_pred).astype(np.int32))

TP = 0
FN = 0
FP = 0
TN = 0
for i in range(len(y_test)):
    if y_test[i] == round(y_pred[i]):
        if y_test[i] == 1:
            TP += 1
        else:
            TN += 1
    else:
        if y_test[i] == 1:
            FN += 1
        else:
            FP += 1

print()
print('Матрица ошибок:')
print('               \t Actual = 0 \t Actual = 1')
print(f'Predicted = 0 \t {TN}(TN)     \t {FN}(FN)')
print(f'Predicted = 1 \t {FP}(FP)      \t {TP}(TP)')
print()

ACC = (TP + TN) / (TP + TN + FN + FP)
print('Меткость:', ACC, '(цель: 1)')

PPV = TP / (TP + FP)
print('Точность:', PPV, '(цель: 1)')

TPR = TP / (TP + FN)
print('Полнота:', TPR, '(цель: 1)')

TNR = TN / (TN + FP)
print('Специфичность:', TNR, '(цель: 1)')

LogRoss = (-1 / len(y_test)) * np.sum(y_test * np.log(y_pred) + (1 - y_test) * np.log(1 - y_pred))
print('Функция потерь логистической регрессии:', LogRoss, '(цель: 0)')

F1 = (2 * PPV * TPR) / (PPV + TPR)
print('F1-мера:', F1)

if F1 > 1:
    print('Полнота имеет приоритет над точностью')
else:
    print('Точность имеет приоритет над полнотой')

P4 = (4 * TP * TN) / (4 * TP * TN + (TP + TN) * (FP + FN))
print('Метрика P4:', P4, '(цель: 1)')

MCC = (TP * TN - FP * FN) / (np.sqrt((TP + FP) * (TP + TN) * (TN + FP) * (TN + FN)))
print('Коэффициент корреляции Мэтьюса:', MCC, '(цель: 1)')

FPR = 1 - TNR
y_roc = np.array([0, TPR, 1])
x_roc = np.array([0, FPR, 1])

AUC_ROC = np.trapezoid(y_roc, x_roc)
print('Площадь под ROC-кривой:', AUC_ROC, '(цель: 1)')

y_pr = np.array([1, PPV, 0])
x_pr = np.array([0, TPR, 1])
AUC_PR = np.trapezoid(y_pr, x_pr)
print('Площадь под кривой полнота-точность:', AUC_PR, '(цель: 1)')

plt.plot(x_roc, y_roc, color='green')
plt.plot(x_pr, y_pr, color='red')
plt.show()
