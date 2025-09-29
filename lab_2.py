import numpy as np


def print_function():
    print('y = ', end='')
    for i in range(len(b)):
        print(b[i], end='')
        if i != 0:
            print(f' * x{i}', end='')

        if i != len(b) - 1:
            print(' + ', end='')
    print()


def function(parameters):
    result = 0
    for i in range(len(b)):
        if i == 0:
            result += b[i]
        else:
            result += b[i] * parameters[i - 1]
    return result


def print_table(**kwargs):
    data = []
    for kwarg in kwargs.items():
        column = list(kwarg[1])
        column.insert(0, kwarg[0])
        data.append(column)

    rows_count = len(data[0])
    nums = [i + 1 for i in range(rows_count)]
    nums.insert(0, '№')
    data.insert(0, nums)

    for i in range(rows_count):
        for j in range(len(data)):
            print(data[j][i], end='\t')
        print()


y = np.array([44, 47, 60, 71, 61, 60, 58, 56, 66, 61, 51, 47, 53])
x = np.array([
    [10, 19, 27, 31, 64, 81, 42, 67, 48, 64, 57, 10, 48],
    [22.1, 22.5, 23.1, 24, 22.6, 21.7, 23.8, 22, 22.4, 22.6, 21.1, 22.5, 22.2],
    [4.9, 3, 1.5, 0.6, 1.8, 3.3, 3.2, 2.1, 6, 1.8, 3.8, 4.5, 4.5],
    [0, 1, 0, 3, 2, 1, 0, 0, 1, 1, 0, 1, 0],
    [2.4, 2.6, 2.8, 2.7, 2, 2.5, 2.5, 2.3, 2.8, 3.4, 3, 2.7, 2.8]
])
x = x.transpose()
x_for_calculation = np.column_stack([np.ones(x.shape[0]), x])
x_grave = x_for_calculation.transpose()

n = len(y)
p = len(x[0])
b = np.linalg.inv(x_grave @ x_for_calculation) @ x_grave @ y
print_function()

y_mean = y.mean()
y_calculated = np.array([function(coefs) for coefs in x])
SSR = np.sum((y - y_mean) ** 2)
SSE = np.sum((y - y_calculated) ** 2)
dfR = p
dfE = n - p - 1

MSR = SSR / dfR
MSE = SSE / dfE

F_calculated = MSR / MSE
F_theoretical = 3.02

if F_calculated > F_theoretical:
    print('Модель значимая')
else:
    print('Модель не значимая')
