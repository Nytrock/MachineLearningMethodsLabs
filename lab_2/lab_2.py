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
        try:
            column = list(kwarg[1])
        except TypeError:
            column = [kwarg[1]]
        title = kwarg[0]
        if len(title) < len(str(column[0])):
            need_spaces = (len(str(column[0])) - len(title)) // 2
            title = ' ' * need_spaces + title + ' ' * need_spaces
        column.insert(0, title)
        data.append(column)

    rows_count = 0
    column_count = 0
    for column in data:
        if len(column) > rows_count:
            rows_count = len(column)
        column_count += 1

    for i in range(len(data)):
        column = data[i]
        if len(column) == rows_count:
            continue

        old_column = list(column)
        new_column = ['' for _ in range(rows_count)]
        title = old_column.pop(0)
        new_column[0] = title

        if len(old_column) == 1:
            new_column[rows_count // 2] = old_column[0]
        else:
            pass

        data[i] = new_column

    print()
    for i in range(rows_count):
        for j in range(column_count):
            print(data[j][i], end='\t')
        print()
    print()


y = np.array([44, 47, 60, 71, 61, 60, 58, 56, 66, 61, 51, 47, 53])
x = np.array([
    [10, 19, 27, 31, 64, 81, 42, 67, 48, 64, 57, 10, 48],
    [22.1, 22.5, 23.1, 24, 22.6, 21.7, 23.8, 22, 22.4, 22.6, 21.1, 22.5, 22.2],
    [4.9, 3, 1.5, 0.6, 1.8, 3.3, 3.2, 2.1, 6, 1.8, 3.8, 4.5, 4.5],
    [0, 1, 0, 3, 2, 1, 0, 0, 1, 1, 0, 1, 0],
    [2.4, 2.6, 2.8, 2.7, 2, 2.5, 2.5, 2.3, 2.8, 3.4, 3, 2.7, 2.8]
])
x_mean = np.array([0.0 for _ in range(len(x))])
for i in range(len(x_mean)):
    x_mean[i] = x[i].mean()

x = x.transpose()
x_for_calculation = np.column_stack([np.ones(x.shape[0]), x])
x_grave = x_for_calculation.transpose()

n = len(y)
p = len(x[0])
b = np.linalg.inv(x_grave @ x_for_calculation) @ x_grave @ y
print_function()

y_mean = y.mean()
y_calculated = np.array([function(coefs) for coefs in x])
SSR = np.sum((y_calculated - y_mean) ** 2)
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

t_crit = 2.3646
x_error = (x - x_mean) ** 2
x_error = x_error.transpose()
x_sums = []

for i in x_error:
    x_sums.append(np.sum(i))
x_sums = np.array(x_sums)

SE = (MSE / x_sums) ** 0.5
T = b[1:] / SE
for i in range(len(T)):
    print(f'Коэффициент b{i + 1} ', end='')
    if abs(T[i]) > t_crit:
        print('значимый')
    else:
        print('не значимый')


print_table(y=y, y_calculated=y_calculated, y_mean=y_mean, SSR=SSR, SSE=SSE,
            dfR=dfR, dfE=dfE, MSR=MSR, MSE=MSE, F_calculated=F_calculated)
print_table(b=b[1:], x_sums=x_sums, SE=SE, T=T)
