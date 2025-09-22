import numpy as np
import matplotlib.pyplot as plt


def function(value):
    return a0 + a1 * value


x = np.arange(1, 11)
y = np.array([178, 182, 190, 199, 200, 213, 220, 231, 235, 242])

x_sum = x.sum()
y_sum = y.sum()
x_square_sum = np.sum(x * x)
x_y_sum = np.sum(x * y)
n = len(x)

a1 = (n * x_y_sum - y_sum * x_sum) / (n * x_square_sum - x_sum ** 2)
a0 = (y_sum - x_sum * a1) / n

print('a0 =', a0)
print('a1 =', a1)
print('Прогноз:', function(11))

y_calculated = np.vectorize(function)(x)
absolute_error = np.sum(abs(y - y_calculated)) / n
square_error = np.sum((y - y_calculated) ** 2) / n

print('Абсолютная ошибка:', absolute_error)
print('Квадратичная ошибка:', square_error)

plt.scatter(x, y)
plt.plot(x, y_calculated, color='green', marker='o')
plt.show()
