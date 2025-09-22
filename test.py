import numpy as np


def multiple_linear_regression(X, y):
    """
    Вычисляет коэффициенты множественной линейной регрессии

    Parameters:
    X : numpy array, матрица признаков (n_samples, n_features)
    y : numpy array, вектор целевой переменной (n_samples,)

    Returns:
    coefficients : numpy array, коэффициенты регрессии
    """
    # Добавляем столбец единиц для intercept (свободного члена)
    X = np.column_stack([np.ones(X.shape[0]), X])

    # Вычисляем коэффициенты по формуле: (X^T * X)^(-1) * X^T * y
    print(len(X), len(X[0]), len(y))
    coefficients = np.linalg.inv(X.T @ X) @ X.T @ y

    return coefficients


# Пример использования
def main():
    # Генерируем пример данных
    np.random.seed(42)
    n_samples = 100
    n_features = 3

    # Матрица признаков
    X = np.random.randn(n_samples, n_features)

    # Истинные коэффициенты (включая intercept)
    true_coeffs = np.array([2.5, 1.8, -0.9, 0.5])  # [intercept, coef1, coef2, coef3]

    # Генерируем целевую переменную с добавлением шума
    y = true_coeffs[0] + X @ true_coeffs[1:] + np.random.randn(n_samples) * 0.1

    # Вычисляем коэффициенты регрессии
    coefficients = multiple_linear_regression(X, y)

    print("Истинные коэффициенты:", true_coeffs)
    print("Предсказанные коэффициенты:", coefficients)

    # Прогнозирование
    predictions = np.column_stack([np.ones(X.shape[0]), X]) @ coefficients

    # Вычисляем R²
    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    print(f"R²: {r_squared:.4f}")


if __name__ == "__main__":
    main()
