import pandas as pd

TOTAL_KEY = 'Total'
COUNT_KEY = 'Count'


def create_dict_from_set(s):
    result = dict()
    for key in set(s):
        result[key] = 0
    return result


def calculate_frequency(a, b):
    result = dict()

    result[TOTAL_KEY] = create_dict_from_set(a)
    result[TOTAL_KEY] |= create_dict_from_set(b)
    result[TOTAL_KEY][COUNT_KEY]  = 0

    for i in range(len(a)):
        if result.get(b[i]) is None:
            result[b[i]] = create_dict_from_set(a)
        result[b[i]][a[i]] += 1
        result[TOTAL_KEY][a[i]] += 1
        result[TOTAL_KEY][b[i]] += 1
        result[TOTAL_KEY][COUNT_KEY] += 1
    return result


def predict(frequency, weather, result):
    pBA = frequency[weather][result] / frequency[TOTAL_KEY][result]
    print(f'P({weather}|{result}) = {pBA}')

    pB = frequency[TOTAL_KEY][weather] / frequency[TOTAL_KEY][COUNT_KEY]
    print(f'P({weather}) = {pB}')

    pA = frequency[TOTAL_KEY][result] / frequency[TOTAL_KEY][COUNT_KEY]
    print(f'P({result}) = {pA}')

    return pBA * pA / pB


def main():
    data = pd.read_excel('data.xlsx')
    b = data[data.columns[0]].tolist()
    a = data[data.columns[1]].tolist()
    frequency = calculate_frequency(a, b)

    print('Виды погоды:', list(set(b)))
    weather = input('Погода: ')
    chance = predict(frequency, weather, 'Yes')
    print('Шанс проведения матча:', chance)


if __name__ == '__main__':
    main()
