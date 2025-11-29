import math
import pandas as pd


def get_entropy(p, p_i):
    s = 0
    for i in range(len(p_i)):
        chance = p_i[i] / p
        s -= chance * math.log2(chance)
    return s


def get_gini(p, p_i):
    s = 1
    for i in range(len(p_i)):
        chance = p_i[i] / p
        s -= math.pow(chance, 2)
    return s


class Node:
    def __init__(self, uncertainty_method):
        self.y = None
        self.x = None
        self.uncertainty_method = uncertainty_method
        self.feature = ''
        self.children = dict()
        self.uncertainty = 0

    def train(self, x, y):
        self.x = x
        self.y = y

        features = x[0].keys()
        features_scores = dict()
        features_accuracies = dict()

        for feature in features:
            uncertainty, child_accuracies = self.test_feature(feature)
            features_scores[feature] = uncertainty
            features_accuracies[feature] = child_accuracies
        best_feature = min(features_scores, key=features_scores.get)

        self.feature = best_feature
        child_accuracies = features_accuracies[best_feature]
        values = set()
        for x in self.x:
            values.add(x[best_feature])

        children_data = {}
        for i in range(len(self.x)):
            x_value = self.x[i][best_feature]
            x = self.x[i].copy()
            x.pop(best_feature)
            y = self.y[i]

            if children_data.get(x_value) is None:
                children_data[x_value] = {'x': [x], 'y': [y], 'uncertainty': child_accuracies[x_value]}
            else:
                children_data[x_value]['x'].append(x)
                children_data[x_value]['y'].append(y)

        for key, data in children_data.items():
            classes = set(data['y'])
            self.children[key] = dict()

            if len(classes) == 1:
                self.children[key]['value'] = classes.pop()
                self.children[key]['uncertainty'] = data['uncertainty']
            else:
                child = Node(self.uncertainty_method)
                child.train(data['x'], data['y'])
                self.children[key]['value'] = child
                self.children[key]['uncertainty'] = data['uncertainty']


    def test_feature(self, feature):
        branches = dict()
        for i in range(len(self.x)):
            value = self.x[i][feature]
            if branches.get(value) is None:
                branches[value] = [i]
            else:
                branches[value].append(i)

        child_accuracies = dict()
        uncertainty = 0

        for value, x in branches.items():
            x_count = len(x)
            results_count = dict()
            for i in x:
                result = self.y[i]
                if results_count.get(result) is None:
                    results_count[result] = 1
                else:
                    results_count[result] += 1

            branch_uncertainty = self.uncertainty_method(x_count, list(results_count.values()))
            child_accuracies[value] = branch_uncertainty
            uncertainty += branch_uncertainty * x_count / len(self.x)
        return uncertainty, child_accuracies

    def show(self, level=0):
        print(' ' * level + self.feature)
        for key, child_data in self.children.items():
            child = child_data['value']
            uncertainty = child_data['uncertainty']

            print(' ' * (level + 1) + f'{key} ({uncertainty}): ', end='')
            if isinstance(child, Node):
                print()
                child.show(level + 3)
            else:
                print(child)

    def predict(self, x_test):
        value = x_test[self.feature]
        child = self.children[value]['value']

        if isinstance(child, Node):
            return child.predict(x_test)
        else:
            return child


def main():
    data = pd.read_excel('data.xlsx')
    data = data.drop(columns=data.columns[0])

    y = data[data.columns[0]].tolist()
    data = data.drop(columns=data.columns[0])
    x = data.to_dict('records')

    print('Дерево с использованием энтропии')
    root = Node(get_entropy)
    root.train(x, y)
    root.show()

    print('Предсказание:', root.predict(
        {'Кредитная история (K)': 'Неизвестно', 'Долг (Дг)': 'Низкий',
         'Поручительство (П)': 'Нет', 'Доход (Дд)': '15-35'}))

    print()
    print('Дерево с использованием индекса Джинни')
    root = Node(get_gini)
    root.train(x, y)
    root.show()

    print('Предсказание:', root.predict(
        {'Кредитная история (K)': 'Неизвестно', 'Долг (Дг)': 'Низкий', 'Поручительство (П)': 'Нет',
         'Доход (Дд)': '15-35'}))


if __name__ == '__main__':
    main()
