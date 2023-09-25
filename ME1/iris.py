import pandas as pd
import os

binary_dataset = []
dataset = pd.read_csv('ME1/iris.csv', header=None, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])

index = dataset[dataset['class'] == 'Iris-versicolor'].index

dataset.drop(dataset[dataset['class'] == 'Iris-versicolor'].index, inplace=True)



dataset.drop(['sepal length', 'sepal width'], axis='columns', inplace=True)

dataset.loc[dataset['class'] == 'Iris-setosa', dataset.columns == 'class'] = 0
dataset.loc[dataset['class'] == 'Iris-virginica', dataset.columns == 'class'] = 1

def sigma(x, w):
    activation = -1.0 * w[-1] #bias
    for i in range(len(x) - 1):
        activation += w[i] * x[i]
    return 1.0 if activation >= 0 else 0.0

def training(data, w0, mu, T):
    w = w0
    for idx in range(T):
        for x in data:
            actiavtion = sigma(x, w)
            error = x[-1] - actiavtion
            w[-1] += 1.0 * mu * error

            for i in range(len(x)-1):
                w[i] += mu * error * x[i]
    
    return w

weights = [0.18, 0.24, 0.45]

weights = training(dataset.values, weights, 0.2, 1)

for sample in dataset.values:
    a = sigma(sample, weights)
    print(f"Target: {sample[-1]}, prediction: {a}")

print(f"Final weights: {weights}")