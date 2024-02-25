import numpy as np
from keras.datasets.mnist import load_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def fit_predict_eval(model, x_train, x_test, y_train, y_test, verbose=True):
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    score = accuracy_score(y_test, prediction)
    if verbose:
        print(f'Model: {model}\nAccuracy: {score}\n')
    return score


def main():
    (x, y), _ = load_data()
    x = x.reshape(x.shape[0], -1)
    x_train, x_test, y_train, y_test = train_test_split(x[:6000], y[:6000], test_size=0.3, random_state=40)
    normalizer = Normalizer()
    x_train_norm = normalizer.fit_transform(x_train)
    x_test_norm = normalizer.transform(x_test)

    models = [KNeighborsClassifier(), DecisionTreeClassifier(random_state=40),
              LogisticRegression(random_state=40), RandomForestClassifier(random_state=40)]
    scores = {}
    scores_norm = {}
    for model in models:
        scores[type(model).__name__] = fit_predict_eval(model, x_train, x_test, y_train, y_test, verbose=False)
        scores_norm[type(model).__name__] = fit_predict_eval(model, x_train_norm, x_test_norm, y_train, y_test)
    print('The answer to the 1st question:', 'yes' if sum(scores_norm.values()) > sum(scores.values()) else 'no')
    sn_sorted = sorted(scores_norm, key=scores_norm.get, reverse=True)
    print(f'The answer to the 2nd question: {sn_sorted[0]}-{scores_norm[sn_sorted[0]]:.3f}, '
          f'{sn_sorted[1]}-{scores_norm[sn_sorted[1]]:.3f}')


if __name__ == '__main__':
    main()
