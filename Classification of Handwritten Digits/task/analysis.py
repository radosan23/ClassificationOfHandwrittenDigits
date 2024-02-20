from keras.datasets.mnist import load_data
from sklearn.model_selection import train_test_split
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

    models = [KNeighborsClassifier(), DecisionTreeClassifier(random_state=40),
              LogisticRegression(random_state=40), RandomForestClassifier(random_state=40)]
    scores = {}
    for model in models:
        scores[type(model).__name__] = fit_predict_eval(model, x_train, x_test, y_train, y_test)
    print(f'The answer to the question: {max(scores)} - {scores[max(scores)]:.3f}')


if __name__ == '__main__':
    main()
