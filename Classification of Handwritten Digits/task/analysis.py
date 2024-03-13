from keras.datasets.mnist import load_data
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def fit_predict_eval(model, x_train, x_test, y_train, y_test, verbose=False):
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    score = accuracy_score(y_test, prediction)
    if verbose:
        print(f'Model: {model}\nAccuracy: {score}\n')
    return score


def find_best_models(models, x_train, x_train_norm, x_test, x_test_norm, y_train, y_test, verbose=False):
    scores = {}
    scores_norm = {}
    for model in models:
        scores[type(model).__name__] = fit_predict_eval(model, x_train, x_test, y_train, y_test)
        scores_norm[type(model).__name__] = fit_predict_eval(model, x_train_norm, x_test_norm, y_train, y_test)
    sn_sorted = sorted(scores_norm, key=scores_norm.get, reverse=True)
    if verbose:
        print(f'Best models: {sn_sorted[0]}-{scores_norm[sn_sorted[0]]:.3f}, '
              f'{sn_sorted[1]}-{scores_norm[sn_sorted[1]]:.3f}')


def main():
    # load, split, normalize data
    (x, y), _ = load_data()
    x = x.reshape(x.shape[0], -1)
    x_train, x_test, y_train, y_test = train_test_split(x[:6000], y[:6000], test_size=0.3, random_state=40)
    normalizer = Normalizer()
    x_train_norm = normalizer.fit_transform(x_train)
    x_test_norm = normalizer.transform(x_test)

    models = [KNeighborsClassifier(), DecisionTreeClassifier(random_state=40),
              LogisticRegression(random_state=40), RandomForestClassifier(random_state=40)]
    find_best_models(models, x_train, x_train_norm, x_test, x_test_norm, y_train, y_test, verbose=True)

    # hyperparameter grid search for 2 best models
    knn_params = {'n_neighbors': [3, 4], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'brute']}
    rf_params = {'n_estimators': [300, 500], 'max_features': ['sqrt', 'log2'],
                 'class_weight': ['balanced', 'balanced_subsample']}
    knn_grid = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=knn_params, scoring='accuracy', n_jobs=-1)
    rf_grid = GridSearchCV(estimator=RandomForestClassifier(random_state=40), param_grid=rf_params,
                           scoring='accuracy', n_jobs=-1)
    knn_acc = fit_predict_eval(knn_grid, x_train_norm, x_test_norm, y_train, y_test)
    rf_acc = fit_predict_eval(rf_grid, x_train_norm, x_test_norm, y_train, y_test)

    print('\nK-nearest neighbours algorithm\nbest estimator:', knn_grid.best_estimator_, '\naccuracy:', knn_acc)
    print('\nRandom forest algorithm\nbest estimator:', rf_grid.best_estimator_, '\naccuracy:', rf_acc)


if __name__ == '__main__':
    main()
