from keras.datasets.mnist import load_data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    (x, y), _ = load_data()
    x = x.reshape(x.shape[0], -1)
    x_train, x_test, y_train, y_test = train_test_split(x[:6000], y[:6000], test_size=0.3, random_state=40)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)
    print('Proportion of samples per class in train set:')
    print(pd.Series(y_train).value_counts(normalize=True))


if __name__ == '__main__':
    main()
