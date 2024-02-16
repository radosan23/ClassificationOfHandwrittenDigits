from keras.datasets.mnist import load_data
import numpy as np


def main():
    (x_train, y_train), test = load_data()
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
    print("Classes:", np.unique(y_train))
    print("Features' shape:", x_train.shape)
    print("Target's shape:", y_train.shape)
    print(f"min: {np.min(x_train)}, max: {np.max(x_train)}")


if __name__ == '__main__':
    main()
