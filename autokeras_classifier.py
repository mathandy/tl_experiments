# import autokeras as ak
# from keras.datasets import cifar10
# from keras import freeze_support
# if __name__ == '__main__':
#     freeze_support()

#     (x_train, y_train), (x_test, y_test) = cifar10.load_data()
#     clf = ak.ImageClassifier()
#     clf.fit(x_train, y_train)
#     results = clf.predict(x_test)


import autokeras as ak
import os
import numpy as np
import random


def load_quickdraw_data(npy_dir='NPYs', image_shape=(28, 28)):
    class_names = []
    image_subsets = []
    labels = []
    npys = [fn for fn in os.listdir(npy_dir) if fn.endswith('.npy')]
    n_classes = len(npys)
    for k, npy in enumerate(npys):
        class_names.append(os.path.splitext(npy)[0])
        class_samples = np.load(os.path.join(npy_dir, npy))
        image_subsets.append(class_samples.reshape((-1, ) + image_shape + (1,)))
        label = [0]*n_classes
        label[k] = 1
        labels += [label]*len(class_samples)
    return np.vstack(image_subsets), np.array(labels), class_names


def split_data(X, Y, testpart, validpart=0, shuffle=True):
    """Split data into training, validation, and test sets.
    Args:
        X: any sliceable iterable
        Y: any sliceable iterable
        validpart: int or float proportion
        testpart: int or float proportion
        shuffle: bool
    """
    m = len(Y)

    # shuffle data
    if shuffle:
        permutation = list(range(m))
        random.shuffle(permutation)
        X = X[permutation]
        Y = Y[permutation]

    if 0 <= validpart < 1 and 0 < testpart < 1:
        m_valid = int(validpart * m)
        m_test = int(testpart * m)
        m_train = len(Y) - m_valid - m_test
    else:
        m_valid = validpart
        m_test = testpart
        m_train = m - m_valid - m_test

    X_train = X[:m_train]
    Y_train = Y[:m_train]

    X_valid = X[m_train: m_train + m_valid]
    Y_valid = Y[m_train: m_train + m_valid]

    X_test = X[m_train + m_valid: len(X)]
    Y_test = Y[m_train + m_valid: len(Y)]

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test


if __name__ == "__main__":
    x, y, class_names = load_quickdraw_data()
    X_train, Y_train, _, _, X_test, Y_test = split_data(x, y, 0.2)
    clf = ak.ImageClassifier(verbose=True, augment=False)
    clf.fit(X_train, np.argmin(Y_train, axis=1))
    clf.load_searcher().load_best_model().produce_keras_model().save('autokeras_model.h5')
    y = clf.evaluate(X_test, Y_test)
    print(y)
