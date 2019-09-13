import numpy as np
import gzip

def load():
    image_size = 28 * 28
    num_train = 60000
    num_test = 10000
    train_imagef = gzip.open('dataset/train-images-idx3-ubyte.gz','r')
    test_imagef = gzip.open('dataset/t10k-images-idx3-ubyte.gz','r')
    train_labelf = gzip.open('dataset/train-labels-idx1-ubyte.gz','r')
    test_labelf = gzip.open('dataset/t10k-labels-idx1-ubyte.gz','r')

    train_imagef.read(16)
    train_labelf.read(8)
    test_imagef.read(16)
    test_labelf.read(8)

    train_imageb = train_imagef.read(image_size * num_train)
    train_labelb = train_labelf.read(num_train)
    test_imageb = test_imagef.read(image_size * num_test)
    test_labelb = test_labelf.read(num_test)

    x_train = np.frombuffer(train_imageb, dtype=np.uint8).astype(np.float32)
    x_train = np.reshape(x_train, (num_train, image_size))

    x_test = np.frombuffer(test_imageb, dtype=np.uint8).astype(np.float32)
    x_test = np.reshape(x_test, (num_test, image_size))

    y_train = np.frombuffer(train_labelb, dtype=np.uint8).astype(np.int64)
    y_test = np.frombuffer(test_labelb, dtype=np.uint8).astype(np.int64)

    return (x_train, y_train),(x_test, y_test)


def one_hot_encode(y_train, y_test, num_classes=10):
    one_hot_train = np.zeros((len(y_train), num_classes))
    one_hot_test = np.zeros((len(y_test), num_classes))
    for i in range(len(y_train)):
        one_hot_train[i, y_train[i]] = 1
    for i in range(len(y_test)):
        one_hot_test[i, y_test[i]] = 1
    return one_hot_train, one_hot_test
