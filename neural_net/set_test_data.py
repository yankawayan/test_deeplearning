import numpy as np
import tensorflow as tf
from function import shuffle_dataset

# tensolflowからのデータの取得と整頓
def arr_num_to_oneHot(arr):
    t = np.zeros((len(arr), 10))
    for i in range(len(arr)):
        t[i, arr[i]] = 1
    return t

def get_data():
    (x_train, t_train), (x_test, t_test) = tf.keras.datasets.mnist.load_data()
    x_train,t_train = shuffle_dataset(x_train,t_train)

    x1 = x_train.reshape(x_train.shape[0], -1)[:60000]
    x2 = x_test.reshape(x_test.shape[0], -1)[:10000]
    t1 = arr_num_to_oneHot(t_train)
    t2 = arr_num_to_oneHot(t_test)
    return (x1, t1),(x2, t2)