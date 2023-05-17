import numpy as np

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x)
#
    if np.any(np.isnan(exp_x / sum_exp_x)):
        print('error in softmax nan')
    if np.any(np.isinf(exp_x / sum_exp_x)):
        print('error in softmax inf')
#
    return exp_x / sum_exp_x

def cross_entropy_error(y,t):
    if y.ndim == 1:
        y = y.reshape(1,y.size)
        t = t.reshape(1,t.size)
    #教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        #最大値のインデックスを求める
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
#
    if np.any(np.isnan(-np.sum(np.log(y[np.arange(batch_size),t]+1e-7))/batch_size)):
        print('error in cross_entropy_error nan')
    if np.any(np.isinf(-np.sum(np.log(y[np.arange(batch_size),t]+1e-7))/batch_size)):
        print('error in cross_entropy_error inf')
#
    return -np.sum(np.log(y[np.arange(batch_size),t]+1e-7))/batch_size

def numerical_gradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x)
    #配列を反復処理するためのイテレータを作成。イテレータは配列の要素に順番にアクセスするためのオブジェクト
    it = np.nditer(x,flags=['multi_index'],op_flags=['readwrite'])
    #イテレータオブジェクトのit全ての要素に対する繰り返し。
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val
        #次の要素に進む。
        it.iternext()

    return grad