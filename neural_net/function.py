import numpy as np

def get_range_for_value(value,lr_range_rank=1):
    #与えられた数値の位の最小値、一つ上の位の最小値のリストを返す。
    #小数点以下の数値の長さ
    decimal_places = str(value).split('.')[-1]
    #0の個数(最初の桁が小数第何位か)
    ct = 0
    for digit in decimal_places:
        if digit == '0':
            ct += 1
        else:
            break
#   論理エラーに注意
    current_min = round(round(value,ct+lr_range_rank)-round(0.1**(ct+lr_range_rank),ct+lr_range_rank)/2,ct+lr_range_rank+1)
    if current_min < 0:
        current_min = 0
    current_max = round(round(value,ct+lr_range_rank)+round(0.1**(ct+lr_range_rank),ct+lr_range_rank)/2,ct+lr_range_rank+1)
    return [current_min,current_max]

def shuffle_dataset(x, t):
    """データセットのシャッフルを行う

    Parameters
    ----------
    x : 訓練データ
    t : 教師データ

    Returns
    -------
    x, t : シャッフルを行った訓練データと教師データ
    """
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]

    return x, t

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド
    pad : パディング

    Returns
    -------
    col : 2次元配列
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 入力データの形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

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