from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def arr_to_img_and_show(arr,height,width):
    if arr.ndim == 1:
    # 1次元配列を2次元配列に変換
        image_matrix = np.reshape(arr, (height, width))
    elif arr.ndim == 2:
        image_matrix = arr

    # 画像を表示
    plt.imshow(image_matrix, cmap='gray')
    plt.axis('off')
    plt.show()

from set_test_data import get_data

(x_train, t_train),(x_test, t_test) = get_data()

# print(x_train.shape[0])
# print(x_test.shape[0])

num = np.random.randint(0, x_train.shape[0])
indices = np.where(t_train[num] == 1)[0]
print(indices)
arr_to_img_and_show(x_train[num],28,28)