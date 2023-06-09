from PIL import Image
import numpy as np

# 1次元配列を作成（例としてランダムな値を使用）
array_1d = np.random.randint(0, 255, 360*360)

# 1次元配列を2次元配列に変換
array_2d = array_1d.reshape((360, 360))

# 2次元配列をPILイメージに変換
image = Image.fromarray(array_2d.astype(np.uint8))

# 画像を保存
image.save('output.jpg')
