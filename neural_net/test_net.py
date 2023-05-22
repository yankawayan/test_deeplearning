"""
2023年5月17日時点で、
・計算途中にinfが発生し、学習がうまくいっていない。
→発生箇所：おおよそ6回目の学習におけるAffineレイヤforward、そのため、おそらく行列の積部分

→未解決理由：値が徐々に増加しているため、どの部分が影響しているのか不明瞭。
学習率を0.01から0.05、0.001にしたが、学習回数を増加させると、再度infが発生。

→原因考察
重みが増加しつづけている？
バイアスが増加し続けている？
レイヤのパラメータに間違いが存在する？

"""

from my_newral_net import MyNewralNet
from set_test_data import get_data
from optimizer import SGD,Momentum,AdaGrad,Adam
import numpy as np
import matplotlib.pyplot as plt

(x_train, t_train),(x_test, t_test) = get_data()

train_iteration_num = 10
train_size = x_train.shape[0]
batch_size = 10
# weight decay（荷重減衰）の設定 =======================
#weight_decay_lambda = 0 # weight decayを使用しない場合
weight_decay_lambda = 0.1
# ====================================================
weight_decay = 10**np.random.uniform(-8,-4)
lr = 10**np.random.uniform(-6,-2)


network = MyNewralNet(
        input_size=784, hidden_size_list=[100, 100, 100, 100],
        output_size=10,weight_decay_lambda=weight_decay_lambda)

optimizer = Momentum()

graph_param_train_loss = []

#→バッチ作成→学習→勾配→精度→更新→バッチ更新→学習→・・・
for i in range(train_iteration_num):
    print('iteration '+str(i))
    #バッチ作成
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch,t_batch)
    optimizer.update(network.params,grads)

    loss = network.loss(x_batch,t_batch)
    graph_param_train_loss.append(loss)

x = np.arange(train_iteration_num)
plt.plot(x,graph_param_train_loss,marker="o")
plt.show()

#精度の途中経過の確認(出力部は削除済)
# network.accuracy(x_train,t_train)
# y:
# [[ 122.80914061 -169.06967115  240.96122019 ... -112.28591095
#   -269.79487476   25.8599475 ]
#  [  53.73216531 -268.39196741  -66.76771369 ...  -30.48498138
#   -359.39839155   52.72545203]
#  [   8.50327682 -166.24574823   88.68635888 ...  -19.93876706
#   -152.44616478    1.18164477]
#  ...
#  [ 115.60037417 -204.21259474  229.81712383 ... -100.94829751
#   -282.05602383  -26.17468307]
#  [  56.00552142 -162.36516074   79.33373394 ... -130.57014361
#   -281.45712493   21.96989973]
#  [  57.03050219 -117.41185309   75.51714434 ... -133.46475379
#   -263.44975338   42.82158484]]
# y:
# [2 6 6 ... 6 6 2]
# t:
# [[0. 0. 0. ... 0. 0. 0.]
#  [1. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  ...
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 1. 0.]]
# t:
# [5 0 4 ... 5 6 8]
# accuracy:
# 0.08731666666666667

#レイヤの確認
# for layer in network.layers.values():
#     print(layer)
# <layer.Affine object at 0x000001B40C1BAC20>
# <layer.Relu object at 0x000001B40C1BAC50>
# <layer.Affine object at 0x000001B40C1BACB0>
# <layer.Relu object at 0x000001B40C1BACE0>
# <layer.Affine object at 0x000001B40C1BB340>
# <layer.Relu object at 0x000001B40C1BB9D0>
# <layer.Affine object at 0x000001B40C1BB970>
# <layer.Relu object at 0x000001B4248033A0>
# <layer.Affine object at 0x000001B4248033D0>