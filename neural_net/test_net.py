import numpy as np
import matplotlib.pyplot as plt

from optimizer import SGD,Momentum,AdaGrad,Adam
from my_newral_net import MyNewralNet
from set_test_data import get_data

import sys,os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
multi_layer_extend_path = os.path.join(parent_dir, 'net_batch_drop_weight', 'multi_layer_extend.py')
sys.path.append(parent_dir)
from net_batch_drop_weight.multi_layer_extend import MultiLayerNetExtend

(x_train, t_train),(x_test, t_test) = get_data()
train_size = x_train.shape[0]

search_hyper_param = 16
check_score = 0.8

train_iteration_num = 400
batch_size = 100

# weight_decay_lambda = 0.01
# lr = 0.01

#activation = 'relu'
activation = 'sigmoid'
#weight_init_std='relu'
weight_init_std='sigmoid'
hidden_size_list=[100]
#optimizer = Adam(lr=lr)
optimizer = SGD()
use_batchnorm=True
use_dropout=False
dropout_ration=0.5

input_size=784
output_size=10
graph_ar = 10
#
graph_title = "learning rate "+str(0)
graph_text = (
    "[ learning_rate             : {:<20} ]\n"
    "[ bach_size                  : {:<20} ]\n"
    "[ epoch                        : {:<20} ]\n"
    "[ hidden_layer             : {:<20}]\n"
    "[ weight_decay_lambda : {:<20} ]\n"
    "[ optimizer                   : {:<20} ]\n"
    "[ activation_func         : {:<20} ]\n"
    "[ batchnorm                : {:<20} ]\n"
    "[ dropout                     : {:<20}: {:<20}]"
).format(
    # str(lr),
    0,
    str(batch_size),
    str(train_iteration_num),
    str(hidden_size_list),
    # str(weight_decay_lambda),
    0,
    str(type(optimizer).__name__),
    activation,
    use_batchnorm,
    use_dropout,
    dropout_ration
)


for k in range(30):
    graph_param_train_loss = []
    graph_param_train_accuracy = []
    graph_wd = []
    graph_lr = []
    graph_score = []
    for j in range(search_hyper_param):
        weight_decay = 10 ** np.random.uniform(-8, -5)
        lr = np.random.uniform(0.04, 0.09)
        print(str(j),"weight_decay: ",weight_decay," lr: ",lr)
        graph_wd.append(weight_decay)
        graph_lr.append(lr)
        network = MultiLayerNetExtend(input_size=input_size, hidden_size_list=hidden_size_list, output_size=output_size, 
                                        weight_init_std=weight_init_std,weight_decay_lambda=weight_decay,use_dropout=use_dropout,dropout_ration=dropout_ration, use_batchnorm=use_batchnorm)
        optimizer.lr = lr
        
        graph_param_train_loss.append([])
        graph_param_train_accuracy.append([])
        graph_score.append(0)

        for i in range(train_iteration_num):
            #print('iteration '+str(i))
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]

            grads = network.gradient(x_batch,t_batch)
            optimizer.update(network.params,grads)
        #
            graph_param_train_loss[j].append(network.loss(x_batch,t_batch))
            if i % graph_ar == 0: 
                accuracy = network.accuracy(x_test,t_test)
                #print("accuracy ",i/graph_ar," : ",accuracy)
                graph_param_train_accuracy[j].append(accuracy)
                if graph_score[j] < accuracy:
                    graph_score[j] = accuracy
        #
    if any(number >= check_score for number in graph_score):
        print("over "+str(check_score)+str(k))
        break
    else:
        print("not "+ str(check_score) +" over "+str(k))


rows = 4  # 行数
cols = int(search_hyper_param/2)  # 列数
x1 = np.arange(train_iteration_num)
x2_range = train_iteration_num/graph_ar
x2 = np.arange(x2_range)
# グラフのサイズと間隔を調整
fig, axes = plt.subplots(rows, cols, figsize=(8, 6))
fig.subplots_adjust(hspace=0.4)

# グラフのプロット
for i in range(search_hyper_param):
    ax1 = axes[int(np.floor(i/8))*2, np.mod(i,8)]
    ax2 = axes[int(np.floor(i/8))*2+1, np.mod(i,8)]
    #print(int(np.floor(i/8)), np.mod(i,8))
    ax1.plot(x1, graph_param_train_loss[i], 'b-')
    ax2.plot(x2, graph_param_train_accuracy[i], 'r-')

    ax1.set_xlabel(str(np.round(graph_lr[i], decimals=5)))
    ax1.set_ylabel(str(graph_score[i]))
    ax2.set_xlabel(str(np.round(graph_wd[i], decimals=8)))
    ax2.set_ylabel(str(i))
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax1.grid(True)
    ax2.grid(True)

# グラフのタイトルとコメントを設定
fig.suptitle(graph_title)
#fig.text(0.3, 0.15, graph_text, ha='left', va='center')

plt.show()