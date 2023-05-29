from my_newral_net import MyNewralNet
from set_test_data import get_data
from optimizer import SGD,Momentum,AdaGrad,Adam
import numpy as np
import matplotlib.pyplot as plt

(x_train, t_train),(x_test, t_test) = get_data()

train_iteration_num = 300
train_size = x_train.shape[0]
batch_size = 30

weight_decay_lambda = 0.1
weight_decay = 10**np.random.uniform(-8,-4)
lr = 10**np.random.uniform(-6,-2)

network = MyNewralNet(
        input_size=784, hidden_size_list=[100,100,100],
        activation='relu',weight_init_std='relu',
        output_size=10,weight_decay_lambda=weight_decay_lambda)

optimizer = Adam(lr=0.001)

graph_param_train_loss = []

for i in range(train_iteration_num):
    #print('iteration '+str(i))
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch,t_batch)
    optimizer.update(network.params,grads)
    graph_param_train_loss.append(network.loss(x_batch,t_batch))
    if i % 10 == 0:
        print(network.accuracy(x_test,t_test))

x = np.arange(train_iteration_num)
plt.plot(x,graph_param_train_loss,marker="o")
plt.show()