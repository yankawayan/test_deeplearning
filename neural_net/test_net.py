from my_newral_net import MyNewralNet
from set_test_data import get_data
from optimizer import SGD,Momentum,AdaGrad,Adam
import numpy as np
import matplotlib.pyplot as plt

(x_train, t_train),(x_test, t_test) = get_data()
train_size = x_train.shape[0]

train_iteration_num = 3000
batch_size = 20
weight_decay_lambda = 0.01
weight_decay = 0.0001
lr = 0.00001
activation = 'relu'
weight_init_std='relu'
hidden_size_list=[100,100,100]
optimizer = Adam(lr=lr)

input_size=784
output_size=10
graph_ar = 10
graph_param_train_loss = []
graph_param_train_accuracy = []
#
graph_title = "learning rate "+str(lr)
graph_text = (
    "[ learning_rate             : {:<20} ]\n"
    "[ bach_size                  : {:<20} ]\n"
    "[ epoch                        : {:<20} ]\n"
    "[ hidden_layer             : {:<20}]\n"
    "[ eight_decay_lambda : {:<20} ]\n"
    "[ weight_decay            : {:<20} ]\n"
    "[ optimizer                   : {:<20} ]\n"
    "[ activation_func         : {:<20} ]"
).format(
    str(lr),
    str(batch_size),
    str(train_iteration_num),
    str(hidden_size_list),
    str(weight_decay_lambda),
    str(weight_decay),
    str(type(optimizer).__name__),
    activation
)
#
# lr = 10**np.random.uniform(-6,-2)
# weight_decay = 10**np.random.uniform(-8,-4)

network = MyNewralNet(
        input_size=input_size, hidden_size_list=hidden_size_list,
        activation=activation,weight_init_std=weight_init_std,
        output_size=output_size,weight_decay_lambda=weight_decay_lambda)

for i in range(train_iteration_num):
    #print('iteration '+str(i))
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch,t_batch)
    optimizer.update(network.params,grads)
#
    graph_param_train_loss.append(network.loss(x_batch,t_batch))
    if i % graph_ar == 0:
        accuracy = network.accuracy(x_test,t_test)
        print("accuracy ",i/graph_ar," : ",accuracy)
        graph_param_train_accuracy.append(accuracy)
#

x1 = np.arange(train_iteration_num)
x2_range = train_iteration_num/graph_ar
x2 = np.arange(x2_range)
fig, (ax1,ax2) = plt.subplots(2,1,sharex=False)
ax1.plot(x1,graph_param_train_loss,'b-',label='loss')
ax2.plot(x2,graph_param_train_accuracy,'r-',label='accuracy')
fig.suptitle(graph_title)
ax1.set_ylabel('Loss')
ax2.set_ylabel('Accuracy')
ax1.grid(True)
ax2.grid(True)
ax1.legend()
ax2.legend()
fig.subplots_adjust(bottom=0.35)
fig.text(0.3,0.15, graph_text, ha='left', va='center')
plt.show()