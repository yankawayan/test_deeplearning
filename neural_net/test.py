from set_test_data import get_data
from optimizer import SGD,Momentum,AdaGrad,Adam
from multi_layer_extend import MultiLayerNetExtend
from trainer import Trainer

optimizer = SGD()
network = MultiLayerNetExtend(input_size=784,\
            hidden_size_list=[100,30],\
            output_size=10,\
            activation='sigmoid',\
            weight_init_std='sigmoid',\
            weight_decay_lambda=0,\
            use_dropout=False,\
            dropout_ration=0.5,\
            use_batchnorm=True)

trainer = Trainer(network,optimizer)

# (x_train, t_train),(x_test, t_test) = get_data()
# trainer.load_data(x_train,t_train,x_test,t_test)
# trainer.search_opt_lr(0.97)

trainer.git_push_filepath("neural_net\\trainer.py")
trainer.git_push_filepath("neural_net\\test.py")