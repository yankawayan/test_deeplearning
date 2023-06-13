from trainer import Trainer
from optimizer import SGD,Momentum,AdaGrad,Adam
from multi_layer_extend import MultiLayerNetExtend
from set_test_data import get_data
(x_train, t_train),(x_test, t_test) = get_data()
from hidden_layer import Hidden_layer
"""
想定としては、何回か学習し、精度が上がらない場合に違うものに変更する
ただ、精度を上げていきたいが、評価方法が曖昧。
パターン
・層の数を増減
・ニューロンの数を増減
・層の数に対応し、""活性化関数と初期値""を変更
・学習率と荷重減衰は、増減をtrainerクラスで実行している
そのため、優先度としてはtrainerクラスの実行で、その後、精度の改善が見られない場合に層とニューロンを変更する。
層とニューロンの変更は一次的に増減させて、固定、その後再度trainerクラスでパラメータを探索。
    初期のニューロンの数と層の数は、入力の数、出力の数、使用ネットワーク(畳み込み等)から検討。
"""

hidden_layer=Hidden_layer([100,50,30])

input_size = 784
output_size = 10
activation = 'relu'
weight_init_std = 'relu'
weight_decay_lambda=0
use_dropout = False
dropout_ration = 0.5
use_batchnorm = True


network = MultiLayerNetExtend(input_size=input_size, hidden_size_list=hidden_layer.list_layer,\
                            output_size=output_size,activation=activation,\
                            weight_init_std=weight_init_std,\
                            weight_decay_lambda=weight_decay_lambda,\
                            use_dropout=use_dropout,dropout_ration=dropout_ration,\
                            use_batchnorm=use_batchnorm)
trainer = Trainer(network,SGD)
trainer.load_train_test(x_train,t_train,x_test,t_test)
trainer.train_multi(16,500,100)