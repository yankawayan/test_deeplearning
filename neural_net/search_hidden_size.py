import sys,os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
multi_layer_extend_path = os.path.join(parent_dir, 'net_batch_drop_weight', 'multi_layer_extend.py')
sys.path.append(parent_dir)

from net_batch_drop_weight.multi_layer_extend import MultiLayerNetExtend
"""
想定としては、何回か学習し、精度が上がらない場合に違うものに変更する
ただ、精度を上げていきたいが、評価方法が曖昧。
パターン
・層の数を増減
・ニューロンの数を増減
・層の数に対応し、活性化関数と初期値を変更
・学習率と荷重減衰は、増減をtrainerクラスで実行している
そのため、優先度としてはtrainerクラスの実行で、その後、精度の改善が見られない場合に層とニューロンを変更する。
層とニューロンの変更は1次的に増減させて、固定、その後再度trainerクラスでパラメータを探索。
"""
hidden_size_list=[100,50,30]


network = MultiLayerNetExtend(input_size=input_size, hidden_size_list=hidden_size_list, output_size=output_size, 
                                        weight_init_std=weight_init_std,weight_decay_lambda=weight_decay,use_dropout=use_dropout,dropout_ration=dropout_ration, use_batchnorm=use_batchnorm)
