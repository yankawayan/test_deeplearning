import numpy as np
import pickle
from search_batch_size import Batch_size
from function import get_range_for_value

"""
Trainerクラス
データの読み込み、データからnetworkの重み、バイアスの更新(更新にoptimizerを使用)

    trainer.train(x_train,t_train)
    #データの中身について要検討
"""

class Trainer:
    def __init__(self,network,optimizer):
        self.network = network
        self.input_size = network.input_size
        self.output_size = network.output_size

        self.hyper_params = {}
        self.__init_hyper_params()

        self.optimizer = optimizer
        self.lr = optimizer.lr

        self.accuracy = None
        self.ac_div = 10
        self.ac_border = 0.9

        self.flag_save = False
        self.flag_log = False
        self.flag_interrupt = False

    def set_hyper_param(self,key,param):
        self.hyper_params[key] = param

    def __init_hyper_params(self):
        self.hyper_params['train_itr'] = 400
        self.hyper_params['batch_size'] = 100
            #HACK:既に作成されたネットワークのactivationとweight_init_stdは不明
        self.hyper_params['activation'] = 'sigmoid'
        self.hyper_params['weight_init_std'] = 'sigmoid'
            #HACK:取ってくる必要性が分からない。どちらでも良いか？
        self.hyper_params['hidden_size_list'] = self.network.hidden_size_list
        self.hyper_params['weight_decay_lambda'] = self.network.weight_decay_lambda
            #HACK:以下二つは、dropoutに関するものだから、セットにできそうならする。
        self.hyper_params['use_dropout'] = False
        self.hyper_params['dropout_ration'] = 0.5
        self.hyper_params['use_batchnorm'] = False

    def reinit_network(self):
        self.network.__init__(self.input_size,self.hyper_params['hidden_size_list'], self.output_size,\
            activation=self.hyper_params['activation'],weight_init_std=self.hyper_params['weight_init_std'],\
            weight_decay_lambda=self.hyper_params['weight_decay_lambda'],\
            use_dropout=self.hyper_params['use_dropout'],dropout_ration=self.hyper_params['dropout_ration'],\
            use_batchnorm=self.hyper_params['use_batchnorm'])
        
    def __init_once_list(self):
        #１回分の学習をリストで保存
        self.loss_list = []
        self.accuracy_list = []
        self.max_ac = 0
        self.lr_range = []

    def __init_multi_list(self):
        #複数回の学習を２次元のリストで保存
        self.multi_loss_list = []
        self.multi_accuracy_list = []
        self.max_ac_list = []
        self.lr_list = []
        self.weight_decay_list = []
        self.lr_range_list = []

    def load_data(self,x_train,t_train,x_test,t_test):
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.train_size = x_train.shape[0]
        self.test_size = x_test.shape[0]

    def save_binary(self,accuracy):
        save_param = self.network.params
        f = open("sc_"+str(accuracy)+".binaryfile","wb")
        pickle.dump(save_param,f)
        f.close

    def save_text(self,accuracy):
        f = open("sc_"+str(accuracy)+'.txt', 'a')
        f.write("score: "+str(accuracy)+\
                ", weight_decay: "+str(self.network.weight_decay_lambda)+\
                ", lr: "+str(self.optimizer.lr)+\
                ", input_size: "+str(self.network.input_size)+\
                ", hidden_size_list: "+str(self.network.hidden_size_list)+\
                ", output_size: "+str(self.output_size)+\
                ", activation: "+str(self.network.activation)+\
                ", weight_init_std: "+str(self.network.weight_init_std)+\
                ", use_dropout: "+str(self.network.use_dropout)+\
                ", dropout_ration: "+str(self.network.dropout_ration)+\
                ", use_batchnorm: "+str(self.network.use_batchnorm)+\
                "\n")
        f.close()

    def get_batch_data(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        return x_batch,t_batch

    def ac_process(self):
        self.accuracy = self.network.accuracy(self.x_test,self.t_test)
        #精度リストへの追加
        self.accuracy_list.append(self.accuracy)
        #損失関数の計算とリストへの追加
        if loss_list_flag:
            self.loss_list.append(self.network.loss(x_batch,t_batch))
        #最大精度の更新
        if self.max_ac < self.accuracy:
            self.max_ac = self.accuracy
        #必要スコア以上の精度の時、精度数値の出力とバイナリファイルの保存
        if self.accuracy > self.score_ac:
            # スコア探索用の範囲を保存
            if len(self.lr_range) == 0:
                self.lr_range = get_range_for_value(self.optimizer.lr,lr_range_rank=lr_range_rank)
            if check_flag:
                print(self.accuracy)
            if save_flag:
                self.save_binary(self.accuracy)
                self.save_text(self.accuracy)
        #精度のチェック(スコアに関わらず)
        if check_flag:
            if ac_check == '--o':
                #精度を---------oで表現
                length = "----------"
                idx = int(self.accuracy*10)
                length = length[:idx]+'|'+length[idx+1:]
                print(length)
            elif ac_check == 'num':
                print("accuracy",i/self.ac_div," : ",self.accuracy)

    def check_ac(self,buff,ct):
        if  buff < self.accuracy:
            buff = self.accuracy
        else:
            ct += 1
            if ct > break_ac_num:
                self.train_itr_num = (i+1)
                return True
        return False
        

    def train_onece(self,save_flag=True,\
                    check_flag=False,loss_list_flag=False,ac_check='--o',\
                    break_flag=False,break_ac_num=5,lr_range_rank=1):
        
        self.__init_once_list()

        buff=0,ct=0
        for i in range(self.train_itr):
            x_data,t_data = self.get_batch_data()
            grads = self.network.gradient(x_data,t_data)
            self.optimizer.update(self.network.params,grads)

            if i % self.ac_div == 0:
                self.ac_process()
                if self.check_ac(buff,ct):
                    break

    def train_multi(self,train_num,\
                    save_flag=True,check_flag=False,loss_list_flag=False,\
                    ac_check='--o',low_lr=0.00001,high_lr=0.1,low_wd=-8,high_wd=-5,\
                    break_flag=False,break_ac_num=5,\
                    lr_range_rank=2):
        """
        できること:訓練を複数回実行
                    最適なパラメータの探索(lr,weight_decay)
                    #隠れ層・ニューロンの数の探索は検討中
        train_num:学習の繰り返し回数(毎回ハイパーパラメータを変更)
        train_itr:epoch回数(学習回数)
        batch_size:バッチサイズ
        score_ac=0.9:要求精度スコア
        check_flag=True
        loss_list_flag=False
        # 探索用パラメータの範囲設定
        # learning late
        low_lr=0.00001
        high_lr=0.1
        # weight_decay (荷重減衰)
        low_wd=-8
        high_wd=-5
        """
        self.__init_multi_list()
        for i in range(train_num):
#
            #探索方法は要検討
            #weight_decayとlrを決定、リストへ保存、設定
            weight_decay = 10 ** np.random.uniform(low_wd,high_wd)
            self.weight_decay_list.append(weight_decay)
            #隠れ層の変更にどう対応するか？
            self.network.weight_decay_lambda = weight_decay
            lr = np.random.uniform(low_lr,high_lr)
            self.lr_list.append(lr)
            self.optimizer.lr = lr
            #設定したパラメータで学習
            self.train_onece(save_flag=save_flag,\
                            check_flag=check_flag,loss_list_flag=loss_list_flag,\
                            ac_check=ac_check,\
                            break_flag=break_flag,break_ac_num=break_ac_num,\
                            lr_range_rank=lr_range_rank)
            #学習した精度の推移リストを保存(複数の学習結果を一つのリスト(２次元)で保存)
            if len(self.lr_range) != 0:
                self.lr_range_list.append(self.lr_range)
            self.multi_accuracy_list.append(self.accuracy_list)
            if loss_list_flag:
                self.multi_loss_list.append(self.loss_list)

    def search_part_param(self,start_train_num=16,start_lr=0.0001,start_train_itr=10000,\
                          start_ac_div=10,start_score=0.3):
        """
        やりたいこと：一部パラメータの自動探索
                train_itr:学習率によって回数を増減させたい。基準は精度の降下手前。
                (異常の検知が必要な可能性アリ)
                batch_size:おおよそ10～100
                ac_div:訓練データの総数による。基本10程度
                score_ac:初期の精度の中から高いものを設定、精度の上昇幅から再調整。
        最終的に欲しいデータ：
            重みとバイアスの最適なパラメータ
            その学習率と荷重減衰、その他調整したハイパーパラメータ
        探索手順
            output,訓練データの数からbatch_sizeの初期値を設定
            lrを低い値で実行、train_itrの初期値を設定
            lrを広範囲で実行、精度が相対的に高かったlrを選択、範囲を限定。(複数の範囲を保存)
            score_acを設定
            精度の下降箇所(変動幅が狭い、幅の設定方法の検討、上昇幅と比較？)からtrain_itrを再設定
            繰り返し、再設定したlrで実行、範囲の限定
            精度の上昇が一定回数起こらなかった場合、lrの指定範囲を変更。
            保存した範囲のlrで目標精度を達成できなかった場合、層及びニューロンの追加。
            層とニューロンの追加に関しては、要検討
        batch_sizeについて
            outputの2倍以上は欲しい。
        """
#　クラスごとの保持パラメータについてよく考える。
#　再帰などを使用すれば、もっと効率化できたかも。要改善。
        self.BATCH_SIZE = Batch_size(self.output_size,self.train_size)
        self.optimizer.lr = start_lr
        #break_flagにより、self.train_itr_numに値を代入
        self.train_onece(start_train_num,start_train_itr,self.BATCH_SIZE.num,\
                        ac_div=start_ac_div,\
                        save_flag=False,break_flag=True,break_ac_num=5)
        # lrの範囲を決定し、保存するために、学習を実行
        self.train_multi(start_train_num,self.train_itr_num,\
                        self.BATCH_SIZE.num,ac_div=start_ac_div,\
                        score_ac=start_score,save_flag=False,\
                        low_lr=start_lr,high_lr=0.1,lr_range_rank=1)
        #scoreを超えた範囲を限定->scoreを再設定->実行->範囲を限定を繰り返し
        #self.lr_range_listにある、スコア越えのリストから、更にスコアを更新してlrのリストも更新する。
        #リストをもとにscore以上のlrを探索、発見したら縮小した範囲でリストに上書き
        #発見できなかった場合、（削除/保留）

        for i in range(len(self.lr_range_list)):
            self.train_multi(start_train_num,self.train_itr_num,\
                             self.BATCH_SIZE.num,)