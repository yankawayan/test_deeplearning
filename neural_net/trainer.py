import numpy as np
import pickle

class Trainer:
    """
    引数：network,optimizer
    関数： 
        # データの読み込み用関数
         load_train_test(x_train,t_train,x_test,t_test)
        # １回学習
         train_onece(train_itr,batch_size,ac_div,score_ac,save_flag,check_flag,loss_list_flag,ac_check)
        # (train_num)回学習、lrとweight_decayを設定(np.randomを使用)
         train_multi(train_num,train_itr,batch_size,ac_div,score_ac,save_flag,check_flag,loss_list_flag,ac_check,low_lr,high_lr,low_wd,high_wd)
    パラメータ
        (loss_list_flagに依存)
        # １回学習時のリスト
            self.loss_list
            self.accuracy_list
            (self.optimizer.lr,
            self.network.weight_decay_lambda)
        # (train_num)回学習時のリスト(multiは2次元)
            self.multi_loss_list
            self.multi_accuracy_list
            self.lr_list
            self.weight_decay_list
    """
    def __init__(self,network,optimizer):
        self.network = network
        self.optimizer = optimizer
        #
        self.x_train = None
        self.t_train = None
        self.x_test = None
        self.t_test = None
        self.train_size = None
        self.test_size = None
        #
        self.save_param = None

        #リストのパラメータ
        self.__init_once_list()
        self.__init_multi_list()

    def __init_once_list(self):
        #１回分の学習をリストで保存
        self.loss_list = []
        self.accuracy_list = []
        self.max_ac = 0

    def __init_multi_list(self):
        #複数回の学習を２次元のリストで保存
        self.multi_loss_list = []
        self.multi_accuracy_list = []
        self.max_ac_list = []
        self.lr_list = []
        self.weight_decay_list = []
#
    #訓練データ、テストデータの読み込み(要改良)
    def load_train_test(self,x_train,t_train,x_test,t_test):
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.train_size = x_train.shape[0]
        self.test_size = x_test.shape[0]

    def train_onece(self,train_itr,batch_size,ac_div=10,score_ac=0.9,save_flag=True,check_flag=False,loss_list_flag=False,ac_check='--o'):
        """
        できること：訓練の回数とバッチのサイズを指定し、学習を行う(重みとバイアスの更新をする)。
                    精度と損失関数の推移のリストを作成する。
                    最大精度を保存する。
                    指定した精度以上の場合の重みとバイアスを保存する。
                    精度を表示する。(数字と図を指定)

        train_itr:訓練回数
        batch_size:バッチサイズ
        ac_div=10:精度の計算頻度(ac_div回に1回)
        score_ac=0.9:要求精度スコア
        save_flag=True:重みとバイアスの保存フラグ、score_acに依存
        check_flag=False:チェック表示フラグ、精度をターミナルで出力(数値or図)
        loss_list_flag=False,,ac_check='--o'):精度の表示方法を選択、数値='num'、図='--o'
        """
        self.__init_once_list()
        for i in range(train_itr):
#
            #バッチを使うか、ランダムにするか否か(要検討)
            batch_mask = np.random.choice(self.train_size, batch_size)
            x_batch = self.x_train[batch_mask]
            t_batch = self.t_train[batch_mask]
#
            #確率的勾配降下法(要検討)
            grads = self.network.gradient(x_batch,t_batch)
            self.optimizer.update(self.network.params,grads)
            #精度の計算と処理
            if i % ac_div == 0: 
                accuracy = self.network.accuracy(self.x_test,self.t_test)    
                #精度リストへの追加
                self.accuracy_list.append(accuracy)
                #損失関数の計算とリストへの追加
                if loss_list_flag:
                    self.loss_list.append(self.network.loss(x_batch,t_batch))
                #最大精度の更新
                if self.max_ac < accuracy:
                    self.max_ac = accuracy
                #必要スコア以上の精度の処理
                if accuracy > score_ac:
                    if check_flag:
                        print(accuracy)
                    #重みとバイアスをバイナリファイルで保存
                    #各パラメータをテキストデータで保存
                    if save_flag:
                        self.save_param = self.network.params
                        f = open("sc_"+str(accuracy)+".binaryfile","wb")
                        pickle.dump(self.save_param,f)
                        f.close
                        f = open("sc_"+str(accuracy)+'.txt', 'a')
                        f.write("score: "+str(accuracy)+\
                                ", weight_decay: "+str(self.network.weight_decay_lambda)+\
                                ", lr: "+str(self.optimizer.lr)+\
                                ", input_size: "+str(self.network.input_size)+\
                                ", hidden_size_list: "+str(self.network.hidden_size_list)+\
                                ", output_size: "+str(self.network.output_size)+\
                                ", activation: "+str(self.network.activation)+\
                                ", weight_init_std: "+str(self.network.weight_init_std)+\
                                ", use_dropout: "+str(self.network.use_dropout)+\
                                ", dropout_ration: "+str(self.network.dropout_ration)+\
                                ", use_batchnorm: "+str(self.network.use_batchnorm)+\
                                "\n")
                        f.close()
                #精度のチェック(スコアに関わらず)
                if check_flag:
                    if ac_check == '--o':
                        #精度を---------oで表現
                        length = "----------"
                        idx = int(accuracy*10)
                        length = length[:idx]+'|'+length[idx+1:]
                        print(length)
                    elif ac_check == 'num':
                        print("accuracy",i/ac_div," : ",accuracy)

    def train_multi(self,train_num,train_itr,batch_size,ac_div=10,score_ac=0.9,save_flag=True,check_flag=True,loss_list_flag=False,ac_check='--o',low_lr=0.00001,high_lr=0.1,low_wd=-8,high_wd=-5):
        """
        できること:訓練を複数回実行
                    最適なパラメータの探索(lr,weight_decay)#隠れ層・ニューロンの数の探索は検討中
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
            self.network.weight_decay_lambda = weight_decay
            lr = np.random.uniform(low_lr,high_lr)
            self.lr_list.append(lr)
            self.optimizer.lr = lr
            #設定したパラメータで学習
            self.train_onece(train_itr=train_itr,batch_size=batch_size,ac_div=ac_div,score_ac=score_ac,save_flag=save_flag,check_flag=check_flag,loss_list_flag=loss_list_flag,ac_check=ac_check)
            #学習した精度の推移リストを保存(複数の学習結果を一つのリスト(２次元)で保存)
            self.multi_accuracy_list.append(self.accuracy_list)
            if loss_list_flag:
                self.multi_loss_list.append(self.loss_list)