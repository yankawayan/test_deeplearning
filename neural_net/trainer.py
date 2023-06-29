import numpy as np
import pickle

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

        self.ac_div = 10
        self.ac_border = 0.9
        self.log_ac_expr = '--o'
        self.ac_break_num = 5
        self.ac_buff = 0
        self.lr_break_num = 5
        self.default_lr = [0.00001,0.1]

        self.flag_save = False
        self.flag_log = False
        self.flag_interrupt = False

        self.flag_list_loss = False
        self.flag_use_weight_decay_lambda = True

    def __init_var(self):
        self.accuracy = 0
        self.max_accuracy = 0
        self.tmp_ac = 0
        self.break_ct = 0

    def set_hyper_param(self,key,param):
        self.hyper_params[key] = param

    def __init_hyper_params(self):
        self.hyper_params['hidden_size_list'] = self.network.hidden_size_list
        self.hyper_params['activation'] = 'sigmoid'
        self.hyper_params['weight_init_std'] = 'sigmoid'
        self.hyper_params['weight_decay_lambda'] = 0
        self.hyper_params['use_dropout'] = False
        self.hyper_params['dropout_ration'] = 0.5
        self.hyper_params['use_batchnorm'] = False

        self.hyper_params['train_itr'] = 400
        self.hyper_params['batch_size'] = 100 
        
        self.hyper_params['lr'] = self.optimizer.lr        

    def reinit_network(self):
        self.network.__init__(\
            self.input_size,\
            self.hyper_params['hidden_size_list'],\
            self.output_size,\
            activation=self.hyper_params['activation'],\
            weight_init_std=self.hyper_params['weight_init_std'],\
            weight_decay_lambda=self.hyper_params['weight_decay_lambda'],\
            use_dropout=self.hyper_params['use_dropout'],\
            dropout_ration=self.hyper_params['dropout_ration'],\
            use_batchnorm=self.hyper_params['use_batchnorm']\
        )
        
    def __init_list_train_once(self):
        self.list_loss = []
        self.list_accuracy = []

    def __init_list_search(self):
        self.multi_loss_list = []
        self.multi_accuracy_list = []
        self.max_ac_list = []
        self.weight_decay_list = []

    def __init_var_search_lr(self):
        self.min_lr = self.default_lr[0]
        self.max_lr = self.default_lr[1]
        self.lr_ct = 0
        self.lr_list = []
        self.lr_range_list = []

    def update_lr_min_max(self):
        if self.lr_ct > self.lr_break_num:
            self.min_lr = self.default_lr[0]
            self.max_lr = self.default_lr[1]
            self.lr_ct = 0
            self.ac_buff = 0
        elif self.accuracy > self.ac_buff:
            self.ac_buff = self.accuracy
            self._get_range_from_value(self.accuracy)
        else:
            self.lr_ct += 1
            

    def load_data(self,x_train,t_train,x_test,t_test):
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.train_size = x_train.shape[0]
        self.test_size = x_test.shape[0]

    def save_binary(self,filename):
        save_param = self.network.params
        f = open(str(filename)+".binaryfile","wb")
        pickle.dump(save_param,f)
        f.close

    def save_text(self,filename):
        f = open(str(filename)+'.txt', 'a')
        #TODO:ハイパーパラメータの辞書を作成したので、それを利用する。
        f.write("score: "+str(filename)+\
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

    #HACK:事前にクラスにデータを読み込む必要がある。
    def set_batch_data(self):
        batch_mask = np.random.choice(self.train_size, self.hyper_params['batch_size'])
        self.x_batch = self.x_train[batch_mask]
        self.t_batch = self.t_train[batch_mask]

    def ac_process(self):
        self.accuracy = self.network.accuracy(self.x_test,self.t_test)
        if self.max_accuracy < self.accuracy:
            self.max_accuracy = self.accuracy
        if self.accuracy > self.ac_border:
            if self.flag_save:
                self.save_binary(self.accuracy)
                self.save_text(self.accuracy)

    def log_process(self):
        if self.flag_log:
            if self.max_accuracy == self.accuracy:
                print("update max_accuracy")
            if self.ac_border < self.accuracy:
                print(self.accuracy)
            if self.log_ac_expr == '--o':
                length = "----------"
                idx = int(self.accuracy*10)
                length = length[:idx]+'|'+length[idx+1:]
                print(length)
            elif self.log_ac_expr == 'num':
                print("accuracy : ",self.accuracy)

    def add_ac_to_list(self):
        self.list_accuracy.append(self.accuracy)
            #HACK:損失関数の計算を含むので、必要な時だけ呼び出す。
        if self.flag_list_loss:
            self.list_loss.append(self.network.loss(self.x_batch,self.t_batch))

    def _check_break_num_and_ct(self):
        if  self.tmp_ac < self.accuracy:
            self.tmp_ac = self.accuracy
        else:
            self.break_ct += 1
            if self.ac_break_num < self.break_ct:
                return True
        return False

    #valueからsizeの桁の範囲で最大・最小を返す。
    def _get_range_from_value(self,value,size=1):
        decimal_places = str(value).split('.')[-1]
        ct = 0
        for digit in decimal_places:
            if digit == '0':
                ct += 1
            else:
                break
        current_min = round(round(value,ct+size)-round(0.1**(ct+size),ct+size)/2,ct+size+1)
        if current_min < 0:
            current_min = 0
        current_max = round(round(value,ct+size)+round(0.1**(ct+size),ct+size)/2,ct+size+1)
        return [current_min,current_max]

    def train_onece(self):
        self.__init_var()
        self.__init_list_train_once()
        for i in range(self.hyper_params['train_itr']):
            self.set_batch_data()
            grads = self.network.gradient(self.x_batch,self.t_batch)
            self.optimizer.update(self.network.params,grads)
            if i % self.ac_div == 0:
                self.ac_process()
                self.log_process()
                self.add_ac_to_list()
                if self._check_break_num_and_ct():
                    break

    def search_opt_lr(self,desir_score):
        self.__init_list_search()
        self.__init_var_search_lr()
        while(desir_score < self.max_accuracy):
            if self.flag_use_weight_decay_lambda:
                self.hyper_params['weight_decay_lambda'] = 10 ** np.random.uniform(-8,-6)
                self.reinit_network()
            self.hyper_params['lr'] = np.random.uniform(self.min_lr,self.max_lr)
            self.optimizer.lr = self.hyper_params['lr']
            self.train_onece()
            self.update_lr_min_max()