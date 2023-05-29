from layer import Sigmoid,Relu,Affine,SoftmaxWithLoss
from function import numerical_gradient
import numpy as np
from collections import OrderedDict

class MyNewralNet:
    def __init__(self, input_size,hidden_size_list,output_size,activation='relu',weight_init_std='relu',weight_decay_lambda=0):
        """全結合多層ニューラルネット
        input_size : 入力サイズ（MNISTの場合は784）
        hidden_size_list : 隠れ層のニューロンの数のリスト（e.g. [100, 100, 100]）
        output_size : 出力サイズ（MNISTの場合は10）
        activation : 'relu' or 'sigmoid'
        weight_init_std : 重みの標準偏差を指定（e.g. 0.01）
            'relu'または'he'を指定した場合は「Heの初期値」を設定
            'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
        # weight_decay_lambda : Weight Decay（L2ノルム）の強さ

        ---method---
        #メソッド：self.predict(x)
        #返り値：順伝播の出力(恒等関数的出力)
        
        #メソッド：self.loss(x,t)
        #返り値：損失関数(交差エントロピー誤差)の出力

        #メソッド：self.accuracy(x,t)
        #返り値：xとtの一致精度(デシマル表記)

        #メソッド：self.numerical_gradient(x,t)　数値微分
        #返り値：各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス

        #メソッド：gradient(x,t)　誤差逆伝播
        #返り値：各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        ---///---

        """
        self.input_size = input_size
        self.output_size = output_size
        self.hitdden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}
        #重みの初期化(バイアスもだが、0)
        self.__init_weight(weight_init_std)
        #レイヤの生成
        #print {'sigmoid': <class 'layer.Sigmoid'>, 'relu': <class 'layer.Relu'>}
        activation_layer = {'sigmoid':Sigmoid, 'relu':Relu}
        #レイヤ格納用の順序付きリストの作成
        self.layers = OrderedDict()
        #隠れ層の数だけレイヤを作成
        for idx in range(1,self.hidden_layer_num+1):
            self.layers['Affine'+str(idx)] = Affine(self.params['W'+str(idx)],self.params['b'+str(idx)])
            self.layers['Activation_function'+str(idx)] = activation_layer[activation]()

        #出力層のlayer,activation_funcの設定、リストへの追加
        idx = self.hidden_layer_num + 1
        self.layers['Affine'+str(idx)] = Affine(self.params['W'+str(idx)],self.params['b'+str(idx)])
        self.last_layer = SoftmaxWithLoss()

    def  __init_weight(self,weight_init_std):
        # sample [784, 100, 100, 100, 100, 10]
        all_size_list = [self.input_size] + self.hitdden_size_list + [self.output_size]
        # len(all_size_list) - 1 回の繰り返し(バイアスと重みの初期化)
        for idx in range(1, len(all_size_list)):
            #標準偏差と重みに関して
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0/all_size_list[idx-1])
            elif str(weight_init_std).lower() in ('sigmoid','xavier'):
                scale = np.sqrt(1.0/all_size_list[idx-1])
            
            self.params['W'+str(idx)] = scale*np.random.randn(all_size_list[idx-1],all_size_list[idx])
            #biasの初期値が0に設定されている。
            self.params['b'+str(idx)] = np.zeros(all_size_list[idx])

    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x
    
    def loss(self,x,t):
        y = self.predict(x)
        weight_decay = 0
        #直下のfor文はweight_decayの設定。計算の詳細は、
        for idx in range(1,self.hidden_layer_num+2):
            W = self.params['W'+str(idx)]
            weight_decay += 0.5*self.weight_decay_lambda*np.sum(W**2)
        #返り値に荷重減衰(weight_decay)を設定
        return self.last_layer.forward(y,t)+weight_decay
    #accuracy=精度
    def accuracy(self,x,t):
        y = self.predict(x)
        #最大値のインデックスを持った配列にyを上書き
        y = np.argmax(y,axis=1)
        if t.ndim != 1 : t = np.argmax(t,axis=1);
        #yとtで一致する要素のみの和を取り、1入力におけるデータ数で割る
        #yの最大値のインデックスとtの最大値のインデックス
        accuracy = np.sum(y==t)/float(x.shape[0])
        return accuracy

    def numerical_gradient(self,x,t):
        #勾配       
        loss_W = lambda W:self.loss(x,t)

        grads = {}
        for idx in range(1,self.hidden_layer_num+2):
            grads['W'+str(idx)] = numerical_gradient(loss_W,self.params['W'+str(idx)])
            grads['b'+str(idx)] = numerical_gradient(loss_W,self.params['b'+str(idx)])

        return grads
    
    def gradient(self,x,t):
        self.loss(x,t)
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        for idx in range(1,self.hidden_layer_num+2):
            #更新先の重みとバイアスのリストを所持？恐らく。
            grads['W'+str(idx)] = self.layers['Affine'+str(idx)].dW + self.weight_decay_lambda*self.layers['Affine'+str(idx)].W
            grads['b'+str(idx)] = self.layers['Affine'+str(idx)].db

        return grads