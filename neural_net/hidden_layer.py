
#　レイヤの追加と削除、層の追加と削除
class Hidden_layer:
    def __init__(self,list_layer):
        self.list_layer = list_layer

    def add_layer(self,num):
        self.list_layer.append(num)

    def del_layer(self,idx):
        del self.list_layer[idx]

    def add_neuron(self,num,idx):
        #idxは0スタート
        self.list_layer[idx]+=num

    def del_neuron(self,num,idx):
        self.list_layer[idx]-=num