import numpy as np

from function import softmax,cross_entropy_error

class Affine:
    def __init__(self,W,b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
#
        self.graph_y = None
    
    def get_graph_y(self):
        return self.graph_y_list
# 

    def forward(self,x):  
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b
        #out = np.dot(x, self.W) + self.b
#
        if np.any(np.isnan(out)):
            print('error in Affine forward nan')
            #print('x:');print(x)
            #print('W[0]:');print(self.W[0])
            #print('b:');print(self.b)
        if np.any(np.isinf(out)):
            print('error in Affine forward inf')
            #print('max x:');print(np.max(x));print('')
        self.graph_y = np.max(x)
            #print('W[0]:');print(self.W[0])
            #print('b:');print(self.b)
#
        return out
    
    def backward(self,dout):
        dx = np.dot(dout,self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
#
        if np.any(np.isnan(dx)):
            print('error in Affine backward nan')
        if np.any(np.isinf(dx)):
            print('error in Affine backward inf')
#
        return dx
    
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self,x):
        out = 1/(1+np.exp(-x))
        self.out = out
        return out
    
    def backward(self,dout):
        dx = dout*(1.0-self.out)*self.out
        return dx

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self,x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self,dout):
        dout[self.mask] = 0
        dx = dout
        return dx
    
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self,x,t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y,self.t)

        return self.loss
    
    def backward(self,dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = (self.y - self.t)/batch_size

        else:
            dx = self.y.copy()
            #dxのnp.arange(batch_size),self.tの要素を-1している。
            dx[np.arange(batch_size),self.t] -= 1
            dx = dx/batch_size
#
        if np.any(np.isnan(dx)):
            print('error in SoftmaxWithLoss backward nan')
        if np.any(np.isinf(dx)):
            print('error in SoftmaxWithLoss backward inf')
#
        return dx
    