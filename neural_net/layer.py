import numpy as np

from function import softmax,cross_entropy_error,im2col,col2im

class Pooling:
    def __init__(self,pool_h,pool_w,stride=2,pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self,x):
        N,C,H,W = x.shape
        out_h = int(1+(H-self.pool_h)/self.stride)
        out_w = int(1+(W-self.pool_w)/self.stride)
        #展開
        col = im2col(x,self.pool_h,self.pool_w,self.stride,self.pad)
        col = col.reshape(-1,self.pool_h*self.pool_w)

        #最大値
        out = np.max(col,axis=1)
        #整形
        out = out.reshape(N,out_h,out_w,C).transpose(0,3,1,2)

        return out
    
    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx

class Convolution:
    def __init__(self,W,b,stride=1,pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 中間データ（backward時に使用）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None

    def forward(self,x):
        FN,C,FH,FW = self.W.shape
        N,C,H,W = x.shape
        out_h = int(1+(H+2*self.pad-FH)/self.stride)
        out_w = int(1+(W+2*self.pad-FW)/self.stride)

        col = im2col(x,FH,FW,self.stride,self.pad)
        col_W = self.W.reshape(FN,-1).T
        out = np.dot(col,col_W) + self.b

        out = out.reshape(N,out_h,out_w,-1).transpose(0,3,1,2)

        return out
    
    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx

class Affine:
    def __init__(self,W,b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

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
    