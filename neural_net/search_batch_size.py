
class Batch_size:
    """
    やりたいこと:バッチサイズの調整
    必要なパラメータ:データサイズ、出力サイズ、精度
    学習率、レイヤサイズ、等との調整が必要。
    """
    def __init__(self,output,train_size,start_size=100):
        self.output = output
        self.train_size = train_size
        self.num = start_size
    
    def twice_output_batch_size(self):
            # 探索方法が未定なため、検討が必要
            if 2*self.output < self.train_size:
                return 2*self.output
            else:
                return self.output
            
    def add_ten_batch_size(self):
         if (self.num + 10) < self.train_size:
              self.num += 10
              return self.num
        
    def sub_ten_batch_size(self):
         if (self.num - 10) > 0:
              self.num -= 10
              return self.num