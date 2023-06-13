
class Batch_size:
    def __init__(self,output,train_size):
        self.output = output
        self.train_size = train_size
        self.serch_param = 0
        return 
     
    def deci_batch_size(self):
            # 探索方法が未定なため、検討が必要
            if 2*self.output < self.train_size:
                return 2*self.output
            else:
                return self.output
            
    def update_batch_size(self):
         