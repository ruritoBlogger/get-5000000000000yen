
# coding: utf-8

# In[1]:

import chainer
from chainer import training, iterators, optimizers, serializers, Chain
import chainer.functions as F
import chainer.links as L
 
from chainer.training import extensions
from chainer.datasets import tuple_dataset
from chainer import optimizers, Chain, dataset, datasets, iterators
import numpy as np


# In[13]:

def data_read( file_name, key):
    teachers = []
    answers =  []

    f = open( file_name, mode = "r" )
    f_string = f.readlines()
    data = []
    cnt = 0
    
    for i in range( 0, len( f_string ) ):
        #引数を用いて正解ラベルを振り分ける
        tmp_data = [0] * 10
        for j in f_string[i].replace( "\n", "" ):
                tmp_data[int(j)] += 1
               
        if(False):
            if( i < key-1):
                data.append(tmp_data)
            else:
                teachers.append(data)
                answers.append(tmp_data)
                data.pop(0)
                data.append(tmp_data)
        
            if( (i + 2)%100 == 0 ):
                print(i/len(f_string))
        else:
            if( cnt == key ):
                cnt = 0
                teachers.append(data)
                data = []
                answers.append(tmp_data)
            else:
                cnt += 1
                data.append(tmp_data)
    f.close()
    
    #teachers = teachers.astype( np.float32 )
    #answers = answers.astype( np.float32 )
  
    #整形
    #teachers = np.reshape( teachers, ( int( len( teachers ) / 10 / key ), key, 10) )
    #answers = np.reshape( answers, ( int( len( answers ) / 10 ) , 10 ) )
    
    data = []
    for i in range(int( len( answers) / 10)):
        tmp_data = []
        tmp_teacher = []
        for j in range(key):
            tmp_teacher.append(teachers[i:i+10])
        tmp_data.append(tmp_teacher)
        tmp_data.append(answers[i:i+10])
        #print(tmp_data)
        break
        #print(list(zip(teachers[i:i+1], answers[i:i+10])))
    return teachers, answers


# In[109]:

#ニューラルネットワークの構築。
class CNN(Chain):
    
    def __init__(self, n_output):
        if(GPU == -1):
            super(CNN, self).__init__(
                l1=L.Convolution2D(1,20,3),
                l2=L.Linear(None, n_output)
            )
        else:
            super(CNN, self).__init__(
                l1=L.Convolution2D(1,20,3).to_gpu(),
                l2=L.Linear(None, n_output).to_gpu()
            )   
        
    def __call__(self, x, t):
        y = self.predict(x)
        loss = F.mean_squared_error(y,t)
        #loss = F.softmax_cross_entropy(y,t)
        data = y.data[:]
        accuracy = self.accuracy(data, t)
        chainer.reporter.report({'accuracy':accuracy},self)
        chainer.reporter.report({'loss':loss},self)
        return loss
    
    def accuracy(self, y, t):
        correct = 0
        for j in range( 0, len(y) ):
            if y[j].size:
                for i in range( 0, 3 ):
                    if( t[j][y[j].argmax()] ):
                        correct += 1
                    y[j][y[j].argmax()]= np.amin(y[j])
        return correct / (len(y) * 3)
        
    
    def predict(self, x):
        h1 = F.max_pooling_2d(F.relu(self.l1(x)),2)
        return F.softmax(self.l2(h1))


# In[113]:

#Updaterを拡張する
from chainer import Variable, reporter

class LSTMUpdater(training.StandardUpdater):
    def __init__(self, data_iter, optimizer, device=None):
        super(LSTMUpdater,self).__init__(data_iter, optimizer, device=None)
        self.device = device
        
    def update_core(self):
        data_iter = self.get_iterator("main")
        optimizer = self.get_optimizer("main")
        
        batch = data_iter.__next__()
        data = chainer.dataset.concat_examples(batch, self.device)
        x_batch = data[:][0]
        t_batch = data[:][1]
        x_batch = np.array(x_batch)
        #x_batch = x_batch[:,np.newaxis,:,:]
        t_batch = np.array(t_batch)
        #print(x_batch)
        #x_batch, t_batch = chainer.dataset.concat_examples(batch, self.device)
        #optimizer.target.reset_state()           
        optimizer.target.cleargrads()
        #loss = optimizer.target(Variable(x_batch), Variable(t_batch))
        loss = optimizer.target(x_batch, t_batch)
        loss.backward()
        loss.unchain_backward()                  
        optimizer.update() 


# In[114]:

teachers, answers = data_read( 'numbers.txt', 3)
GPU = -1


# In[115]:

print(len(teachers))
print(len(answers))
print(teachers[-1])
print(answers[1])


# In[116]:

teachers = np.array(teachers)
teachers = teachers.astype(np.float32)[:,np.newaxis,:,:]
print(teachers.shape)
answers = np.array(answers)
answers = answers.astype(np.float32)
data = tuple_dataset.TupleDataset(teachers, answers)


# In[117]:

#　教師データのtupleを作成する
#data = list(zip(teachers, answers))

#data = np.array(data)
N = len(data)
n_batchsize = 30
n_epoch = 10

#モデルを使う準備。オブジェクトを生成
n_output = 10
model = CNN(n_output)
optimizer = optimizers.Adam()
optimizer.setup(model)

#学習用データと検証用データに分ける
train, test = chainer.datasets.split_dataset_random(data, int(N * 0.8))
train_iter = chainer.iterators.SerialIterator(train, n_batchsize, shuffle=False)
test_iter = chainer.iterators.SerialIterator(test, n_batchsize, repeat=False, shuffle=False)
updater = LSTMUpdater(train_iter, optimizer, device=GPU)
trainer = training.Trainer(updater, (n_epoch, "epoch"), out="result")
trainer.extend(extensions.Evaluator(test_iter, model, device=GPU))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport( ["epoch", "main/loss", "validation/main/loss", "main/accuracy", "validation/main/accuracy", "elapsed_time"])) # エポック、学習損失、テスト損失、学習正解率、テスト正解率、経過時間
trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
trainer.extend(extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
trainer.run()


# In[ ]:



