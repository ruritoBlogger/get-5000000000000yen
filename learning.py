
# coding: utf-8

# In[81]:

import chainer
from chainer import training, iterators, optimizers, serializers, Chain
import chainer.functions as F
import chainer.links as L
 
from chainer.training import extensions
from chainer.datasets import tuple_dataset
from chainer import optimizers, Chain, dataset, datasets, iterators
import numpy as np


# In[82]:

def data_read( file_name, key):
    teachers = np.array([] )
    answers =  np.array([] )

    f = open( file_name, mode = "r" )
    f_string = f.readlines()
    data = np.array([] )
    print("test")
    
    for i in range( 0, len( f_string ) ):
        #引数を用いて正解ラベルを振り分ける
        tmp_data = np.zeros(10)
        
        #for j in f_string[i].replace( "\n", "" ):
                #tmp_data[int(j)] += 1
                
        #if( i < key-1):
                #data = np.append( data, tmp_data )
        #else:
            #teachers = np.append( teachers, data )
            #answers = np.append( answers, tmp_data )
            #data = np.delete( data, 0 )
            #data = np.append( data, tmp_data )
        
        #if( (i + 2)%100 == 0 ):
            #print(i/len(f_string))

        if( i != 0 and i%key == 0 ):
            teachers = np.append( teachers, data)
            data = np.array([] )
            for j in f_string[i].replace( "\n", "" ):
                tmp_data[int(j)] += 1
            answers = np.append( answers, tmp_data )
        else:
            for j in f_string[i].replace( "\n", "" ):
                tmp_data[int(j)] += 1
                data = np.append( data, tmp_data )
    
    f.close()
    
    teachers = teachers.astype( np.float32 )
    answers = answers.astype( np.float32 )
    
    teachers = np.reshape( teachers, ( int( len( teachers ) / 10 / key ), key, 10 ) )
    
    answers = np.reshape( answers, ( int( len( answers ) / 10 ) , 10 ) )
    return teachers, answers


# In[ ]:

teachers, answers = data_read( 'numbers.txt', 3)


# In[ ]:

print(len(teachers))
print(len(answers))
print(teachers[0])
print(answers[0])


# In[68]:

#ニューラルネットワークの構築。
class RNN(Chain):
    
    def __init__(self, n_hidden, n_output):
        super(RNN, self).__init__()
        with self.init_scope():
            self.l1=L.LSTM(None, n_hidden).to_gpu(),
            self.l2=L.Linear(None, n_output).to_gpu()
        
    def reset_state(self):
        self.l1.reset_state()
        
    def __call__(self, x, t):
        y = self.predict(x)
        loss = F.mean_squared_error(y, t)
        data = y.data[:]
        accuracy = self.accuracy(data, t)
        chainer.reporter.report({'accuracy':accuracy},self)
        chainer.reporter.report({'loss':loss},self)
        return loss
    
    def accuracy(y, t):
        correct = 0
        for j in range( 0, len(y) ):
            if y[j].size:
                for i in range( 0, 3 ):
                    if( t[j][y[j].argmax()] ):
                        correct += 1
                    y[j][y[j].argmax()]= np.amin(y[j])
        return correct / (len(y) * 3)
        
    
    def predict(self, x):
        if train:
            h1 = F.dropout(self.l1(x),ratio = 0.5)
        else:
            h1 = self.l1(x)
        return self.l2(h1)


# In[69]:

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
        x_batch, t_batch = chainer.dataset.concat_examples(batch, self.device)
        
        optimizer.target.reset_state()           
        optimizer.target.cleargrads()
        #loss = optimizer.target(Variable(x_batch), Variable(t_batch))
        loss = optimizer.target(x_batch, t_batch)
        loss.backward()
        loss.unchain_backward()                  
        optimizer.update() 


# In[70]:

#　教師データのtupleを作成する
data = list(zip(teachers, answers))
N = len(data)
n_batchsize = 30
n_epoch = 1

#モデルを使う準備。オブジェクトを生成
n_hidden = 10
n_output = 10
model = RNN(n_hidden, n_output)
optimizer = optimizers.Adam()
optimizer.setup(model)

#学習用データと検証用データに分ける
train, test = chainer.datasets.split_dataset_random(data, int(N * 0.8))
train_iter = chainer.iterators.SerialIterator(train, n_batchsize, shuffle=False)
test_iter = chainer.iterators.SerialIterator(test, n_batchsize, repeat=False, shuffle=False)
updater = LSTMUpdater(train_iter, optimizer, device=0)
trainer = training.Trainer(updater, (n_epoch, "epoch"), out="result")
trainer.extend(extensions.Evaluator(test_iter, model, device=0))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport( ["epoch", "main/loss", "validation/main/loss", "main/accuracy", "validation/main/accuracy", "elapsed_time"])) # エポック、学習損失、テスト損失、学習正解率、テスト正解率、経過時間
trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
trainer.extend(extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
trainer.run()


# In[27]:

def accuracy(y, t):
        correct = 0
        for j in range( 0, len(y) ):
            if y[j].size:
                for i in range( 0, 3 ):
                    if( t[j][y[j].argmax()] ):
                        correct += 1
                    y[j][y[j].argmax()]= np.amin(y[j])
        print(correct)
        return correct / (len(y) * 3)

y = np.array([[ 2.32572779e-02, 1.50612950e-01, -7.73513615e-02,
            1.24777101e-01, 1.66107967e-01, 4.45912518e-02,
            3.39158834e-03, -3.49103138e-02, 6.63672294e-03,
           -9.04915929e-02],
            [ 2.32572779e-02, 1.50612950e-01, -7.73513615e-02,
            1.24777101e-01, 1.66107967e-01, 4.45912518e-02,
            3.39158834e-03, -3.49103138e-02, 6.63672294e-03,
           -9.04915929e-02]])
t = np.array([[0,1,0,1,1,0,0,0,0,0],[0,1,0,1,1,0,0,0,0,0]])
accuracy(y,t)


# In[ ]:



