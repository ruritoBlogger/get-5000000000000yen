# coding: utf-8

import chainer
from chainer import training, iterators, optimizers, serializers, Chain
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import tuple_dataset
from chainer import optimizers, dataset, datasets, Variable, reporter
import numpy as np
import os

#ニューラルネットワークの構築。
class CNN(Chain):

    def __init__(self, GPU, n_output):
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

#Updaterを拡張する
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


class Learning():

    def __init__(self, key_day, n_batchsize):
        self.key_day = key_day
        self.n_batchsize = n_batchsize

    def data_read(self, file_name, key):
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

        data = []
        for i in range(int( len( answers) / 10)):
            tmp_data = []
            tmp_teacher = []
            for j in range(key):
                tmp_teacher.append(teachers[i:i+10])
            tmp_data.append(tmp_teacher)
            tmp_data.append(answers[i:i+10])
            break
        return teachers, answers

    def save(self):
        if(os.path.isfile('./best.net')):
            os.remove('./best.net')
        chainer.serializers.save_npz('best.net', self.model)

    def run(self):

        teachers, answers = self.data_read( 'numbers.txt', self.key_day)
        GPU = -1

        teachers = np.array(teachers)
        teachers = teachers.astype(np.float32)[:,np.newaxis,:,:]
        answers = np.array(answers)
        answers = answers.astype(np.float32)
        data = tuple_dataset.TupleDataset(teachers, answers)


        N = len(data)
        n_epoch = 11

        #モデルを使う準備。オブジェクトを生成
        n_output = 10
        self.model = CNN(GPU, n_output)
        optimizer = optimizers.Adam()
        optimizer.setup(self.model)

        #学習用データと検証用データに分ける
        train, test = chainer.datasets.split_dataset_random(data, int(N * 0.8))
        train_iter = chainer.iterators.SerialIterator(train, self.n_batchsize, shuffle=False)
        test_iter = chainer.iterators.SerialIterator(test, self.n_batchsize, repeat=False, shuffle=False)
        updater = LSTMUpdater(train_iter, optimizer, device=GPU)
        trainer = training.Trainer(updater, (n_epoch, "epoch"), out="result")
        #trainer.extend(extensions.Evaluator(test_iter, model, device=GPU))
        #trainer.extend(extensions.LogReport(trigger=(10, "epoch")))
        #trainer.extend(extensions.PrintReport( ["epoch", "main/loss", "validation/main/loss", "main/accuracy", "validation/main/accuracy", "elapsed_time"])) # エポック、学習損失、テスト損失、学習正解率、テスト正解率、経過時間
        trainer.run()

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):

            # 少ない検証データの場合
            data = test_iter.next()
            data = chainer.dataset.concat_examples(data, self.model.device)
            test_teach = data[:][0]
            test_ans = data[:][1]
            y_test = self.model.predict(test_teach)
            return self.model.accuracy(y_test.data[:], test_ans)

# ハイパーパラメータの最適化を行う

# ハイパーパラメータの候補の生成
key_list = np.arange(3,50,1)
batchsize_list = np.arange(10,100,10)

accuracy = 0
cnt = 0
# 生成した候補一つ一つで学習を回す 
for key in key_list:
    for batchsize in batchsize_list:
        cnt += 1
        learn = Learning(key, batchsize)
        tmp = learn.run()
        print( cnt/(len(key_list) + len(batchsize_list)) )
        if(accuracy < tmp ):
            accuracy = tmp
            learn.save()
print(accuracy)
