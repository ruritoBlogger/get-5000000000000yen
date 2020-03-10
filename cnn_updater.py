from cnn_model import CNN
import chainer.functions as F
from chainer import serializers, optimizers
import numpy as np


class CNN_updater():

    def __init__(self):
        self.reset()

    def reset(self):
        self._model = None
        self._optimizer = None

    def initialize(self, is_load_other_model, other_model_name=""):

        self._model = CNN()

        if(is_load_other_model):
            serializers.load_npz(other_model_name, self.model)

        self._optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

    def save(self, model_name):
        serializers.save_npz(model_name, self.model)

    def update(self, teach, ans):
        predict = self.model.forward(teach)
        ans = np.array(ans)
        for i in range(3):
            tmp_ans = np.array([0] * len(ans))
            for j in range(len(ans)):
                tmp_ans[j] = ans[j][i]

            loss_val = F.softmax_cross_entropy(predict, tmp_ans)
            self.model.cleargrads()
            loss_val.backward()
            self.optimizer.update()

    def evaluate(self, teach, ans):
        predict = self.model.forward(teach)
        ans = np.array(ans)
        loss = 0
        accuracy = 0
        num = len(teach)

        for i in range(3):
            tmp_ans = np.array([0] * len(ans))
            for j in range(len(ans)):
                tmp_ans[j] = ans[j][i]

            loss_val = F.softmax_cross_entropy(predict, tmp_ans)
            accuracy_val = F.accuracy(predict, tmp_ans)
            loss += loss_val.data
            accuracy += accuracy_val.data

        return loss/num, accuracy/num

    def debug(self, teach):
        return self.model.debug_forward(teach)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        print("please don't overwrite model instance")

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        print("please don't overwrite optimizer instance")
