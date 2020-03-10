

class CNN_updater():

    def __init__(self, file_name):
        self.file_name = file_name
        self.reset()

    def reset(self):
        self._model = None
        self._optimizer = None

    def initialize(self, use_other_model, other_model_name=""):

        self._model = CNN()

        if(use_other_model):
            self.model.load(other_model_name)

        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

    def save(self, file_name):
        serializers.save_npz(file_name, self.model)

    def update(self, teach, ans):
        predict = self.model.forward(teach)
        loss_val = F.softmax_cross_entropy(predict, ans)
        self.model.cleargrads()
        loss_val.backward()
        self.optimizer.update()

    def debug(self, teach):
        return self.model.debug_forward(teach)
