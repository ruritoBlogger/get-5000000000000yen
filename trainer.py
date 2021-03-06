import matplotlib.pyplot as plt

class Trainer():

    def __init__(self, batch_size, epoch, slack, observer, updater, useSlack, loss_picture_name, acc_picture_name, cnn_model_name):
        self.batch_size = batch_size
        self.epoch = epoch
        self.useSlack = useSlack
        self.loss_picture_name = loss_picture_name
        self.acc_picture_name = acc_picture_name
        self.cnn_model_name = cnn_model_name

        self._slack = slack
        self._observer = observer
        self._updater = updater

        self.loss_list = []
        self.acc_list = []

    def learn(self):
        data = self.observer.get_data("./numbers.txt")
        teach, ans = self.observer.transform(data)
        
        key = int(len(teach)*0.8)
        train_teach = teach[:key]
        train_ans = ans[:key]

        test_teach = teach[key:]
        test_ans = ans[key:]

        for j in range( self.epoch ):

            for i in range( int(len(train_teach) / self.batch_size ) ):
                batch_teach = train_teach[i: i+self.batch_size]
                batch_ans = train_ans[i: i+self.batch_size]

                if( len(batch_teach) == 0 ):
                    break

                self.updater.update(batch_teach, batch_ans)
            
            loss, accuracy = self.updater.evaluate(test_teach, test_ans)

            self.loss_list.append(loss)
            self.acc_list.append(accuracy)
            self.show_process(j+1, loss, accuracy)
            self.send_message(j)

        self.make_picture()
        self.save_model()

    def show_process(self, epoch, loss, accuracy):
        print("[{}] loss:{}, accuracy:{}".format(epoch, loss, accuracy))

    def send_message(self, episode):
        message = ""
        message += "現在{0}エピソード\n".format(episode)
        if(self.useSlack): 
            self.slack.send_message(message, "#random")

    def make_picture(self):
        plt.figure()
        plt.plot(self.loss_list, label="loss")
        plt.savefig(self.loss_picture_name)

        plt.figure()
        plt.plot(self.acc_list, label="acc")
        plt.savefig(self.acc_picture_name)

    def save_model(self):
        self.updater.save(self.cnn_model_name)

    @property
    def slack(self):
        return self._slack

    @slack.setter
    def slack(self, slack):
        print("please don't overwrite slack instance")

    @property
    def observer(self):
        return self._observer

    @observer.setter
    def observer(self, observer):
        print("please don't overwrite observer instance")

    @property
    def updater(self):
        return self._updater

    @updater.setter
    def updater(self, updater):
        print("please don't overwrite updater instance")
