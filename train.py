from cnn_updater import CNN_updater
from cnn_observer import CNN_observer
from trainer import Trainer

from slackbot import SlackBot
from dotenv import load_dotenv
import os

def main():

    batch_size = 50
    epoch = 30
    vertical_len = 3

    useSlack = False
    is_load_other_model = False

    loss_picture_name = "loss.png"
    accuracy_picture_name = "acc.png"
    cnn_model_name = "model.npz"
    use_cnn_model_name = "model.npz"

    observer = CNN_observer(vertical_len)
    updater = CNN_updater()
    updater.initialize(is_load_other_model, use_cnn_model_name)

    # dotenv用の初期設定 + slackAPIの初期設定
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(dotenv_path)
    token = os.environ.get("API_KEY")
    slack = SlackBot(token)

    trainer = Trainer(batch_size, epoch, slack, observer, updater, useSlack, loss_picture_name, accuracy_picture_name, cnn_model_name)

    trainer.learn()

if __name__ == "__main__":
    main()
