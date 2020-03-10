from cnn_updater import CNN_updater
from cnn_observer import CNN_observer

from slackbot import SlackBot
from dotenv import load_dotenv
from os.path import join

def main():

    batch_size = 50
    epoch = 30

    useSlack = False

    loss_picture_name = "loss.png"
    accuracy_picture_name = "acc.png"
    cnn_model_name = "model.npz"

    observer = CNN_Observer()
    updater = CNN_updater()

    # dotenv用の初期設定 + slackAPIの初期設定
    dotenv_path = join(dirname(__file__), '.env')
    load_dotenv(dotenv_path)
    token = os.environ.get("API_KEY")
    slack = SlackBot(token)

    trainer = CNNTrainer(batch_size, epoch, slack, observer, updater, useSlack, loss_picture_name, accuracy_picture_name, cnn_model_name)

    trainer.learn()

if __name__ == "__main__":
    main()
