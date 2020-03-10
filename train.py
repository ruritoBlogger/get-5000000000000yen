from cnn_updater import CNN_updater
from cnn_observer import CNN_observer
from trainer import Trainer

from slackbot import SlackBot
from dotenv import load_dotenv
import os

def main():

    #バッチサイズ
    batch_size = 50

    #学習回数
    epoch = 30

    #過去何回分のデータを用いて次の回の当選番号を予想するか
    vertical_len = 3

    #slackに通知を飛ばすかどうか
    useSlack = False

    #既に学習させたモデルを使用するかどうか
    is_load_other_model = False

    #誤差の遷移図の名前
    loss_picture_name = "loss.png"
    
    #精度の遷移図の名前
    accuracy_picture_name = "acc.png"

    #保存するモデルの名前
    cnn_model_name = "model.npz"

    #既に学習させたモデルを用いる場合のモデルのパス
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
