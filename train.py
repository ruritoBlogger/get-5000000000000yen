

def main():

    batch_size = 50
    epoch = 30

    useSlack = False

    loss_picture_name = "loss.png"
    accuracy_picture_name = "acc.png"

    observer = Observer()
    model = CNN_model()

    # dotenv用の初期設定 + slackAPIの初期設定
    dotenv_path = join(dirname(__file__), '.env')
    load_dotenv(dotenv_path)
    token = os.environ.get("API_KEY")
    slack = SlackBot(token)

    trainer = CNNTrainer(batch_size, epoch, slack, observer, model, useSlack, loss_picture_name, accuracy_picture_name)

    trainer.learn()

if __name__ == "__main__":
    main()
