import requests

class SlackBot:

    def __init__(self, _token):
        self._token = _token  # api_token
        self._headers = {'Content-Type': 'application/json'}

    def send_message(self, message, channel):
        params = {"token": self._token, "channel": channel, "text": message}

        r = requests.get('https://slack.com/api/chat.postMessage',
                          headers=self._headers,
                          params=params)
        print("return ", r.json())

    def send_picture(self, picture, channel):
        files = {'file': open(picture, 'rb')}
        params = {'token':self._token, 'channels':channel}
        res = requests.post(url="https://slack.com/api/files.upload",params=params, files=files)
