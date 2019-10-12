from bs4 import BeautifulSoup
import pandas as pd
import requests
from datetime import datetime
import time

#持ってくるナンバーズ3の過去のデータ
#first_dataが1であれば1回目から取ってくる
#last_dataが5020であれば5020回目まで取ってくる
first_number = 280
last_number = 5280

#一回辺りに取ってくる量
amount = 19
#保存先のディレクトリ
output = open('numbers.txt', mode='w')

while first_number < last_number:
    #スクレイピング先のurl
    url = 'https://takarakuji.rakuten.co.jp/backnumber/numbers3_detail/{}-{}/'.format('{0:04d}'.format(first_number), '{0:04d}'.format(first_number + amount) )
    soup = BeautifulSoup(requests.get(url).content,'html.parser')

    tag_tr = soup.find_all('tr')
    #テーブルの各データの取得
    data = []
    for i in range(1,len(tag_tr)):
        data.append([d.text for d in tag_tr[i].find_all('td')])
        cnt = 0;
        for j in data[i-1]:
            #当選番号のみを持ってくる
            if( cnt <= 1):
                cnt += 1
                continue
            output.write(j)
            output.write('\n')
            cnt += 1
    
    first_number += amount
    #アクセスを集中しないための処置
    time.sleep(1)
