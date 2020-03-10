class CNN_observer():

    def __init__(self, vertical_len):
        self.vertical_len = vertical_len

    def get_data(self, file_name):
        data = []
        with open(file_name) as f:
            line = f.readline()
            while(line):
                number = int(line.replace("\n",""))
                data.append(number)
                line = f.readline()

        return data

    def transform(self, data):
        teach = []
        ans = []

        for i in range(len(data)-self.vertical_len-1):
            
            tmp_teach_data = data[i: i+self.vertical_len]
            tmp_ans_data = data[i+self.vertical_len+1]
            tmp_teach_data = self.transform_picture(tmp_teach_data)
            tmp_ans_data = self.transform_picture(tmp_ans_data)

            teach.append(tmp_teach_data)
            ans.append(tmp_ans_data)

        return teach, ans

    def transform_picture(self, data):
        picture_data = []

        if( type(data) == int ):
            #tmp_data = [0] * 10
            tmp_data = []
            if( len(str(data)) != 3 ):
                for i in range( 3 - len(str(data)) ):
                    tmp_data.append(0)
            
            for i in range(len(str(data))):
                key = int(str(data)[i])
                #tmp_data[key] += 1
                tmp_data.append(key)
            picture_data = tmp_data
        
        else:
            for i in range(len(data)):
                tmp_data = [0] * 10
                
                for j in range(len(str(data[i]))):
                    key = int(str(data[i])[j])
                    tmp_data[key] += 1

                picture_data.append(tmp_data)

        return picture_data
