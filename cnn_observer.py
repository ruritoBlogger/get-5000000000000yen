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

        for i in range(len(data)-vertical_len):
            
            tmp_teach_data = data[i: i+vertical_len]
            tmp_ans_data = data[i+vertical_len+1]

            tmp_teach_data = self.transform_picture(tmp_teach_data)
            tmp_ans_data = self.transform_picture(tmp_ans_data)

            teach.append(tmp_teach_data)
            ans.append(tmp_ans_data)

        return teach, ans

    def transform_picture(self, data):
        picture_data = []
        for i in range(len(data)):
            tmp_data = [0] * 10
            
            for j in range(len(str(data[i]))):
                key = int(str(data[i])[j])
                tmp_data[key] += 1

            picture_data.append(tmp_data)

        return picture_data
