import matplotlib.pyplot as plt
import matplotlib.cm as cm
from chainer import Chain
import chainer.functions as F
import chainer.links as L
import numpy as np

class CNN(Chain):

    def __init__(self):
        super(CNN,self).__init__(
            conv0 = L.Convolution2D(1, 2, (1,2), pad=1),
            conv1 = L.Convolution2D(2, 4, (1,2), pad=1),
            conv2 = L.Convolution2D(4, 8, (1,2), pad=1),
            conv3 = L.Convolution2D(8, 16, (1,2), pad=1),
            affine1 = L.Linear(None, 1000),
            affine2 = L.Linear(None, 10)
        )

    def forward(self, x):
        x = np.array(x).astype(np.float32)
        x = x.reshape([int(x.size/x[0].size), 1, int(x[0].size/10), 10])
        h0 = F.max_pooling_2d(F.relu(self.conv0(x)), 2)
        h1 = F.max_pooling_2d(F.relu(self.conv1(h0)), 2)
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), 2)
        h3 = F.max_pooling_2d(F.relu(self.conv3(h2)), 2)

        h4 = self.affine1(h3)
        h5 = self.affine2(h4)

        return F.softmax(h5)

    def debug_forward(self, x):
        x = np.array(x).astype(np.float32)
        x = x.reshape([int(x.size/x[0].size), 1, int(x[0].size/10), 10])
        self.make_picture("before.png",x)
        
        h0 = F.max_pooling_2d(F.relu(self.conv0(x)), 2)
        self.make_picture("h0.png",h0)
        
        h1 = F.max_pooling_2d(F.relu(self.conv0(h0)), 2)
        self.make_picture("h1.png",h1)

        h2 = F.max_pooling_2d(F.relu(self.conv0(h1)), 2)
        self.make_picture("h2.png",h2)
        
        h3 = F.max_pooling_2d(F.relu(self.conv0(h2)), 2)
        self.make_picture("h3.png",h3)

        h4 = self.affine1(h3)
        h5 = self.affine2(h4)

        return F.softmax(h5)
    
    def make_picture(self, file_name, data):
        plt.figure()
        if( type(data) == Variable ):
            data.data = data.data.astype(np.float32)
            plt.matshow(data.data[:, :], cmap='viridis')
        else:
            plt.matshow(data[:, :], cmap='viridis')
        plt.colorbar()
        #plt.show()
        plt.savefig(file_name) 

