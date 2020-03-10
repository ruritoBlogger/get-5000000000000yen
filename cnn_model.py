import matplotlib.pyplot as plt
import matplotlib.cm as cm
from chainer import Chain
import chainer.functions as F
import chainer.links as L
import numpy as np


class CNN(Chain):

    def __init__(self):
        super(CNN,self).__init__(
            conv0 = L.Convolution2D(),
            conv1 = L.Convolution2D(),
            conv2 = L.Convolution2D(),
            conv3 = L.Convolution2D(),
            affine1 = L.Linear(),
            affine2 = L.Linear()
        )

    def forward(self, x):
        h0 = F.max_pooling_2d(F.relu(self.conv0(x)), 2)
        h1 = F.max_pooling_2d(F.relu(self.conv0(h0)), 2)
        h2 = F.max_pooling_2d(F.relu(self.conv0(h1)), 2)
        h3 = F.max_pooling_2d(F.relu(self.conv0(h2)), 2)

        h4 = self.affine1(h3)
        h5 = self.affine2(h4)

        return F.softmax(h5)

    def debug_forward(self, x):
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

