import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class dnn(nn.Module):
    def __init__(self, im_size, n_classes,depth=1, act="sig"):
        '''
        Create components of a softmax classifier and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            n_classes (int): Number of classes to score
        '''
        super(dnn, self).__init__()
        
        channels, height, width = im_size
        self.l1 = nn.Linear(channels * height *width, n_classes)
        self.l2 = nn.Linear(n_classes, n_classes)
        self.fnn = []
        for i in range(depth):
            layer = nn.Linear(n_classes,n_classes)
            self.fnn.append(layer)
        if act == "sig":
            self.act = nn.Sigmoid()
        else:
            self.act = nn.LeakyReLU()

        

        

    def forward(self, images):
        '''
        Take a batch of images and run them through the classifier to
        produce a score for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        '''
        scores = None
        images = images.permute(0,2,3,1)
        x = torch.flatten(images, start_dim = 1)
        x = self.l1(x)
        x = self.act(x)
        x = self.fnn[0](x)
        # for layer in self.fnn:
        #     x = self.act(x)
        #     x = layer(x)
        scores = F.softmax(x,1)
        return scores

