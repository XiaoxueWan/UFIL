# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 20:43:06 2022

@author: Lenovo
"""
import torch.nn.functional as F

from torch import nn
from .shapeletsDistBlocks import ShapeletsDistBlocks

class LearningShapeletsModel(nn.Module):
    """
    Implements Learning Shapelets. Just puts together the ShapeletsDistBlocks with a
    linear layer on top.
    ----------
    shapelets_size_and_len : dict(int:int)
        keys are the length of the shapelets for a block and the values the number of shapelets for the block
    in_channels : int
        the number of input channels of the dataset
    num_classes: int
        the number of classes for classification
    dist_measure: 'string'
        the distance measure, either of 'euclidean', 'cross-correlation', or 'cosine'
    to_cuda : bool
        if true loads everything to the GPU
    """
    def __init__(self, shapelets_size_and_len, in_channels=1, num_classes=2, dist_measure='euclidean',ucr_dataset_name='comman',
                 to_cuda=True):
        super(LearningShapeletsModel,self).__init__()

        self.to_cuda = to_cuda
        self.shapelets_size_and_len = shapelets_size_and_len
        self.num_shapelets = sum(shapelets_size_and_len.values())
        self.shapelets_blocks = ShapeletsDistBlocks(in_channels=in_channels,
                                                    shapelets_size_and_len=shapelets_size_and_len,
                                                    dist_measure=dist_measure,ucr_dataset_name=ucr_dataset_name, to_cuda=to_cuda)
        
        self.num_classes = num_classes
        if self.to_cuda:
            self.cuda()

    def forward(self, x_t,m0='transform'):
        """
        Calculate the distances of each time series to the shapelets and stack a linear layer on top.
        @param x: the time series data
        @type x: tensor(float) of shape (n_samples, in_channels, len_ts)
        @return: the logits for the class predictions of the model
        @rtype: tensor(float) of shape (num_samples, num_classes)
        """
        m0 = self.shapelets_blocks(x_t,m0)
        #m0 = F.normalize(m0,dim=-1)
        clone_m0 = m0.clone()
        return m0,clone_m0
    
    def transform(self, X):
        """
        Performs the shapelet transform with the input time series data x
        @param X: the time series data
        @type X: tensor(float) of shape (n_samples, in_channels, len_ts)
        @return: the shapelet transform of x
        @rtype: tensor(float) of shape (num_samples, num_shapelets)
        """
        previous_distance='transform'
        return self.shapelets_blocks(X,previous_distance)

    def get_shapelets(self):
        """
        Return a matrix of all shapelets. The shapelets are ordered (ascending) according to
        the shapelet lengths and padded with NaN.
        @return: a tensor of all shapelets
        @rtype: tensor(float) with shape (in_channels, num_total_shapelets, shapelets_size_max)
        """
        return self.shapelets_blocks.get_shapelets()

    def set_shapelet_weights(self, weights):
        """
        Set the weights of all shapelets. The shapelet weights are expected to be ordered ascending according to the
        length of the shapelets. The values in the matrix for shapelets of smaller length than the maximum
        length are just ignored.
        @param weights: the weights to set for the shapelets
        @type weights: array-like(float) of shape (in_channels, num_total_shapelets, shapelets_size_max)
        @return:
        @rtype: None
        """
        start = 0
        for i, (shapelets_size, num_shapelets) in enumerate(self.shapelets_size_and_len.items()):
            end = start + num_shapelets
            self.set_shapelet_weights_of_block(i, weights[start:end, :, :shapelets_size])
            start = end

    def set_shapelet_weights_of_block(self, i, weights):
        """
        Set the weights of shapelet block i.
        @param i: The index of the shapelet block
        @type i: int
        @param weights: the weights for the shapelets of block i
        @type weights: array-like(float) of shape (in_channels, num_shapelets, shapelets_size)
        @return:
        @rtype: None
        """
        self.shapelets_blocks.set_shapelet_weights_of_block(i, weights)

    def set_weights_of_shapelet(self, i, j, weights):
        """
        Set the weights of shapelet j in shapelet block i.
        @param i: the index of the shapelet block
        @type i: int
        @param j: the index of the shapelet in shapelet block i
        @type j: int
        @param weights: the weights for the shapelet
        @type weights: array-like(float) of shape (in_channels, shapelets_size)
        @return:
        @rtype: None
        """
        self.shapelets_blocks.set_shapelet_weights_of_single_shapelet(i, j, weights)