# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 20:41:35 2022

@author: Lenovo
"""
import sys 
sys.path.append("..") 

import torch
import warnings
import numpy as np
from torch import tensor
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from .shapelet_network_pre_trained.shapelet_network import LearningShapeletsModel

class LearningShapeletsIncremental:
    def __init__(self, 
                     shapelets_size_and_len, 
                     in_channels=1, 
                     num_classes=2,
                     dist_measure='euclidean',
                     ucr_dataset_name='comman',
                     temp_factor=2, 
                     online=0, 
                     scalar=0, 
                     verbose=0, 
                     to_cuda=False
                     ):

        self.model = LearningShapeletsModel(
                                            shapelets_size_and_len = shapelets_size_and_len,
                                            in_channels = 1, 
                                            num_classes = num_classes, 
                                            dist_measure = dist_measure,
                                            ucr_dataset_name = ucr_dataset_name,
                                            to_cuda = to_cuda
                                            )
        self.to_cuda = to_cuda
        if self.to_cuda:
            self.model.cuda()

        self.shapelets_size_and_len = shapelets_size_and_len
        self.shapelet_length = sum(self.shapelets_size_and_len.keys())
        self.shapelet_num = sum(self.shapelets_size_and_len.values())
        self.verbose = verbose
        self.optimizer = None
        self.temp_factor = temp_factor
        self.scalar = scalar  #归一化的尺度标准

    def set_optimizer(self, optimizer):
        """
        """
        self.optimizer = optimizer

    def set_shapelet_weights(self, weights):
        """
        """
        self.model.set_shapelet_weights(weights)
        if self.optimizer is not None:
            warnings.warn("Updating the model parameters requires to reinitialize the optimizer. Please reinitialize"
                          " the optimizer via set_optimizer(optim)")

    def set_shapelet_weights_of_block(self, i, weights):
        """
        """
        self.model.set_shapelet_weights_of_block(i, weights)
        if self.optimizer is not None:
            warnings.warn("Updating the model parameters requires to reinitialize the optimizer. Please reinitialize"
                          " the optimizer via set_optimizer(optim)")
            
    def prototype_class_loss(self, D, y, Prototypes):
        '''
           定义一个原型的损失函数,与交叉熵损失不同在于没有分类器参数
           Prototypes:
                 代表所有类别的原型集,假设总共是k个类别
                 --{1:[1Xdim],2:[1Xdim],...,k:[1Xdim]} 
           similar_join: 
                 代表所有原型与每个样本对应特征集的距离. similar_join = exp(余弦相似度(D,p)/温度系数)
                 --NXk
           similar_sum: 
                 代表所有原型与每个样本对应特征集的距离. similar_join = 求和（exp(余弦相似度(D,p)/温度系数)）
                 --NX1
        '''
        similar_join = torch.Tensor()
        for classes,Prototype in Prototypes.items():
            D = D.to(torch.float32)
            cos = torch.exp(torch.cosine_similarity(D,Prototype,dim=-1)/self.temp_factor) #余弦相似度的计算公式
            similar_join = torch.cat([similar_join,cos],1)  #将每个原型对应的余弦相似度值进行拼接
        y = y.unsqueeze(1)
        similar = torch.gather(similar_join, dim=1, index=y)      #取计算损失函数的公式的分母，用y对余弦相似度做映射，因此可以取对应类别标签对应原型的余弦相似度
        
        similar = torch.squeeze(similar)
        similar_sum = torch.sum(similar_join,dim=1)
        similar_div = torch.div(similar,similar_sum)
        return -torch.mean(torch.log10(similar_div))
    
    def get_prototype(self, D, y):
        '''
           新来一个任务的数据，首选初始化计算当前任务对应的特征原型.
           特征矩阵 tensor D: [num_samples, 1, num_shapelets] 变为 D: [num_samples, num_shapelets]
           X对应的标签 tensor y: [1,2,3]
           原型集 tensor Prototypes:{1:[],2:[],3:[]} R:原型数*num_shapelets
        '''
       # D = self.C(D)
        D = torch.squeeze(D)
        y = y.unsqueeze(1)
        y = y.float()
        D_y = torch.cat((D,y),1) #将D拼上y 
        y = torch.squeeze(y)
        y = y.numpy()
        Prototypes_current_task = dict.fromkeys(list(set(y)))  #新建一个空的字典，以y的值作为键
        for i in list(set(y)):
           mask = torch.where(D_y[:,-1]==i)
           D_y = D_y.float()
           D_y_class = torch.mean(D_y[mask][:,:-1],0)
           if Prototypes_current_task[i] == None:
               Prototypes_current_task[i] = D_y_class
        #print(Prototypes_current_task,'&&&&&&&&&&&&&&Prototypes_current_task')
        return Prototypes_current_task

    def update(self, x, y, Prototypes):
        """
        """
        D0,_ = self.model(x)
        #print(D0.shape,y.shape,'***********D0,y')
        Prototypes_current_task = self.get_prototype(D0,y)
        Prototypes.update(Prototypes_current_task)
        return Prototypes

    def fit(self, X, Y, Prototypes, epochs=1, batch_size=256, shuffle=False, drop_last=False):
        
        train_ds = TensorDataset(X, Y)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

        # set model in train mode
        self.model.train()

        progress_bar = tqdm(range(epochs), disable=False if self.verbose > 0 else True)
        current_loss_ce = 0
        for _ in progress_bar:
            for j, (x, y) in enumerate(train_dl):
                # check if training should be done with regularizer
                Prototypes = self.update(x, y, Prototypes)
                progress_bar.set_description(f"Loss: {current_loss_ce}")
                if _==0:
                    Prototypes0 = Prototypes
                elif _==399:
                   # print(Prototypes,self.W0)
                   pass
        return Prototypes

    def transform(self, X):
        """
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float)
        if self.to_cuda:
            X = X.cuda()

        with torch.no_grad():
            shapelet_transform = self.model.transform(X)
        return shapelet_transform.squeeze().cpu().detach().numpy()

    def fit_transform(self, X, Y, epochs=1, batch_size=256, shuffle=False, drop_last=False):
        """
        """
        self.fit(X, Y, epochs=epochs, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        return self.transform(X)
    
    def predict_by_prototypes(self, Pre_D, Prototypes):
        distance_proto_D = torch.tensor([])
        for classes,Prototype in  Prototypes.items():
            cos = torch.cosine_similarity(Pre_D,Prototype,dim=-1)
            cos = cos.unsqueeze(1)
            
            distance_proto_D = torch.cat((distance_proto_D,cos),dim=1)
        return torch.argmax(distance_proto_D,dim=1),Prototypes

    def predict(self, X,  Prototypes=None, batch_size=256):
        """
        预测离线数据集的准确率
        """
        X = tensor(X, dtype=torch.float32)
        if self.to_cuda:
            X = X.cuda()
        ds = TensorDataset(X)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

        # set model in eval mode
        self.model.eval()

        """Evaluate the given data loader on the model and return predictions"""
        result = None
        with torch.no_grad():
            for x in dl:
                Pre_D,_ = self.model(x[0])
                y_hat,Prototypes = self.predict_by_prototypes(Pre_D, Prototypes)  
                #print(y_hat, Prototypes,'y_hat, prototypes')
                y_hat = y_hat.cpu().detach().numpy()
                result = y_hat if result is None else np.concatenate((result, y_hat), axis=0)
        return result

    def get_shapelets(self):
        """
        """
        return self.model.get_shapelets().clone().cpu().detach().numpy()

    def get_weights_linear_layer(self):
        """
        """
        return (self.model.linear.weight.data.clone().cpu().detach().numpy(),
                self.model.linear.bias.data.clone().cpu().detach().numpy())