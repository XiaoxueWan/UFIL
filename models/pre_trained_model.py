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
import gc
import matplotlib.pyplot as plt
gc.collect()

from torch import tensor
from torch.utils.data import DataLoader, TensorDataset
from utils import normalize_data
from tqdm import tqdm
from utils import get_weights_via_kmeans,plot_shapelets,plot_sub_fig
from .shapelet_network_pre_trained.shapelet_network import LearningShapeletsModel
#from memory_profiler import profile

#@profile(precision=4,stream=open('memory_profiler.log','w+'))
class LearningShapeletsPretrain:
    def __init__(self, shapelets_size_and_len, in_channels=1, num_classes=2,
                 dist_measure='euclidean',ucr_dataset_name='comman',temp_factor=2, online=0, scalar=0, verbose=0, to_cuda=False):

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
        
        '''test'''
        self.online = online
        self.scalar = scalar

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

    def update(self, x, y):
        """
        """
        D0,_ = self.model(x)
        #print(D0.shape,y.shape,'***********D0,y')
        Prototypes_current_task = self.get_prototype(D0,y)
        loss = self.prototype_class_loss(D0, y, Prototypes_current_task)
        Prototypes = Prototypes_current_task
        #loss.backward(retain_graph=True)
#        for name, parms in self.model.named_parameters():
#	        print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', parms, ' -->grad_value:', parms.grad)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item(),Prototypes

    def fit(self, X, Y, epochs=1, batch_size=256, shuffle=False, drop_last=False):
        """
        """
        if self.optimizer is None:
            raise ValueError("No optimizer set. Please initialize an optimizer via set_optimizer(optim)")

        # convert to pytorch tensors and data set / loader for training
        if not isinstance(X, torch.Tensor):
            X = tensor(X, dtype=torch.float).contiguous()
        if not isinstance(Y, torch.Tensor):
            Y = tensor(Y, dtype=torch.long).contiguous()
        if self.to_cuda:
            X = X.cuda()
            Y = Y.cuda()
        if Y.min()<0:
            Y=torch.sub(Y,Y.min())
        #print(Y,'y_label*************')
        train_ds = TensorDataset(X, Y)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

        # set model in train mode
        self.model.train()

        losses_ce = []
        progress_bar = tqdm(range(epochs), disable=False if self.verbose > 0 else True)
        current_loss_ce = 0
        for _ in progress_bar:
            for j, (x, y) in enumerate(train_dl):
                #print(j,'epoch')
                # check if training should be done with regularizer
                current_loss_ce, Prototypes = self.update(x, y)
                losses_ce.append(current_loss_ce)
                progress_bar.set_description(f"Loss: {current_loss_ce}")
                if _==0:
                    Prototypes0 = Prototypes
                elif _==399:
                   # print(Prototypes,self.W0)
                   pass
        return losses_ce,Prototypes,Prototypes0

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
               # print(Pre_D.shape,'Pre_D.shape')
                y_hat,Prototypes = self.predict_by_prototypes(Pre_D, Prototypes)  
                #print(y_hat, Prototypes,'y_hat, prototypes')
                y_hat = y_hat.cpu().detach().numpy()
                result = y_hat if result is None else np.concatenate((result, y_hat), axis=0)
        return result
    
    def caculated_probability(self, Prototypes, D):
        '''
        计算每个样本对应不同类别原型的概率值
        
        参数：
        --------
        Prototypes: {0:tensor([]),1:tensor([])}
                  不同类别对应的原型
        D: [num of samples, num of shapelets]
           tensor([[[ ]]])
        
        返回值：
        --------
        distance_proto_D:[num of shapelets]
        normalize_distance_proto_D:[num of shapelets]
                                   将每个distance_proto_D值除以distance_proto_D的和
        '''
        distance_proto_D = torch.tensor([])
        for classes,Prototype in  Prototypes.items():
            cos = torch.cosine_similarity(D,Prototype,dim=-1)
            distance_proto_D = torch.cat((distance_proto_D,cos),dim=1)
        normalize_distance_proto_D = torch.div(distance_proto_D,torch.sum(distance_proto_D))
        return distance_proto_D, normalize_distance_proto_D
    
    def calculated_entropy_ymax(self, Prototypes, D):
        '''
        计算分数
        
        参数
        -------------
        Prototypes: {0:tensor([]),1:tensor([])}
                  不同类别对应的原型
        D: [num of samples, num of shapelets]
           tensor([[[ ]]])
           
        返回值
        ---------------
        ENTROPY：数值
               信息熵
        y_max: [数值]
             最大概率值
        y_hat: [数值]
            预测的标签
        '''
        #计算概率值
        distance_proto_D, normalize_distance_proto_D = self.caculated_probability(Prototypes,D)
        y_hat = torch.argmax(distance_proto_D,dim=1) #计算真实的y值
        
        '''计算最大的y值'''
        y_max,_ = torch.max(normalize_distance_proto_D, dim=1)  
        
        '''计算信息熵'''
        def entropy(p):
            return -torch.sum(p*torch.log10(p))
        ENTROPY = entropy(normalize_distance_proto_D)
        return ENTROPY.detach().numpy(),y_max.detach().numpy(),y_hat.numpy()
    
    def online_monitor(self, Prototypes):
        '''
        在线监测，用训练好的模型测试在线数据的分类精度。
        
        返回值
        ---------------
        buffer_X: Tensor([num of new classes, 1, window_size])
                  new class samples
        buffer_Y: Tensor(num of new classes)
                  new class labels
        '''
        
        last_distance0 = torch.full((1,1,self.shapelet_num),100000.0)
        last_distance1 = torch.full((1,1,self.shapelet_num),100000.0)
        X, ENTROPY_list, result_list, Y_max_list, Score_list = [],[],[],[],[] #记录数据的列表
        buffer_X, buffer_Y = torch.Tensor([]), torch.Tensor([])
        num = 0
        for t in range(self.shapelet_length,4000+1,1):    #代表遍历的数据点
            if num>20: #表示每多少个点开始重新计算shapelet与时间序列的相似度
                num=0
                last_distance0=last_distance1
                
            '''截取数据，并对数据进行z-normalize规范化，规范的指标是训练集的方差和均值'''
            X_t = self.online[:,:,t-self.shapelet_length:t]
            X_t,_ = normalize_data(X_t,self.scalar)
            X_t = torch.from_numpy(X_t).float()
            
            '''判断一个数据是否是新的数据'''
            D,last_distance0 = self.model(X_t,last_distance0)
            ENTROPY,y_max,y_hat = self.calculated_entropy_ymax(Prototypes,D)
            if abs(ENTROPY/len(Prototypes)-y_max)>0.08 and num == 10:
                #print(t,ENTROPY-y_max,'%%%%%%%%%%%%%')
                buffer_X = torch.cat((buffer_X,X_t),dim=0)
                buffer_Y = torch.cat((buffer_Y,torch.Tensor(len(Prototypes)+1)),dim=0)
            '''对数据进行记录'''
            X.append(t)
            result_list.append(y_hat)
            ENTROPY_list.append(ENTROPY)
            Y_max_list.append(y_max)
            Score_list.append(ENTROPY-y_max)
            num+=1
        #print(result_list)
        plot_sub_fig(X,result_list,'pretrain_result')
        plot_sub_fig(X,Y_max_list,'pretrain_Y_max')
        plot_sub_fig(X,Score_list,'pretrain_Score')
        plot_sub_fig(X,ENTROPY_list,'pretrain_ENTROPY')
        return buffer_X, buffer_Y

    def get_shapelets(self):
        """
        """
        return self.model.get_shapelets().clone().cpu().detach().numpy()

    def get_weights_linear_layer(self):
        """
        """
        return (self.model.linear.weight.data.clone().cpu().detach().numpy(),
                self.model.linear.bias.data.clone().cpu().detach().numpy())