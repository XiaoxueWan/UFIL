# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 16:05:13 2024

@author: Lenovo
"""
import torch

def get_prototype(D, y):
    '''
       新来一个任务的数据，首选初始化计算当前任务对应的特征原型.
       特征矩阵 tensor D: [num_samples, 1, num_shapelets] 变为 D: [num_samples, num_shapelets]
       X对应的标签 tensor y: [1,2,3]
       原型集 tensor Prototypes:{1:[],2:[],3:[]} R:原型数*num_shapelets
    '''
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

def prototype_class_loss(D, y, Prototypes):
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
        #print(D.shape,Prototype.shape,'*******D.shape,Prototype.shape')
        cos = torch.exp(torch.cosine_similarity(D,Prototype,dim=-1)/1) #余弦相似度的计算公式
        print(cos)
        '''欧式距离的计算公式'''
        #Prototype = Prototype.unsqueeze(0)
        #cos0 = torch.exp(-torch.cdist(D,Prototype))
        #cos1 = torch.squeeze(cos0)
        #cos = cos1.unsqueeze(1)
        
        similar_join = torch.cat([similar_join,cos],1)  #将每个原型对应的余弦相似度值进行拼接
    y = y.unsqueeze(1)
    similar = torch.gather(similar_join, dim=1, index=y)      #取计算损失函数的公式的分母，用y对余弦相似度做映射，因此可以取对应类别标签对应原型的余弦相似度
    
    similar = torch.squeeze(similar)
    similar_sum = torch.sum(similar_join,dim=1)
    similar_div = torch.div(similar,similar_sum)
    return -torch.mean(torch.log(similar_div))