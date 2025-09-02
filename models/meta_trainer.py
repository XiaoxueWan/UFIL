# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 20:41:35 2022

@author: Lenovo
"""
import sys 
sys.path.append("..") 

import random
import torch
import copy
import warnings
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from torch import tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from .shapelet_network.shapelet_network import LearningShapeletsModel
from .transformer.net import MultiHeadAttention,MultiHeadAttentionNewClass

class FakeTrainer:
    def __init__(self, 
                     shapelets_size_and_len, 
                     in_channels=1, 
                     num_classes=2,
                     dist_measure='euclidean',
                     ucr_dataset_name='comman',
                     epoch = 5,
                     temp_factor=2, 
                     verbose=0, 
                     to_cuda=False,
                     alpha_new_class=20
                     ):

        self.model = LearningShapeletsModel(
                                            shapelets_size_and_len = shapelets_size_and_len,
                                            in_channels = 1, 
                                            num_classes = num_classes, 
                                            dist_measure = dist_measure,
                                            ucr_dataset_name = ucr_dataset_name,
                                            to_cuda = to_cuda
                                            )
        
        self.slf_attn = MultiHeadAttention(
                                           1, 
                                           list(shapelets_size_and_len.values())[0], 
                                           list(shapelets_size_and_len.values())[0], 
                                           list(shapelets_size_and_len.values())[0], 
                                           dropout=0.3
                                           )
        
        self.slf_attn_classify = MultiHeadAttentionNewClass()
        
        self.to_cuda = to_cuda
        if self.to_cuda:
            self.model.cuda()
            
        self.epoch = epoch
        self.shapelets_size_and_len = shapelets_size_and_len
        self.shapelet_length = sum(self.shapelets_size_and_len.keys())
        self.shapelet_num = sum(self.shapelets_size_and_len.values())
        self.verbose = verbose
        self.optimizer = None
        self.temp_factor = temp_factor
        self.loss_new_class = nn.BCELoss()
        self.alpha_new_class = alpha_new_class

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
    
    def generate_support_query(self, base_dataset):
        '''
           随机生成 支持集support 和 查询集query, 根据论文《Few-Shot Class-Incremental Learning by Sampling Multi-Phase Tasks》中的Algorithm 1的伪代码
           input:
                base_dataset: {'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test}
           output:
               3,1 为需要学习的新类标签
               support_set:{
                            3:{'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test},  #class:0
                            1:{'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test}   #class:1
                            } 
               query_set:{
                          3:{'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test}, #class:0
                          1:{'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test}  #class:0,1
                          }
               base_class_number_dict: 3, 表示基础任务的任务id
               
        '''
        train = np.concatenate((base_dataset['X_train'].reshape(base_dataset['X_train'].shape[0],base_dataset['X_train'].shape[-1]),
                                base_dataset['y_train'].reshape(base_dataset['y_train'].shape[0],1)), axis=1)
        test = np.concatenate((base_dataset['X_test'].reshape(base_dataset['X_test'].shape[0],base_dataset['X_test'].shape[-1])
                               ,base_dataset['y_test'].reshape(base_dataset['y_test'].shape[0],1)), axis=1)
        
        class_list = list(set(np.reshape(base_dataset['y_test'],(len(base_dataset['y_test'])))))
        class_list = [int(x) for x in class_list]    #获取所有的类别标签
        
        base_class_list= random.sample(class_list,3)   #随机从类别标签中选择3个数组成一个base_class_list torch.Tensor([1,2,3])
        
        keys = [a for a in class_list if a not in base_class_list]  #除去基础类别的所有类别，构建keys，用于组成新类
        support_x_dict =  dict.fromkeys(keys,{})    #新建空的支持集和查询集，任务数量由除了任务0以外的其他任务组成
        support_y_dict =  dict.fromkeys(keys,{})
        query_x_dict =  dict.fromkeys(keys,{})
        query_y_dict =  dict.fromkeys(keys,{})
        
        query_class_list = copy.deepcopy(base_class_list)
        for i in class_list:
            if i not in base_class_list:
                #print(i,'***********i')
                #支持集
                mask = np.isin(train[:, -1],[i])
                X_train = train[mask][:, :-1]
                y_train = train[mask][:, -1]
                support_x_dict[i] = X_train.reshape(X_train.shape[0],1,X_train.shape[-1])
                support_y_dict[i] = y_train
                
                #查询集
                X_test,y_test = np.empty((1,base_dataset['X_test'].shape[-1])),np.empty(0) #构建查询集
                query_class_list += [i]
               # print(query_class_list,'*********query_class_list')
                for j in query_class_list:
                    if j == i: #假如是新类,取4个样本来维持样本平衡
                        mask = np.isin(test[:, -1],[j])   #j代表新类
                        X_test = np.concatenate((X_test, test[mask][:4, :-1]), axis=0)  #从基础类别中每个类别选择3个样本,构成查询集
                        y_test = np.concatenate((y_test, test[mask][:4, -1]), axis=0)
                    else:
                        mask = np.isin(test[:, -1],[j])   #j代表新类
                        X_test = np.concatenate((X_test, test[mask][:1, :-1]), axis=0)  #从基础类别中每个类别选择3个样本,构成查询集
                        y_test = np.concatenate((y_test, test[mask][:1, -1]), axis=0)
                X_test = X_test[1:,:]
                query_x_dict[i] = X_test.reshape(X_test.shape[0],1,X_test.shape[-1])
                query_y_dict[i] = y_test
        return support_x_dict,support_y_dict,query_x_dict,query_y_dict,base_class_list
    
    def get_prototype0(self, D, y, label):
        '''
           根据D，获取y中标签为label的类原型
        '''
        D = torch.squeeze(D)
        y = y.unsqueeze(1)
        y = y.float()
        D_y = torch.cat((D,y),1) #将D拼上y 
        y = torch.squeeze(y)
        y = y.numpy()
        
        mask = torch.where(D_y[:,-1]==label)
        D_y = D_y.float()
        D_y_class = torch.mean(D_y[mask][:,:-1],0)
        D_y_class = D_y_class.unsqueeze(0)
        return D_y_class
            
    def prototype_class_loss(self, D, y, Prototypes, temperature_parameter):
        '''
           定义一个原型的损失函数,与交叉熵损失不同在于没有分类器参数
            Input
           ---------------
           D:  R:[num_samples,1,dim]
              查询集的特征映射
           y:  tensor R:[num_samples]
               查询集的标签y
           Prototypes:
                 代表所有类别的原型集,假设总共是k个类别
                 --{1:[1Xdim],2:[1Xdim],...,k:[1Xdim]}  R:[num_protos,dim]
           similar_join: 
                 代表所有原型与每个样本对应特征集的距离. similar_join = exp(余弦相似度(D,p)/温度系数)
                 --NXk
           similar_sum: 
                 代表所有原型与每个样本对应特征集的距离. similar_join = 求和（exp(余弦相似度(D,p)/温度系数)）
                 --NX1
           Output
           ---------------
           loss
        '''
        #print(D.shape, y.shape,'Dy&&&&&&&&&&')
        unrecognized_class = [key for key in list(set(y.numpy())) if key not in list(Prototypes.keys())]#找到未能识别成功的新类
        if len(unrecognized_class)!=0:
            D_y_class = self.get_prototype0(D, y, unrecognized_class[0])
            Prototypes[unrecognized_class[0]] = D_y_class
        
        similar_join = torch.Tensor()
        for classes,Prototype in Prototypes.items():
            D = D.to(torch.float32)
            cos = torch.exp(torch.cosine_similarity(D,Prototype,dim=-1)/self.temp_factor) #余弦相似度的计算公式
            similar_join = torch.cat([similar_join,cos],1)  #将每个原型对应的余弦相似度值进行拼接
           
        '''由于prototypes的keys如 0,1,3,4, 并不能直接对similar 做标签映射，因此需要把0,1,3,4映射到0,1,2,3'''
        y = y.numpy()
        Prototypes_keys = dict.fromkeys(Prototypes.keys()) 
        num = 0
        map_dict = {}
        for i in Prototypes_keys:
            map_dict[i] = num
            num+=1
     
        y = np.array(list(map(map_dict.get,y)))
        y = torch.from_numpy(y)
        y = y.unsqueeze(1)
        similar = torch.gather(similar_join, dim=1, index=y.long())      #取计算损失函数的公式的分子，用y对余弦相似度做映射，因此可以取对应类别标签对应原型的余弦相似度
     
        similar = torch.squeeze(similar)
        similar_sum = torch.sum(similar_join,dim=1)
        similar_div = torch.div(similar,similar_sum)
        return -torch.mean(torch.log10(similar_div))
    
    def change_D_Proto(self, D, Prototypes):
        '''
            利用transformer中的自注意力机制，变换D和原型Prototypes,从而对齐
            D: [num_samples, num_shapelets]  shapelet 映射特征
            Prototypes：{1:torch.tensor([]),2:torch.tensor([]),3:torch.tensor([])} 原型数*num_shapelets
        '''
        
        type_prototypes = type(Prototypes)
        if type_prototypes==dict:
            tensor_Prototype = torch.Tensor() #把字典变为tensor
            for i, val in Prototypes.items():
                #print(i,'i&&&&&&&&&&&&')
                tensor_Prototype = torch.cat([tensor_Prototype,val.unsqueeze(0)],0)
        else:
            tensor_Prototype = Prototypes
        
        tensor_Prototype = tensor_Prototype.unsqueeze(0).expand(D.shape[0],tensor_Prototype.shape[0],tensor_Prototype.shape[1]).contiguous()
        combined = torch.cat([tensor_Prototype, D],1)
        combined = self.slf_attn(combined, combined, combined)  #用transformer进行变换
        tensor_Prototype, D = combined.split(tensor_Prototype.shape[1], 1)
        tensor_Prototype = tensor_Prototype.narrow(0,0,1)
        tensor_Prototype = torch.squeeze(tensor_Prototype)
        
        if type_prototypes==dict:
            Prototypes = dict.fromkeys(Prototypes.keys())
            num=0
            for i, val in Prototypes.items():#'再把tensor_Prototype 变为字典Prototype'
                Prototypes[i] = tensor_Prototype[num]
                num+=1
        else:
            Prototypes = tensor_Prototype
        return D, Prototypes 
    
    def classfy_new_class(self, D, Prototypes):
        '''
            利用transformer中的自注意力机制，变换D和原型Prototypes,从而对齐
            D: [batch_size, num_shapelets]  shapelet 映射特征
            Prototypes：{1:torch.tensor([]),2:torch.tensor([]),3:torch.tensor([])} 原型数*num_shapelets
        '''
        D = D.squeeze(dim=1)
        cos_similarities = torch.Tensor()
        for classes,Prototype in Prototypes.items():
            Prototype0 = Prototype.unsqueeze(0).expand(D.shape[0],Prototype.shape[0]).contiguous()
            cos = nn.CosineSimilarity(dim=1)(D,Prototype0) #余弦相似度的计算公式
            cos_similarities = torch.cat([cos_similarities,cos.unsqueeze(1)],1)  #将每个原型对应的余弦相似度值进行拼接
        
        #print(Prototypes,'***********Prototypes')
      #  print(cos_similarities.shape,D.shape,Prototypes,'***********cos_similarities')
        output = self.slf_attn_classify(cos_similarities).squeeze(dim=1)  #用transformer进行变换
        return output
    
    def get_prototypes_from_predict_Dy(self, predict_class_list, before_or_pretrained_prototype_dict, D_y):
        #新的原型中，不在新数据集中出现的类别要么用基类原型补充，要么用上一次的原型补充。
        i = predict_class_list[0]
        
        mask = torch.where(D_y[:,-1]==i)
        D_y = D_y.float()
        D_y_class = torch.mean(D_y[mask][:,:-1],0)
        before_or_pretrained_prototype_dict[i] = D_y_class
        return before_or_pretrained_prototype_dict
    
    def get_prototype(self, D, y, base_class_prototypes_dict, fake_task_num, Prototypes):
        '''
           新来一个任务的数据，这个数据本来是新类的数据，但是可能被判断为新类，也有可能被判断为以前的类，用这个新类的数据来更新原型
           Input
           ---------------
                D: [num_samples, num_shapelets] 
                    tensor 特征矩阵 
               y: [1,2,3] 
                   tensor X对应的标签 
               base_class_prototype_dict: {0:torch.Tensor([2.1,2.4,3.12,2.34]),2:torch.Tensor([]),4:torch.Tensor([])}
                                        dict  基础类别的原型
                fake_task_num: 数值
                              处于假增量的第几个阶段
               Prototypes: {0：torch.Tensor([]),1:torch.Tensor([])}
                        字典 原型集, 关键字代表原型的类别
           Output
           ----------------
              Prototypes: 字典，更新之后的原型集
        '''
        #将数据先进行处理,将D拼接上y得到Dy
        D = torch.squeeze(D)
        y = y.unsqueeze(1)
        y = y.float()
        #print(y.shape,'*****y.shape')
        D_y = torch.cat((D,y),1) #将D拼上y 
        y = torch.squeeze(y)
        y = y.numpy()
        
        predict_class_list = list(set(y))  #预测的y的类别，包含新类
        
        #假如是第一个任务,就结合基础类别原型获取新的总的原型，假如是后面的迭代，就结合上一次的原型来获取新的总的原型
        if fake_task_num == 0:
            prototypes = self.get_prototypes_from_predict_Dy(predict_class_list,base_class_prototypes_dict,D_y)
        else:
            prototypes = self.get_prototypes_from_predict_Dy(predict_class_list,Prototypes, D_y)
        return prototypes
    
    def change_label_onehot(self, query_y, new_class):
        '''将查询集query_y中的class,不是新类的全部变为0，并且进行one_hot编码
           quer_y:tensor([])
           new_class:0
           one_hot:tensor([[1,0],[0,1]])
        '''
        condition = (query_y==new_class)
        label = torch.zeros_like(query_y)
        label[condition] = 1
        #print(label)
        one_hot = torch.zeros((len(query_y),2)).scatter_(1,label.long().reshape(-1,1),1)
        return one_hot,label.unsqueeze(-1).float()

    def meta_train(self, base_dataset, pretrain_prototypes_dict):
        '''元训练'''
        progress_bar = tqdm(range(self.epoch))
        loss_plot = []
        for i in progress_bar:
            torch.autograd.set_detect_anomaly(True)
            support_x_dict, support_y_dict, query_x_dict, query_y_dict, base_class_list = self.generate_support_query(base_dataset) #生成随机的支持集和查询集
           # print(support_x_dict,support_y_dict,query_x_dict,query_y_dict,'support_x_dict,support_y_dict,query_x_dict,query_y_dict')
            loss, fake_task_num, Prototypes = 0, 0, {}
            
            base_class_prototypes_dict = {k: pretrain_prototypes_dict[k] for k in pretrain_prototypes_dict.keys() if k in base_class_list}
            for key,value in support_x_dict.items():#遍历查询集和支持集中的每个样本，即遍历每个增量阶段
                support_x, support_y0, query_x, query_y = support_x_dict[key], support_y_dict[key], query_x_dict[key], query_y_dict[key]
               # print(support_x, support_y0, query_x, query_y,'***********')
                if not isinstance(support_x, torch.Tensor):
                    support_x = tensor(support_x, dtype=torch.float).contiguous()
                if not isinstance(query_x, torch.Tensor):
                    query_x = tensor(query_x, dtype=torch.float).contiguous()
                if not isinstance(support_y0, torch.Tensor):
                    support_y0 = tensor(support_y0, dtype=torch.long).contiguous()
                if not isinstance(query_y, torch.Tensor):
                    query_y = tensor(query_y, dtype=torch.long).contiguous()
                
                '''用支持集来更新原型'''
                D_support,_ = self.model(support_x)
                D_support = D_support.detach() 
                
              #  print(D_support, support_y0, Prototypes, '**********D_support, Prototypes')
                Prototypes = self.get_prototype(D_support, support_y0, base_class_prototypes_dict, fake_task_num, Prototypes)
                #print(Prototypes,'*********later Prototypes')
                
                '''用查询集来计算损失函数'''
                D_query,_ = self.model(query_x)
               
                '''获取新类模型的结果和预测值'''
                Prototypes_without_class = {k:v for k,v in Prototypes.items() if k!=support_y0[0]}
                
                D_new_class, Prototypes0 = self.change_D_Proto(D_query, Prototypes_without_class)
                outputs = self.classfy_new_class(D_new_class,Prototypes0).squeeze(dim=1)
                
                input_query = query_y
                one_hot_label,label = self.change_label_onehot(input_query,support_y0[0])
                
                #print(outputs,'outputs',label,'label')
                
                '''用transformer矫正一下'''
                D_query_changed, Prototypes_changed = self.change_D_Proto(D_query, Prototypes)
                loss = loss + self.prototype_class_loss(D_query_changed, query_y, Prototypes_changed, self.temp_factor) + self.alpha_new_class*self.loss_new_class(outputs,label)
                fake_task_num +=1
           
            loss_plot.append(loss)  #用于绘制损失图
            progress_bar.set_description(f"Loss: {loss}")
            torch.autograd.set_detect_anomaly(True)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return self.change_D_Proto, self.classfy_new_class, self.model, loss_plot, Prototypes #返回训练好的元校正机制
    
    def predict_by_prototypes(self, Pre_D, Prototypes):
        '''
        对每个shapelet嵌入之后的Pre_D,根据原型，获取它的预测的标签
        '''
        cos_dict = {}
       # print(Pre_D,'Pre_D**')
      #  print(Prototypes,'Prototypes***')
        Pre_D = torch.squeeze(Pre_D)
        for classes,Prototype in  Prototypes.items():
            cos = torch.cosine_similarity(Pre_D,Prototype,dim=-1)
            #print(cos,'cos************')
            cos = cos.unsqueeze(0)
            cos_dict[classes] = cos
        
       # print(cos_dict.values(),'cos_dict.values*********')
        max_cos = max(cos_dict.values())
        
        max_keys = [key for key, value in cos_dict.items() if value == max_cos]
        return max_keys[0]
    
    def predict(self, X,  Prototypes=None, batch_size=256):
        """
        预测离线数据集的准确率
        """
        X = tensor(X, dtype=torch.float32)
        if self.to_cuda:
            X = X.cuda()
        ds = TensorDataset(X)
        dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

        # set model in eval mode
        self.model.eval()

        """Evaluate the given data loader on the model and return predictions"""
        result = None
        with torch.no_grad():
            for x in dl:
                Pre_D,_ = self.model(x[0])
               # print(Pre_D.shape,'*********pre_D.shapebefore')
                Pre_D, Prototypes0 = self.change_D_Proto(Pre_D, Prototypes)
              #  print(Pre_D.shape,'*********pre_D.shapelatter')
                y_hat = self.predict_by_prototypes(Pre_D, Prototypes0)  
                y_hat = np.array([y_hat])
                result = y_hat if result is None else np.concatenate((result, y_hat), axis=0)
        return result