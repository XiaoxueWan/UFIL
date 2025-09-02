# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 10:37:54 2022

@author: Lenovo
"""

from utils import get_weights_via_kmeans,plot_shapelets,normalize_data,plot_sub_fig
from get_data.get_data import get_data_ACS, GetSimulateData, get_data_TEP

'''从模型部分别导入预训练，元学习和增量阶段的模型'''
from models.incremental_session import LearningShapeletsIncremental
from models.pre_trained_model import LearningShapeletsPretrain
from models.meta_trainer import FakeTrainer
from sklearn.metrics import f1_score,accuracy_score

from matplotlib import pyplot
from torch import optim
from torch import tensor

import pandas as pd
import numpy as np
import time
import torch
import gc
gc.collect()

class Main():
    def __init__(self,dataset_name,
                      K,
                      Lmin,
                      temp_factor_pre_train,
                      temp_factor_meta_learning,
                      learning_rate=0.01,
                      epoch=2000,
                      learning_rate_meta=0.01,
                      epoch_meta=200,
                      batch_size=234,
                      lw=0.01,
                      ONLINE_MONITOR=False,
                      alpha_new_class = 20
                      ):
        '''
        s_num:shapelet数量，K为shapelet数量占所有的比例
        s_length:shapelet的长度，Lmin为shapelet长度占所有的比例
        self.lr:学习率
        self.online: 流时间序列数据
        '''
        '''输入数据'''
        if dataset_name=='Simulate':
            self.X_train,self.y_train,self.X_test,self.y_test,online,online_label,self.scaler = GetSimulateData().main()
        elif dataset_name=='Alu':
            self.task_set = get_data_ACS().main()
            self.X_train = self.task_set[0]['X_train']
            self.y_train = self.task_set[0]['y_train']
            self.X_test = self.task_set[0]['X_test']
            self.y_test = self.task_set[0]['y_test']
            online = np.array([[0,0],[0,0]])
        elif dataset_name=='TEP':
            self.task_set = get_data_TEP().main()
            self.X_train = self.task_set[0]['X_train']
            self.y_train = self.task_set[0]['y_train']
            self.X_test = self.task_set[0]['X_test']
            self.y_test = self.task_set[0]['y_test']
            online = np.array([[0,0],[0,0]])
        else:
            raise ValueError("dataset_name必须是Simulate,或者是Alu,或者是TEP")
        #print(self.y_train.shape,'******self.y_train.shape')
        
        if self.y_test.min()<0:
            self.y_test = self.y_test-self.y_test.min()
        self.dataset_name = dataset_name
        
        #print(self.X_train.shape, self.y_train.shape,'self.X_train.shape, self.y_train.shape')
        '''主要参数'''
        self.lr = learning_rate
        self.w = lw
        self.epsilon = 1e-7
        self.epoch = epoch
        self.temp_factor_pre_train = temp_factor_pre_train
        self.temp_factor_meta_learning = temp_factor_meta_learning
        self.batch_size = batch_size
        self.ONLINE_MONITOR = ONLINE_MONITOR
        self.epoch_meta = epoch_meta
        self.learning_rate_meta = learning_rate_meta
        self.alpha_new_class = alpha_new_class
        
        '''shapelet参数'''
        s_num = int(K*self.X_train.shape[0])   
        s_lenght = int(Lmin*self.X_train.shape[2])  
        self.shapelets_size_and_len = {s_lenght:s_num}  
        self.s_num = K
        self.s_lenght = Lmin
        self.shapelet_length = sum(self.shapelets_size_and_len.keys())
        self.shapelet_num = sum(self.shapelets_size_and_len.values())
        
        '''在线监测数据'''
        if dataset_name!='TEP' and dataset_name!='Alu':
            self.online = torch.Tensor(online) #在线数据的获取
            self.online = torch.reshape(self.online,[1,1,online.shape[1]])
            self.online_label = online_label
            self.online_label_Tensor = torch.Tensor(online_label)
        else:
            self.online = 0
            self.scaler = 0
        
        '''实验记录'''
        t = time.localtime()
        self.record = {
                     'time':str(t.tm_year) + '/'+ str(t.tm_mon) +'/'+ str(t.tm_mday)+'/'+ str(t.tm_hour)+':'+ str(t.tm_min), 
                     'dataset_name':dataset_name,
                     'K': K,
                     'Lim': Lmin, 
                     'temp_factor_pre_train':temp_factor_pre_train,
                     'temp_factor_meta_learning':temp_factor_meta_learning,
                     'epoch':self.epoch,
                     'epoch_meta':self.epoch_meta,
                     'learning_rate': self.lr, 
                     'meta_learning_rate': self.learning_rate_meta, 
                     'alpha_new_class':self.alpha_new_class,
                     'lw': self.w, 
                     'batch_size':self.batch_size, 
                     'train_time':0, 
                     'test_time':0
                     }
        
    '''-----------------------------------------pretrain_model_initial_and_training--------------------------------------------------'''
    def pretrain(self): 
        '''对第0个阶段的数据集进行预训练'''
        n_ts,n_channels,len_ts = self.X_train.shape
        num_classes = len(set(self.y_train))
        dist_measure = 'euclidean'
        learning_shapelets_pretrained = LearningShapeletsPretrain(
                                                                   shapelets_size_and_len = self.shapelets_size_and_len,
                                                                   in_channels = n_channels,
                                                                   num_classes = num_classes,
                                                                   to_cuda = False,
                                                                   verbose = 1,
                                                                   dist_measure = dist_measure,
                                                                   ucr_dataset_name = self.dataset_name,
                                                                   temp_factor = self.temp_factor_pre_train,
                                                                   online = self.online,
                                                                   scalar = self.scaler
                                                                   )
        '''初始化预训练模型的优化器'''
        for i, (shapelets_size, num_shapelets) in enumerate(self.shapelets_size_and_len.items()):
            weights_block = get_weights_via_kmeans(self.X_train, shapelets_size, num_shapelets)
            learning_shapelets_pretrained.set_shapelet_weights_of_block(i, weights_block)
            
        optimizer = optim.Adam(learning_shapelets_pretrained.model.parameters(), lr=self.lr, weight_decay=self.w, eps=self.epsilon)
        learning_shapelets_pretrained.set_optimizer(optimizer)
        '''训练'''
        losses,Prototypes,Prototypes0 = learning_shapelets_pretrained.fit(
                                                                            self.X_train, 
                                                                            self.y_train, 
                                                                            epochs=self.epoch, 
                                                                            batch_size=256, 
                                                                            shuffle=False, 
                                                                            drop_last=False
                                                                            )
       # Accuracy,Accuracy0,test_time = self.eval_accuracy(learning_shapelets_pretrained, self.X_test, self.y_test, Prototypes, Prototypes0)
        
        '''plot_shapelets'''
        shapelets = learning_shapelets_pretrained.get_shapelets()
        shapelet_transform = learning_shapelets_pretrained.transform(self.X_test)
        plot_shapelets(self.X_test, shapelets, self.y_test, shapelet_transform, self.dataset_name)
        #_,_ = learning_shapelets_pretrained.online_monitor(Prototypes)
        return Prototypes, learning_shapelets_pretrained, losses
    
    '''-----------------------------------------meta_learning_model_initial--------------------------------------------------'''
    #'''初始化元模型'''
    def initialize_meta_training_model(self, pretrain_model):
        meta_train = FakeTrainer(
                                   shapelets_size_and_len = self.shapelets_size_and_len,
                                   in_channels = 1,
                                   num_classes = 1,
                                   dist_measure = 'euclidean',
                                   ucr_dataset_name = self.dataset_name,
                                   temp_factor = self.temp_factor_meta_learning,
                                   epoch = self.epoch_meta,
                                   verbose=0, 
                                   to_cuda = False,
                                   alpha_new_class = self.alpha_new_class
                                   )
        #初始化模型参数
        meta_train.set_shapelet_weights_of_block(0, pretrain_model.get_shapelets())  #用预训练的shapelet作为元模型的初始化shapelets
        optimizer = optim.SGD([
                                #{"params":meta_train.model.parameters()},
                                {"params":meta_train.slf_attn.parameters()},
                                {"params":meta_train.slf_attn_classify.parameters()}], 
                                 lr=self.learning_rate_meta, 
                                 momentum=0.9)
        meta_train.set_optimizer(optimizer)
        return meta_train
    
    '''-----------------------------------------incremental_model_initial--------------------------------------------------'''
    '''@初始化后面的增量阶段模型'''
    def initialize_incremental_model(self, learning_shapelets_last_task):
        
        def initialize_incremental_model0(meta_model):
            optimizer = optim.SGD([{"params":meta_model.model.parameters()}], lr=0.001, momentum=0.9) #设置这个优化器其实是没有意义的，因为根本就没有用到优化器
            meta_model.set_optimizer(optimizer)
            return meta_model
        
        learning_shapelets = LearningShapeletsIncremental( shapelets_size_and_len = self.shapelets_size_and_len, 
                                                             in_channels = 1, 
                                                             num_classes = 2,
                                                             dist_measure='euclidean',
                                                             ucr_dataset_name='comman', 
                                                             temp_factor=2, 
                                                             online=self.online
                                                             )   #----------------------------------
        
        learning_shapelets.set_shapelet_weights_of_block(0, learning_shapelets_last_task.get_shapelets())
        initialize_incremental_model0(learning_shapelets)
        return learning_shapelets
    
    '''-----------------------------------------model_evaluate_plot--------------------------------------------------'''
    def eval_accuracy(self, model, X, Y, Prototypes, Prototypes0, IS_NEW_CLASS, new_class_value):
        '''
        评估模型的预测精度
           
        参数：
        --------
        Prototypes是最后得到的原型
        Prototypes0是最开始训练时候的原型
        '''
        start0 = time.clock()
        predictions0 = model.predict(X, Prototypes0) 
        if len(predictions0.shape) == 2:
            predictions0 = np.squeeze(predictions0)
        print(Y,predictions0)
        Accuracy0 = (predictions0 == Y).sum() / Y.size
        print(f"Accuracy_begin:", Accuracy0)
        end0 = time.clock()
        test_time = end0-start0
        
        new_class_accuracy_numbers = (torch.sum(IS_NEW_CLASS==1)).sum()
        new_class_total_numbers = len(IS_NEW_CLASS)
        
        predictions = model.predict(X, Prototypes) 
        if len(predictions.shape) == 2:
            predictions = np.squeeze(predictions)
        print(type(Y),type(predictions),'*********type(Y),type(predictions)')
        Accuracy = ((predictions == Y).sum() + new_class_accuracy_numbers) / (Y.size+new_class_total_numbers)
        print(f"Accuracy:", Accuracy)
        
        #将所有的已知类和未知类拼接在一起
        
        y_true_all_class = np.concatenate((Y,np.array([new_class_value]*int(IS_NEW_CLASS.shape[0]))))
        y_pred_all_class = np.concatenate((predictions,torch.squeeze(IS_NEW_CLASS).numpy()))
        F1 = f1_score(y_true_all_class, y_pred_all_class, average = 'weighted')
        return Accuracy,Accuracy0,test_time,F1
    
    def test_all_task(self, learning_shapelets, key, Prototypes, Prototypes0, IS_NEW_CLASS, new_class_value):
        '''测试所有任务'''
        X_test = self.task_set[key]['X_test']
        y_test = self.task_set[key]['y_test']
        
        Accuracy, Accuracy0, test_time, F1 = self.eval_accuracy(learning_shapelets,
                                                              X_test, 
                                                              y_test, 
                                                              Prototypes, 
                                                              Prototypes0, 
                                                              IS_NEW_CLASS,
                                                              new_class_value)
        
        self.record['accuracy'] = Accuracy
        self.record['F1'] = F1
        self.record['test_time'] = test_time
      #  print(self.record,'********selfrecord')
        
        '''to_excel'''
        record = pd.read_excel('results/record_auto.xlsx')
        record = record.append(self.record, ignore_index = True)
        record = record.reindex(
                                 columns=['time',
                                          'dataset_name',
                                          'K',
                                          'Lim', 
                                          'temp_factor_pre_train',
                                          'temp_factor_meta_learning',
                                          'epoch',
                                          'epoch_meta',
                                          'learning_rate', 
                                          'meta_learning_rate',
                                          'lw', 
                                          'batch_size', 
                                          'train_time', 
                                          'test_time',
                                          'new_class_accuracy',
                                          'accuracy',
                                          'F1'
                                           ]+list(record.columns.drop(['time',
                                                                       'dataset_name',
                                                                       'K',
                                                                       'Lim', 
                                                                       'temp_factor_pre_train',
                                                                       'temp_factor_meta_learning',
                                                                       'epoch',
                                                                       'epoch_meta',
                                                                       'learning_rate', 
                                                                       'meta_learning_rate',
                                                                       'lw', 
                                                                       'batch_size', 
                                                                       'train_time', 
                                                                       'test_time',
                                                                        'new_class_accuracy',
                                                                       'accuracy',
                                                                       'F1'
                                                                       ])))
        record.to_excel('results/record_auto.xlsx', index = False)
        return
    
    '''-----------------------------------------New_class_learning--------------------------------------------------'''
    #旧类分类
    def predict_by_prototypes(self, Pre_D, Prototypes):
        '''
        对每个shapelet嵌入之后的Pre_D,根据原型，获取它的预测的标签
        '''
        cos_dict = {}
        for classes,Prototype in  Prototypes.items():
            cos = torch.cosine_similarity(Pre_D,Prototype,dim=-1)
           # print(cos,'*********cos')
            cos = cos.unsqueeze(1)
            cos_dict[classes] = cos
            
        max_cos = max(cos_dict.values())
        
        max_keys = [key for key, value in cos_dict.items() if value == max_cos]
        return max_keys[0]
    
    def New_class_learning_task(self,learning_shapelets_last_task, meta_train_model, Prototypes_dict, num_task, task):
        '''
        切分好的数据的新类学习
        '''
        learning_shapelets = self.initialize_incremental_model(learning_shapelets_last_task) #对增量阶段的shapelet学习进行初始化
        
        #获取当前任务的数据
        X_train = task['X_train']
        y_train = task['y_train']
        X_train = tensor(X_train, dtype=torch.float32)
        y_train = tensor(y_train, dtype=torch.float32)
        
        #将新类的样本先用旧模型预测一下
        new_class_predict_in_old_class = torch.Tensor(meta_train_model.predict(X_train,Prototypes_dict))
        new_class_predict_in_old_class = new_class_predict_in_old_class.unsqueeze(-1).unsqueeze(-1)
        new_class_value = int(len(Prototypes_dict))
        
        #获取当前是否是新类的判断值IS_NEW_CLASS_0_or_1
        last_distance0 = torch.full((1,1,self.shapelet_num),100000.0)
        D,last_distance0 = learning_shapelets_last_task(X_train,last_distance0)
       # print(D,Prototypes_dict,'D.shape,Prototypes_dict')
        D0, Prototypes0 = self.transformer(D, Prototypes_dict)
        IS_NEW_CLASS_prob = self.classfy_new_class(D0,Prototypes0)
        
        #假如预测的概率值大于0.5,就判断为新类，否则从旧类里面选个类别作为预测的类别
        IS_NEW_CLASS = torch.where(IS_NEW_CLASS_prob>0.5, torch.Tensor([new_class_value]), torch.Tensor(new_class_predict_in_old_class))
        buffer_X, buffer_Y = torch.Tensor([]), torch.Tensor([])  #buffer清空
        
        #判断是否更新模型
        if torch.sum(IS_NEW_CLASS==new_class_value) > len(IS_NEW_CLASS)/2:
            buffer_X = X_train
            buffer_Y = torch.Tensor([len(Prototypes_dict)]*len(IS_NEW_CLASS))
             #对新类数据进行测试 
        else:
            buffer_X, buffer_Y = torch.Tensor([]), torch.Tensor([])  #buffer清空
        
        new_class_accuracy = (torch.sum(IS_NEW_CLASS==new_class_value)).sum() / len(IS_NEW_CLASS)
        self.record['new_class_accuracy'] = new_class_accuracy.item()
        
        #用有监督的数据进行模型更新
        Prototypes_dict = learning_shapelets.fit(
                                                X_train, 
                                                y_train, 
                                                Prototypes_dict,
                                                epochs=1, 
                                                batch_size=256, 
                                                shuffle=False, 
                                                drop_last=False
                                                ) #假如新类样本较多，就用新类样本来更新模型的原型。
        
        #对旧类数据进行测试
        self.test_all_task(meta_train_model, num_task, Prototypes_dict, Prototypes0, IS_NEW_CLASS, new_class_value)
        return Prototypes_dict
    
    def New_class_learning_streaming(self, learning_shapelets_last_task, Prototypes):
        '''
        流时间序列数据的在线监测，用训练好的模型测试在线数据的分类精度。
        '''
        learning_shapelets = self.initialize_incremental_model(learning_shapelets_last_task) #对增量阶段的shapelet学习进行初始化
        
        last_distance0 = torch.full((1,1,self.shapelet_num),100000.0)
        last_distance1 = torch.full((1,1,self.shapelet_num),100000.0)
        X, IS_NEW_CLASS_0_or_1_list, y_hat_list = [],[],[] #记录数据的列表
        buffer_X, buffer_Y = torch.Tensor([]), torch.Tensor([])
        num, num_new_class = 0,0
        for t in range(self.shapelet_length,4000,10):    #代表遍历的数据点
            if len(buffer_X)<5 and num_new_class<5:   
                '''截取数据，并对数据进行z-normalize规范化，规范的指标是训练集的方差和均值'''
                X_t = self.online[:,:,t-self.shapelet_length:t]
                X_t,_ = normalize_data(X_t,self.scaler)
                X_t = torch.from_numpy(X_t).float()
                
                '''判断一个数据是否是新的数据'''
                D,last_distance0 = learning_shapelets_last_task(X_t,last_distance0)
              
                '''新类数据放入buffer中'''
                
                if num>10: #表示每多少个点开始重新计算shapelet与时间序列的相似度
                    num=0
                    last_distance0=last_distance1
                    D0, Prototypes0 = self.transformer(D, Prototypes)
                    IS_NEW_CLASS = self.classfy_new_class(D0,Prototypes0)
                    IS_NEW_CLASS_0_or_1 = torch.where(IS_NEW_CLASS>0.5, torch.Tensor([1]), torch.Tensor([0]))
                    
                    if self.online_label[t]==5: #如果是新类
                        buffer_X = torch.cat((buffer_X,X_t),dim=0)
                        buffer_Y = torch.cat((buffer_Y,torch.Tensor([5])),dim=0)
                        num_new_class += 1
                        y_hat = len(Prototypes)
                    else:
                        #如果不是新类
                        num_new_class -= 1
                        y_hat = self.predict_by_prototypes(D0, Prototypes0)  
                    
                    '''对数据进行记录'''
                    X.append(t)
                    IS_NEW_CLASS_0_or_1_list.append(torch.squeeze(IS_NEW_CLASS_0_or_1))
                    y_hat_list.append(y_hat)
                else:
                    num+=1
                
            else:
                Prototypes = learning_shapelets.fit(
                                                    buffer_X, 
                                                    buffer_Y, 
                                                    Prototypes,
                                                    epochs=1, 
                                                    batch_size=256, 
                                                    shuffle=False, 
                                                    drop_last=False
                                                    ) #假如新类样本较多，就用新类样本来更新模型的原型。
                buffer_X, buffer_Y = torch.Tensor([]), torch.Tensor([])  #buffer清空
                num_new_class = 0
        plot_sub_fig(X,IS_NEW_CLASS_0_or_1_list,'IS_NEW_CLASS_0_or_1_list')
        #print(len(y_hat_list),len(self.online_label),'********len(y_hat_list),len(self.online_label)')
        plot_sub_fig(X,y_hat_list,'y_hat_list')
        
        def calculate_y_true_list(X,y):
            y_ture_list = []
            for index in X:
                #print(index,y[index],'index*****')
                y_ture_list.append(y[index])
            return y_ture_list
        
        y_ture_list = calculate_y_true_list(X, self.online_label)   
        
        def calculate_streaming_accuracy_F1(y_ture_list,y_hat_list):
            F1_list,accuracy_list = [],[]
            for i in range(len(y_ture_list)):
                F1 = f1_score(y_ture_list[:i], y_hat_list[:i], average = 'weighted')
                F1_list.append(F1)
                accuracy = accuracy_score(y_ture_list[:i], y_hat_list[:i])
                accuracy_list.append(accuracy)
            return accuracy_list,F1_list
            
        accuracy_list,F1_list = calculate_streaming_accuracy_F1(y_ture_list,y_hat_list)
        X_pd = pd.DataFrame(X)
        accuracy_pd = pd.DataFrame(accuracy_list)
        F1_pd = pd.DataFrame(F1_list)
        y_true_pd = pd.DataFrame(y_ture_list)
        
        t = time.localtime()
        to_file_name = str(t.tm_year) + '_'+ str(t.tm_mon) +'_'+ str(t.tm_mday)+'_'+ str(t.tm_hour)+'_'+ str(t.tm_min)+'_'+'K'+str(self.s_num)+'L'+str(self.s_lenght)
        X_pd.to_excel('results/results_values/'+to_file_name+'index_streaming.xlsx',index=False,header=False)
        accuracy_pd.to_excel('results/results_values/'+to_file_name+'accuracy_streaming.xlsx',index=False,header=False)
        F1_pd.to_excel('results/results_values/'+to_file_name+'F1_streaming.xlsx',index=False,header=False)
        y_true_pd.to_excel('results/results_values/'+to_file_name+'y_true.xlsx',index=False,header=False)
        
        plot_sub_fig([1,2,3],accuracy_list,'streaming_accuracy')
        plot_sub_fig([1,2,3],F1_list,'streaming_accuracy')
        return buffer_X, buffer_Y
    
    '''-----------------------------------------main_model--------------------------------------------------'''
    def train_model(self):
        '''预训练'''
        pretrain_Prototypes, pretrain_model, losses_pre = self.pretrain()    #获取预训练模型
        for key in pretrain_Prototypes:
            pretrain_Prototypes[key] = pretrain_Prototypes[key].detach()
        plot_sub_fig(losses_pre,0,'pretrain_loss')
       # print(pretrain_model.get_shapelets(),'&&&&&&&&&&&&&&&&')
        '''元训练'''
        #TEP数据就用不同的任务的方式训练，非TEP数据就用流时间序列的方式获取数据
        if self.dataset_name == 'TEP' or self.dataset_name == 'Alu':
            for num_task, value in self.task_set.items():
                if num_task == 0:
                    #用基础任务的数据训练元学习模型
                    meta_train_model = self.initialize_meta_training_model(pretrain_model)
                    self.transformer, self.classfy_new_class, learning_shapelets_last_task, losses, Prototypes_0 = meta_train_model.meta_train(
                                                                                                                                                 {'X_train':self.X_train,
                                                                                                                                                 'y_train':self.y_train,
                                                                                                                                                 'X_test':self.X_test,
                                                                                                                                                 'y_test':self.y_test},
                                                                                                                                                  pretrain_Prototypes
                                                                                                                                                )
                    plot_sub_fig(losses,0,'meta_learning_loss')
                    prototypes_dict = pretrain_Prototypes
                else:
                    prototypes_dict = self.New_class_learning_task(learning_shapelets_last_task,meta_train_model,prototypes_dict,num_task,value)
        else:
            meta_train_model = self.initialize_meta_training_model(pretrain_model)
            self.transformer, self.classfy_new_class, learning_shapelets_last_task, losses, Prototypes_0 = meta_train_model.meta_train(
                                                                                                                                         {'X_train':self.X_train,
                                                                                                                                         'y_train':self.y_train,
                                                                                                                                         'X_test':self.X_test,
                                                                                                                                         'y_test':self.y_test},
                                                                                                                                          pretrain_Prototypes
                                                                                                                                          )
            plot_sub_fig(losses)
    #        '''在线监测，当遇到新类的时候触发增量学习'''
           # print(learning_shapelets_last_task.get_shapelets(),'&&&&&&&&&&&&&&&&')
            self.New_class_learning_streaming(learning_shapelets_last_task,pretrain_Prototypes)
        return
        
        
        
    
    
    
    