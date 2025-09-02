# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 10:05:30 2022

@author: Lenovo
"""

from os import path
from numpy import genfromtxt
from utils import normalize_data,preprocess0,normalize_y,normalize_label
import os
import pandas as pd
import numpy as np

task0_mode3 = r'E:\code\Data\TE\CIL\multiple_mode3\simout\class0' # 
task1_mode3 = r'E:\code\Data\TE\CIL\multiple_mode3\simout\class1' # 
task2_mode3 = r'E:\code\Data\TE\CIL\multiple_mode3\simout\class2' # 
task3_mode3 = r'E:\code\Data\TE\CIL\multiple_mode3\simout\class3' # 
task4_mode3 = r'E:\code\Data\TE\CIL\multiple_mode3\simout\class4' # 
task5_mode3 = r'E:\code\Data\TE\CIL\multiple_mode3\simout\class5' # 
task6_mode3 = r'E:\code\Data\TE\CIL\multiple_mode3\simout\class6' # 
task7_mode3 = r'E:\code\Data\TE\CIL\multiple_mode3\simout\class7' # 
task8_mode3 = r'E:\code\Data\TE\CIL\multiple_mode3\simout\class8' # 
task9_mode3 = r'E:\code\Data\TE\CIL\multiple_mode3\simout\class9' # 
task10_mode3 = r'E:\code\Data\TE\CIL\multiple_mode3\simout\class10' # 
task11_mode3 = r'E:\code\Data\TE\CIL\multiple_mode3\simout\class11' # 
task12_mode3 = r'E:\code\Data\TE\CIL\multiple_mode3\simout\class12' # 
task13_mode3 = r'E:\code\Data\TE\CIL\multiple_mode3\simout\class13' # 
task14_mode3 = r'E:\code\Data\TE\CIL\multiple_mode3\simout\class14' # 
task15_mode3 = r'E:\code\Data\TE\CIL\multiple_mode3\simout\class15' # 

task0 = r'E:\code\Data\2min_AE_zhonglv\volt\CIL\five_shot\class0'
task1 = r'E:\code\Data\2min_AE_zhonglv\volt\CIL\five_shot\class1'
task2 = r'E:\code\Data\2min_AE_zhonglv\volt\CIL\five_shot\class2'
task3 = r'E:\code\Data\2min_AE_zhonglv\volt\CIL\five_shot\class3'
task4 = r'E:\code\Data\2min_AE_zhonglv\volt\CIL\five_shot\class4'
task5 = r'E:\code\Data\2min_AE_zhonglv\volt\CIL\five_shot\class5'
    
class get_data_ACS():
    def __init__(self):
        """
        Args:
            slef.path_total:输入数据路径
            self.path_adj:邻接矩阵路径
        """
        self.ucr_dataset_base_folder = [task0,task1,task2,task3,task4,task5]
        self.number_class_each_task = {0:[0,1,2,3],1:[4],2:[5]}
        
    def csv_to_excel(self,path_total):
        """
        csv_to_excel:
            将数据中csv的数据转换为excel
        Args:
            path_total:包含所有文件路径的列表
        """
        for i in path_total:
            if i[-4:]=='.csv':
                csv=pd.read_csv(i,encoding='utf-8',header=None,engine='python')
                csv.to_excel(i.replace('.csv','.xlsx'),encoding='utf-8')
                os.remove(i)
        return 
    
    def del_excess_columns_indexs(self,path_total):
        """
        del_excess_columns_indexs:
            如果数据包含多余的行和列就将多余的行列删除，并写入新的excel中
        Args:
            path_total:包含所有文件路径的列表
        """
        for i in path_total:
            if i[-4:]=='.csv':
                csv=pd.read_csv(i,header=None,engine='python')
            else:
                csv=pd.read_excel(i,header=None)
            if pd.isnull(csv.iloc[0,0]):
                print(i)
                csv=csv.drop(csv.index[[0]])
                csv=csv.drop(csv.columns[[0]],axis=1)
                csv.to_excel(i,header=None,index=False)
        return
    
    def change_data_total_all(self,path_total):
        """
        change_data：
             对数据进行处理,将csv数据转为excel数据,删除出现多余第一行和第一列的数据
        Args:
            total_path_：处理完之后的数据的路径列表
            y_total：所有的标签列表
        """
        total_path_,y_total=self.data_label(path_total)
        self.csv_to_excel(total_path_)
        total_path_,y_train=self.data_label(path_total)
        self.del_excess_columns_indexs(total_path_)
        return total_path_,y_total
    
    def data_label(self,files):
        """
        data_label:
            获取数据和标签
        Args:
            total_path:用于记录所有文件的路径
            label_y:用于记录每个文件的标签
        """
        label_y=[]
        total_path=[]
        for filenames,dirnames,files in os.walk(files):
            for name in files:
                total_path+=[filenames+'/'+name]
                label_y.append(int(filenames[-1]))
        return total_path,label_y
    

    def load_single_task_dataset(self,path,task_id):
        total_path_,y_total= self.change_data_total_all(path)
       # print(total_path_,'&&&&&&&&&&total_path')
        total = np.array([np.array(preprocess0(pd.read_excel(total_path)[['potVolt']]).T) for total_path in total_path_])
         #将顺序打乱
        index = [i for i in range(len(total))]
        np.random.shuffle(index) 
        total = total[index]
        #total, scalar = normalize_data(total)
        
        train_num = int(total.shape[0]*0.8)
        if task_id == 0:
            X_train = total[:train_num,:,:]   
            X_test = total[train_num:,:,:]
        else:
            X_train = total[:5,:,:]   #假如不是第一个阶段,那么训练数据应该是小样本
            X_test = total[5:10,:,:]
        #数据标签 
        y_total = np.array(y_total)
      #  y_total = normalize_label(y_total)
        y_total = y_total[index]
        
        if task_id == 0:
            y_train = y_total[:train_num]   
            y_test = y_total[train_num:]
        else:
            y_train = y_total[:5]   #假如不是第一个阶段,那么训练数据应该是小样本
            y_test = y_total[5:10]
        #y_train = y_total[:train_num]
        return X_train,y_train,X_test,y_test
    
    def main(self):
        task_classes_dict = self.number_class_each_task
        Task = {i:[] for i in range(len(task_classes_dict))}
        X_test, y_test = None,None
        X_test_new_class_before_task, y_test_new_class_before_task = None, None
        for task_id,task in task_classes_dict.items():  
            X_train, y_train = None,None
            for flod_id in task:
                X_train0, y_train0, X_test0, y_test0 = self.load_single_task_dataset(self.ucr_dataset_base_folder[flod_id],task_id)
                X_train = X_train0 if X_train is None else np.concatenate((X_train0, X_train), axis=0)
                y_train = y_train0 if y_train is None else np.concatenate((y_train0, y_train), axis=0)
                
                #X_test集里面不能包含新类
                if task_id == 0:
                    X_test = X_test0 if X_test is None else np.concatenate((X_test0, X_test), axis=0)
                    y_test = y_test0 if y_test is None else np.concatenate((y_test0, y_test), axis=0)
                else:
                    X_test = X_test0 if X_test is None else np.concatenate((X_test_new_class_before_task, X_test), axis=0)
                    y_test = y_test0 if y_test is None else np.concatenate((y_test_new_class_before_task, y_test), axis=0)
               
                X_test_new_class_before_task = X_test0
                y_test_new_class_before_task = y_test0
            
            #假如是第0个任务，就取任务0的数据，是1个任务也取任务0的数据，假如是第二个任务就取任务1的数据加上任务1加上新类的数据
            if task_id == 0 :
                X_test = X_test
                y_test = y_test
                X_test_task_before = X_test
                y_test_task_before = y_test
            elif task_id == 1:
                X_test = X_test_task_before
                y_test = y_test_task_before
            else:
                X_test = X_test
                y_test = y_test
                
            #print(y_train,y_test,'y_trainX_test***********')
            Task[task_id] = {'X_train':X_train,'y_train':y_train,'X_test':X_test,'y_test':y_test}    
        return Task

class get_data_ucr():
    def __init__(self, ucr_dataset_name,ucr_dataset_base_folder):
        self.ucr_dataset_name=ucr_dataset_name
        self.ucr_dataset_base_folder=ucr_dataset_base_folder
    
    def load_dataset(self):
        dataset_path = path.join(self.ucr_dataset_base_folder, self.ucr_dataset_name)
        train_file_path = path.join(dataset_path, '{}_TRAIN'.format(self.ucr_dataset_name))
        test_file_path = path.join(dataset_path, '{}_TEST'.format(self.ucr_dataset_name))

        # training data
        train_raw_arr = genfromtxt(train_file_path, delimiter=',')
        train_data = train_raw_arr[:, 1:]
        train_labels = train_raw_arr[:, 0] - 1
        # one was subtracted to change the labels to 0 and 1 instead of 1 and 2
        
        # test_data
        test_raw_arr = genfromtxt(test_file_path, delimiter=',')
        test_data = test_raw_arr[:, 1:]
        test_labels = test_raw_arr[:, 0] - 1
        return train_data, train_labels, test_data, test_labels
    
    def main(self):
        X_train, y_train, X_test, y_test = self.load_dataset()
        y_train = normalize_label(y_train)
        y_test = normalize_label(y_test)

        X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1])
       # print(X_train,'X_train')
        #X_train, scaler = normalize_data(X_train)
       # X_test,_ = normalize_data(X_test,scaler)
        return X_train,y_train,X_test,y_test
    
class get_data_TEP():
    def __init__(self):
        '''
         __init__:
            函数初始化
        Args:
            ucr_dataset_name:[dataname1,dataname2]
            ucr_dataset_path:文件路径
            variable_selection:int 选择变量
            variable_dict:{} 变量对应的变量名
        '''
        self.ucr_dataset_base_folder = [task0_mode3,
                                        task1_mode3,
                                        task2_mode3,
                                        task3_mode3,
                                        task4_mode3,
                                        task5_mode3,
                                        task6_mode3,
                                        task7_mode3,
                                        task8_mode3,
                                        task9_mode3,
                                        task10_mode3,
                                        task11_mode3,
                                        task12_mode3,
                                        task13_mode3,
                                        task14_mode3,
                                        task15_mode3]
        self.number_class_each_task = {0:[0,1,2,3,4,5,6,7,8],1:[9],2:[10],3:[11],4:[12],5:[13],6:[14],7:[15]}
        self.variable_selection = 21
        self.variable_dict = {3:'A_and_C_feed',16:'Stripper underflow',17:'Stripper_temperature', 21:'Separator_cooling_water_outlet_temperature'}
        
    def data_label(self,files,file_choose):
        """
        data_label:
            将所有子文件的路径记录下来，并选择‘simout’或者'xmv'
        Args:
            files:所有文件路径
            file_choose:从文件里面选择‘simout’或者'xmv'
        
        Output:
            Task:{0:{'X_train':[],'y_train':[0,1,2,3,4,5,6,7,8],'X_test':[],'y_train':[0,1,2,3,4,5,6,7,8]},
                  1:{'X_train':[],'y_train':[9],'X_test':[],'y_train':[0,1,2,3,4,5,6,7,8]},
                  2:{'X_train':[],'y_train':[10],'X_test':[],'y_train':[0,1,2,3,4,5,6,7,8,9]}}
        """
        label_y=[]
        total_path=[]
        for filenames,dirnames,files in os.walk(files):
            if dirnames==[] and file_choose in filenames:
                for name in files:
                    #print(filenames,'&&&&&&&&&&&&name**************')
                    if '~' not in name and self.variable_dict[self.variable_selection] in name: 
                        total_path.append(filenames+'/'+name)
                        if any(char in filenames for char in ['10','11','12','13','14','15']):
                            label_y.append(int(filenames[-2:]))
                        else:
                            label_y.append(int(filenames[-1:]))
        return total_path,label_y
    
    def load_single_task_dataset(self,path,task_id):
        total_path_,y_total= self.data_label(path,'simout')
       # print(total_path_,'&&&&&&&&&&total_path')
        total = np.array([np.array(preprocess0(pd.read_excel(total_path)).T) for total_path in total_path_])
         #将顺序打乱
        index = [i for i in range(len(total))]
        np.random.shuffle(index) 
        total = total[index]
        #total, scalar = normalize_data(total)
        
        train_num = int(total.shape[0]*0.8)
        if task_id == 0:
            X_train = total[:train_num,:,:]   
        else:
            X_train = total[:5,:,:]   #假如不是第一个阶段,那么训练数据应该是小样本
        
        X_test = total[train_num:,:,:]
        #数据标签 
        y_total = np.array(y_total)
      #  y_total = normalize_label(y_total)
        y_total = y_total[index]
        
        if task_id == 0:
            y_train = y_total[:train_num]   
        else:
            y_train = y_total[:5]   #假如不是第一个阶段,那么训练数据应该是小样本
        #y_train = y_total[:train_num]
        y_test = y_total[train_num:]
        return X_train,y_train,X_test,y_test
    
    def main(self):
        task_classes_dict = self.number_class_each_task
        Task = {i:[] for i in range(len(task_classes_dict))}
        X_test, y_test = None,None
        X_test_new_class_before_task, y_test_new_class_before_task = None, None
        for task_id,task in task_classes_dict.items():  
            X_train, y_train = None,None
            for flod_id in task:
                X_train0, y_train0, X_test0, y_test0 = self.load_single_task_dataset(self.ucr_dataset_base_folder[flod_id],task_id)
                X_train = X_train0 if X_train is None else np.concatenate((X_train0, X_train), axis=0)
                y_train = y_train0 if y_train is None else np.concatenate((y_train0, y_train), axis=0)
                
                #X_test集里面不能包含新类
                if task_id == 0:
                    X_test = X_test0 if X_test is None else np.concatenate((X_test0, X_test), axis=0)
                    y_test = y_test0 if y_test is None else np.concatenate((y_test0, y_test), axis=0)
                else:
                    X_test = X_test0 if X_test is None else np.concatenate((X_test_new_class_before_task, X_test), axis=0)
                    y_test = y_test0 if y_test is None else np.concatenate((y_test_new_class_before_task, y_test), axis=0)
               
                X_test_new_class_before_task = X_test0
                y_test_new_class_before_task = y_test0
            
            #假如是第0个任务，就取任务0的数据，是1个任务也取任务0的数据，假如是第二个任务就取任务1的数据加上任务1加上新类的数据
            if task_id == 0 :
                X_test = X_test
                y_test = y_test
                X_test_task_before = X_test
                y_test_task_before = y_test
            elif task_id == 1:
                X_test = X_test_task_before
                y_test = y_test_task_before
            else:
                X_test = X_test
                y_test = y_test
                
            #print(y_train,y_test,'y_trainX_test***********')
            Task[task_id] = {'X_train':X_train,'y_train':y_train,'X_test':X_test,'y_test':y_test}    
        return Task

path_simulate47 = 'E:/code/7new_class_learning/Data/TrainDataset4_7/TrainDatasetA3' #总共5个类别
path_streaming_simulate47 = 'E:/code/7new_class_learning/Data/StreamingTimeSeries4_7/StreamingTimeSeriesA3.xlsx' 
    
class GetSimulateData():
    def __init__(self):
        """
        Args:
            slef.path_total:输入数据路径
            self.path_adj:邻接矩阵路径
        """
        path_simulate58 = path_simulate47
        path_streaming_simulate58 = path_streaming_simulate47
        
        self.path_total = path_simulate58
        self.path_streaming_data = path_streaming_simulate58
    
    def get_streaming_data_label(self, streaming_array):
        fault_dict = {
                      1:list(range(500,550)), #故障1
                      2:list(range(900,950)), #故障2
                      3:list(range(1300,1350)), #故障3
                      5:list(range(2100,2150)), #故障4  将故障4设置为新故障,给故障序号5，因此故障4和故障5的序号是相反的
                      4:list(range(3100,3150))  #故障5  故障5但是标签是4，因为所有故障要按照从0到4排序，所以故障5的索引是4.
                      }
        fault_index_list = [value for value_list in fault_dict.values() for value in value_list]
        
        streaming_label = []
        for i in range(len(streaming_array[0])):
            if i not in fault_index_list:
                streaming_label.append(0)  #表示是正常的类别
            else:
                key = [key for key, value in fault_dict.items() if i in value]
                streaming_label.append(key[0])
        return np.array(streaming_label)
        
    def data_label(self,files0):
        """
        data_label:
            获取数据和标签
        Args:
            total_path:用于记录所有文件的路径
            label_y:用于记录每个文件的标签
        """
        label_y = []
        total_path = []
        for filenames,dirnames,files in os.walk(files0):
            for name in files:
                total_path += [filenames+'/'+name]
                label_y.append(int(filenames[-1]))
        return total_path,label_y
    
    def change_data(self,path_total):
        """
        change_data：
             对数据进行处理,将csv数据转为excel数据,删除出现多余第一行和第一列的数据
        Args:
            total_path_：处理完之后的数据的路径列表
            y_total：所有的标签列表
        """
        total_path_,y_total = self.data_label(path_total)
        return total_path_,y_total
    
    def main(self):
        total_path_,y_total = self.change_data(self.path_total)
        #total = np.array([np.array(preprocess0(pd.read_csv(total_path,header=None)).T) for total_path in total_path_])
        total = np.array([np.array(pd.read_csv(total_path,header=None).T) for total_path in total_path_])
        streaming_array = np.array(pd.read_excel(self.path_streaming_data,header=None).T)
        
        #将顺序打乱
        index = [i for i in range(len(total))]
        np.random.shuffle(index)
        total = total[index]
        
        train_num = int(total.shape[0]*0.7)
        
        X_train = total[:train_num,:,:]
        X_test = total[train_num:,:,:]
        
        #数据标签 
        y_total = np.array(y_total)
        y_total = y_total[index]
        y_train = y_total[:train_num]
        y_test = y_total[train_num:]
        y_train, y_test = normalize_y(y_train,y_test)
        
     #   print(y_test,y_train,'************y_test,y_train')
        X_train, scaler = normalize_data(X_train)
        X_test,_= normalize_data(X_test,scaler)
        #print(y_train,'&&&&&&&&&&&&&&&&y_train')
        
        #获取流时间序列数据的标签
        streaming_label = self.get_streaming_data_label(streaming_array)
        #print(streaming_label,'streaming_label*****')
        return X_train,y_train,X_test,y_test,streaming_array,streaming_label,scaler

#path0 = 'E:/code/7new_class_learning/Data/TrainDataset'
#simulate = GetSimulateData(path0)
#X_train,y_train,X_test,y_test = simulate.main()
    