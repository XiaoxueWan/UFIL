# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 19:45:49 2022

@author: Lenovo
"""
'''
2022_04_22: 增加中铝阳极电流数据样本训练
'''
import os
import json
import shutil
import argparse

from main import Main

cache_dir = os.path.join(os.getcwd(),"__pycache__")
shutil.rmtree(cache_dir)

'''---------------------------------------------------Data-------------------------------------------------------'''
path='E:/Code/Data/Common_dataset/UCR_TS_Archive_2015'
path_zhonglv=r'E:\Python\Classification_ACS_20211027\shapelet_learning_ACS\data\zhong_lv'
path_yangxin=r'E:\Python\Classification_ACS_20211027\shapelet_learning_ACS\data\total_all_origin'
path_zhonglv8310=r'E:\Python\Data\2min_AE_zhonglv\yu_zhong_lv8310'
path_zhonglv8311=r'E:\Python\Data\2min_AE_zhonglv\yu_zhong_lv8311'
path_zhonglv8312=r'E:\Python\Data\2min_AE_zhonglv\yu_zhong_lv8312'
path_zhonglv8331=r'E:\Python\Data\2min_AE_zhonglv\yu_zhong_lv8331'
path_zhonglv8331_20_5=r'E:\Python\Data\2min_AE_zhonglv\zhong_lv8331_20__5'
path_zhonglv8331min=r'E:\Code\Data\2min_AE_zhonglv\yu_zhong_lv8331_min'
path_zhonglv_knowledge2=r'E:\code\Data\2min_AE_zhonglv\volt\knowledge\FAE_nor'#含有知识的
path_zhonglv_knowledge=r'E:\code\Data\2min_AE_zhonglv\volt\knowledge\FAE_SAE_nor'

path_zhonglv8331_20_5_multi_class=r'E:\Python\Data\2min_AE_zhonglv\zhong_lv8331_20__5_multi_class_normal_100_100'

'''仿真数据集'''
path_simulate315 = 'E:/code/7new_class_learning/Data/TrainDataset315/TrainDatasetA0' #总共5个类别
path_streaming_simulate315 = 'E:/code/7new_class_learning/Data/StreamingTimeSeries315/StreamingTimeSeriesA0.xlsx'
path_simulate314 = 'E:/code/7new_class_learning/Data/TrainDataset314/TrainDatasetA4' #总共5个类别
path_streaming_simulate314 = 'E:/code/7new_class_learning/Data/StreamingTimeSeries314/StreamingTimeSeriesA4.xlsx'

path_simulate42 = 'E:/code/7new_class_learning/Data/TrainDataset4_2/TrainDataset' #总共5个类别
path_streaming_simulate42 = 'E:/code/7new_class_learning/Data/StreamingTimeSeries4_2/StreamingTimeSeriesA0.xlsx'

path_simulate47 = 'E:/code/7new_class_learning/Data/TrainDataset4_7/TrainDatasetA3' #总共5个类别
path_streaming_simulate47 = 'E:/code/7new_class_learning/Data/StreamingTimeSeries4_7/StreamingTimeSeriesA3.xlsx'

path_simulate_imbalance = 'E:/code/7new_class_learning/Data/TrainDataset_Imbalance'

#5月8号整的新的仿真数据
path_simulate58 = 'E:/code/7new_class_learning/Data/TrainDataset5_8'
path_streaming_simulate58 = 'E:/code/7new_class_learning/Data/StreamingTimeSeries5_8/StreamingTimeSeriesInsect_telemetry.xlsx'

'''Alu数据集'''
path_Alu = r'E:/code/Data/2min_AE_zhonglv/volt/CIL/five_shot'

'''UCR数据集'''
path_UCR='E:/code/Data/Common_dataset/UCR_TS_Archive_2015'

'''-------------------------------------main--------------------------------------'''

def load_arg(args):
    main = Main(
                dataset_name = args['dataset_name'],  #'Simulate'
                K = args['K'], 
                Lmin = args['Lmin'], 
                temp_factor_pre_train = args['temp_factor_pre_train'],
                temp_factor_meta_learning = args['temp_factor_meta_learning'],
                learning_rate = args['learning_rate'],
                epoch = args['epoch'], 
                learning_rate_meta = args['learning_rate_meta'],
                epoch_meta = args['epoch_meta'],
               # batch_size = 10,
                lw = args['lw'],
                ONLINE_MONITOR = args['ONLINE_MONITOR'],
                alpha_new_class = args['alpha_new_class']
                )
    
    main.train_model()
    return

def load_json(settings_path):
    with open(settings_path,"r",encoding="utf-8") as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/NCL_TEP.json',
                        help='Json file of settings.')
    return parser

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json
    load_arg(args)
    return

if __name__=="__main__":
    main()