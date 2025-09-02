# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 20:49:15 2022

@author: Lenovo
"""

from torch import nn
from matplotlib import pyplot
from matplotlib import cm
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import TimeSeriesKMeans

import torch
import random
import numpy
import numpy as np
import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd

def plot_shapelet():
    return

def normalize_label(label):
    '''
       将标签变为从0开始的自然数
    '''
    label_tensor = torch.tensor(label)
    label_tensor = label_tensor.float()
    for i in list(set(label)):
        label_tensor = torch.where(label_tensor==i,torch.Tensor([list(set(label)).index(i)]).expand(label_tensor.shape[0]),label_tensor)
    label = label_tensor.numpy()
    return label

def plot_sub_fig(X, Y=0, label0='label'):
    if type(Y)==int:
        plt.plot(X)
    else:
        if len(X)!=len(Y):
            plt.plot(range(len(Y)),Y)
        else:
            plt.plot(X,Y)
            plt.title(label0)
    plt.savefig('results/'+label0+'.png')
    plt.show()
    plt.close()
    return

def normalize_y(y_train,y_test):
     #当标签不是0,1,2这种，而是1,2这种时
    if min(y_train)!=0:
        y_train = y_train-min(y_train)
        y_test = y_test-min(y_test)
    #当数据标签是0,2这种时
    dict_ = {} #新建一个字典,key为更改前的y,value为更改后的y
    if max(y_train)-min(y_train) >= len(set(y_train)):
        for i in range(len(set(y_train))):
            dict_[sorted(list(set(y_train)))[i]]=i
        asign = lambda t: dict_[t] 
        y_train = list(map(asign, y_train))
        y_test = list(map(asign, y_test))
        y_train = np.array(y_train)
        y_test = np.array(y_test)
    return y_train,y_test

def normalize_tensor(X):
    '''X维度为：[num,1,length]'''
    #print(X.shape,'X.shape*****')
    X = X.numpy()
    dim0, dim1, dim2, dim3 = X.shape
    for i in range(dim0):
        for j in range(dim2):
            element = X[i][0][j] 
            #element = element.numpy()
            X[i][0][j],_ = normalize_standard(element)
    X = torch.from_numpy(X)
    return X

def plot_initia(shapelets):
    fig = pyplot.figure(facecolor='white')
    fig.set_size_inches(20, 8)

    for j in range(shapelets.shape[1]):
        fig_ax1 = fig.add_subplot(4,8,1+j)
        fig_ax1.plot(shapelets[0,j,:].detach().numpy(),color='black', alpha=0.5)
    return 

def preprocessing_standard0(L):
    """
       preprocessing_standard: 对一行数据进行数据标准化,L[i]=(L[i]-L.min)/(L.max-L.min)
    """
    datamax=max(L)
    datamin=min(L)
    L1=[]
    for index,row in enumerate(L):
        if datamax-datamin!=0:
            m=(row-datamin)/float((datamax-datamin))
            L1.append(round(m,4))
        else:
            L1.append(0)
   #        matlabshow(row,index=str(index)+'_')    
    return L1

def preprocess0(dataframe):
    dataframe_new = pd.DataFrame(columns=dataframe.columns)
    for i in dataframe.columns:
        dataframe_new[i] = preprocessing_standard0(dataframe[i])
        #dataframe_new[i]=pywt_pro_0(list(dataframe_new[i]))
    return dataframe_new

def normalize_standard(X, scaler=None):
    shape = X.shape
    data_flat = X.flatten()
    if scaler is None:
        scaler = StandardScaler()
        data_transformed = scaler.fit_transform(data_flat.reshape(numpy.product(shape), 1)).reshape(shape)
    else:
        data_transformed = scaler.transform(data_flat.reshape(numpy.product(shape), 1)).reshape(shape)
    return data_transformed, scaler

def normalize_data(X, scaler=None):
    if scaler is None:
        X, scaler = normalize_standard(X)
    else:
        X, scaler = normalize_standard(X, scaler)
    return X, scaler

def sample_ts_segments(X, shapelets_size, n_segments=10000):
    """
    Sample time series segments for k-Means.
    """
    n_ts, n_channels, len_ts = X.shape
    samples_i = random.choices(range(n_ts), k=n_segments)
    
    '''因为n_channels为24，但是只想要一个通道的shapelet，所以改为了1'''
    segments = numpy.empty((n_segments, 1, shapelets_size))
    for i, k in enumerate(samples_i):
        samples_dim = random.choices(range(n_channels), k=1)
        s = random.randint(0, len_ts - shapelets_size)\
        #s=15
        segments[i] = X[k, samples_dim, s:s+shapelets_size]
    return segments

def get_weights_via_kmeans(X, shapelets_size, num_shapelets, n_segments=10000):
    """
    Get weights via k-Means for a block of shapelets.
    """
    segments = sample_ts_segments(X, shapelets_size, n_segments).transpose(0, 2, 1)
    #print(segments.shape,'segments.shape')
    k_means = TimeSeriesKMeans(n_clusters=num_shapelets, metric="euclidean", max_iter=50).fit(segments)
    clusters = k_means.cluster_centers_.transpose(0, 2, 1)
    #print(clusters.shape,'clusters.shape')
    return clusters

class ShapeletsDistanceLoss(nn.Module):
    """
    """
    def __init__(self, dist_measure='euclidean', k=6):
        super(ShapeletsDistanceLoss, self).__init__()
        if not dist_measure == 'euclidean' and not dist_measure == 'cosine':
            raise ValueError("Parameter 'dist_measure' must be either of 'euclidean' or 'cosine'.")
        if not isinstance(k, int):
            raise ValueError("Parameter 'k' must be an integer.")
        self.dist_measure = dist_measure
        self.k = k

    def forward(self, x):
        """
        """
        y_top, y_topi = torch.topk(x.clamp(1e-8), self.k, largest=False if self.dist_measure == 'euclidean' else True,
                                   sorted=False, dim=0)
        # avoid compiler warning
        y_loss = None
        if self.dist_measure == 'euclidean':
            y_loss = torch.mean(y_top)
        elif self.dist_measure == 'cosine':
            y_loss = torch.mean(1 - y_top)
        return y_loss
    
class ShapeletsSimilarityLoss(nn.Module):
    """
    Calculates the cosine similarity of each block of shapelets and averages over the blocks.
    ----------
    """
    def __init__(self):
        super(ShapeletsSimilarityLoss, self).__init__()

    def cosine_distance(self, x1, x2=None, eps=1e-8):
        """
    
        """
        x2 = x1 if x2 is None else x2
        # unfold time series to emulate sliding window
        x1 = x1.unfold(2, x2.shape[2], 1).contiguous()
        x1 = x1.transpose(0, 1)
        # normalize with l2 norm
        x1 = x1 / x1.norm(p=2, dim=3, keepdim=True).clamp(min=1e-8)
        x2 = x2 / x2.norm(p=2, dim=2, keepdim=True).clamp(min=1e-8)

        # calculate cosine similarity via dot product on already normalized ts and shapelets
        x1 = torch.matmul(x1, x2.transpose(1, 2))
        # add up the distances of the channels in case of
        # multivariate time series
        # Corresponds to the approach 1 and 3 here: https://stats.stackexchange.com/questions/184977/multivariate-time-series-euclidean-distance
        # and average over dims to keep range between 0 and 1
        n_dims = x1.shape[1]
        x1 = torch.sum(x1, dim=1) / n_dims
        return x1

    def forward(self, shapelet_blocks):
        """
        Calculate the loss as the sum of the averaged cosine similarity of the shapelets in between each block.
        @param shapelet_blocks: a list of the weights (as torch parameters) of the shapelet blocks
        @type shapelet_blocks: list of torch.parameter(tensor(float))
        @return: the computed loss
        @rtype: float
        """
        losses = 0.
        for block in shapelet_blocks:
            shapelets = block[1]
            shapelets.retain_grad()
            sim = self.cosine_distance(shapelets, shapelets)
            losses += torch.mean(sim)
        return losses
    
def torch_dist_ts_shapelet(ts, shapelet, cuda=False):
    """
    Calculate euclidean distance of shapelet to a time series via PyTorch and returns the distance along with the position in the time series.
    """
    if not isinstance(ts, torch.Tensor):
        ts = torch.tensor(ts, dtype=torch.float)
    if not isinstance(shapelet, torch.Tensor):
        shapelet = torch.tensor(shapelet, dtype=torch.float)
    if cuda:
        ts = ts.cuda()
        shapelet = shapelet.cuda()
    shapelet=shapelet[:1,:]
    shapelet = torch.unsqueeze(shapelet, 1)
    # unfold time series to emulate sliding window
    ts = ts.unfold(1, shapelet.shape[2], 1)
    # calculate euclidean distance
    dists = torch.cdist(ts, shapelet, p=2)
    #print(dists.shape,'dists.shape')
    if dists.shape[0]>1:
        #阳极电流数据是多维的，min_single_dim是子序列与单个序列的匹配位置，min_total_dim是子序列最匹配的维度
        min_num,min_single_dim = torch.min(dists, dim=1)
        d_min, min_total_dim = torch.min(min_num, 0)
        return (min_single_dim[min_total_dim.item()].item(), min_total_dim.item())
    else:
        #公共数据
        dists = torch.sum(dists, dim=0)
        d_min,d_argmin = torch.min(dists, dim=0)
        return (d_min.item(), d_argmin.item())

def lead_pad_shapelet(shapelet, pos):
    """
    Adding leading NaN values to shapelet to plot it on a time series at the best matching position.
    """
    pad = numpy.empty(pos)
    pad[:] = numpy.NaN
    padded_shapelet = numpy.concatenate([pad, shapelet])
    return padded_shapelet

def record_shapelet_value(ucr_dataset_name,shapelets,X_test,pos,i,j):
    '''将shapelet的值保存到文档里面'''
    path='results/shapelet_value/'+str(ucr_dataset_name)+str(X_test.shape[0])+'/'
    if not os.path.exists(path):
        os.mkdir(path)
    path_shapelet_value=path='results/shapelet_value/'+str(ucr_dataset_name)+str(X_test.shape[0])+'/'+'shapelet'+str(j)+'/'
    if not os.path.exists(path_shapelet_value):
        os.mkdir(path_shapelet_value)
    excel_shapelet=lead_pad_shapelet(shapelets[j, 0], pos)
    if X_test.shape[1]>1:
        excel_time_series=X_test[i,pos]
    else:
        excel_time_series=X_test[i]
    excel_shapelet=pd.DataFrame(excel_shapelet)
    excel_time_series=pd.DataFrame(excel_time_series)
    excel_shapelet.to_excel(path_shapelet_value+'shapelet.xlsx')
    excel_time_series.to_excel(path_shapelet_value+'time_series.xlsx')
    return

def plot_sub(i,j,fig,shapelets,X_test,record_data_plot,test_y,ucr_dataset_name):
    '''
    i:num of sample
    j:num of sub_graph
    shapelets:[num of shapelets,1, len_of_shapelets]
    X_test:[num of test,1, len of test]
    '''
    font = {'family': 'Times New Roman',
        'style': 'normal',
        'stretch': 1000,
        'weight': 'bold',
        }
    fig_ax1 = fig.add_subplot(4,int(shapelets.shape[0]/4)+1,j+1)
    plt.subplots_adjust(left=None, bottom=0.05, right=None, top=None, wspace=0.3, hspace=0.5)#wspace 子图横向间距， hspace 代表子图间的纵向距离，left 代表位于图像不同位置
    fig_ax1.text(0.01,0.01,'',fontdict=font)
    fig_ax1.set_title("shapelet"+str(j+1),fontproperties="Times New Roman",)
    if X_test.shape[1]>1:
        _, pos = torch_dist_ts_shapelet(X_test[i], shapelets[j])
        fig_ax1.plot(X_test[i, pos], color='black', alpha=0.02, )
        fig_ax1.plot(lead_pad_shapelet(shapelets[j, 0], _), color='#F03613', alpha=0.02)
        record_shapelet_value(ucr_dataset_name,shapelets,X_test,pos,i,j)
    else:
        fig_ax1.plot(X_test[i, 0], color='black', alpha=0.5)
        _, pos = torch_dist_ts_shapelet(X_test[i], shapelets[j])
        fig_ax1.plot(lead_pad_shapelet(shapelets[j, 0], pos), color='#F03613', alpha=0.5)
        record_shapelet_value(ucr_dataset_name,shapelets,X_test,pos,i,j)
    record_data_plot['fig'+str(j)]['x']=X_test[i, 0]
    record_data_plot['fig'+str(j)]['s']=shapelets[j, 0]
    record_data_plot['fig'+str(j)]['class']=test_y[i]
    record_data_plot['fig'+str(j)]['dim']=pos
    return record_data_plot

def featrue_map(shapelet_transform, y_test, weights, ucr_dataset_name, X_test, shapelet_num):
    '''设置全局字体'''
    pyplot.rcParams['font.sans-serif']='Times New Roman'
    pyplot.rcParams['font.weight']='bold'
    pyplot.rcParams['font.size']=14
    pyplot.rc('xtick',labelsize=10)
    pyplot.rc('ytick',labelsize=10)
    
    fig = pyplot.figure(facecolor='white')
    #fig.set_size_inches(20, 8)
    gs = gridspec.GridSpec(2, 2)
    fig_ax3 = fig.add_subplot(gs[:, :])
    #font0 = FontProperties(family='serif',weight='bold',size=14)
    #fig_ax3.set_title("The decision boundaries learned by the model to separate the two classes.", fontproperties=font0)
    color = {-1:'#00FF00',0: '#F03613', 1: '#7BD4CC', 2: '#00281F', 3: '#BEA42E',4:'#FFC0CB',5:'#FFF0F5',6:'#FF69B4'}
             
    dist_s1=shapelet_transform[:,shapelet_num]
    dist_s2=shapelet_transform[:,shapelet_num+1]
    fig_ax3.scatter(dist_s1, dist_s2, color=[color[l] for l in y_test])
    
    # Create a meshgrid of the decision boundaries
    xmin = numpy.min(shapelet_transform[:, shapelet_num]) - 0.1
    xmax = numpy.max(shapelet_transform[:, shapelet_num]) + 0.1
    ymin = numpy.min(shapelet_transform[:, shapelet_num+1]) - 0.1
    ymax = numpy.max(shapelet_transform[:, shapelet_num+1]) + 0.1
    xx, yy = numpy.meshgrid(numpy.arange(xmin, xmax, (xmax - xmin)/200),
                            numpy.arange(ymin, ymax, (ymax - ymin)/200))
    Z = []
    num_class=len(weights)
    for x, y in numpy.c_[xx.ravel(), yy.ravel()]:
        Z.append(numpy.argmax([weights[i][0]*x + weights[i][1]*y
                               for i in range(num_class)]))
   # Z = numpy.array(Z).reshape(xx.shape)
    #fig_ax3.contourf(xx, yy, Z / 3, cmap=viridis, alpha=0.25)
    fig_ax3.set_xlabel("shapelet"+str(shapelet_num))
    fig_ax3.set_ylabel("shapelet"+str(shapelet_num+1))
    fig_ax3.tick_params(labelsize=13)
    
    path='results/shapelets_plots/'+str(ucr_dataset_name)+'shapelet'+str(shapelet_num)+'_'+str(shapelet_num+1)+'feature_map'+'/'
    if not os.path.exists(path):
        os.mkdir(path)
    pyplot.savefig(path+'.pdf',format='pdf', facecolor=fig.get_facecolor(), bbox_inches="tight")
    
    #pyplot.savefig(path+'.png', facecolor=fig.get_facecolor(), bbox_inches="tight")
    return 

def plot_shapelets(X_test, shapelets, y_test, shapelet_transform, ucr_dataset_name):
    fig = pyplot.figure(facecolor='white')
    fig.set_size_inches(20, 8)
    dist={}
    nums_shapelets=shapelet_transform.shape[1]
    for i in range(nums_shapelets):
        dist[i]=shapelet_transform[:, i]
   # gs = gridspec.GridSpec(12, 8)
    #fig_ax1 = fig.add_subplot(gs[0:3, :4])
    record_data_plot={}
    for i in range(nums_shapelets):
        record_data_plot['fig'+str(i)]={}
   # fig_ax1.set_title("top of its 1 best matching time series.")
    for j in range(nums_shapelets):
        for i in numpy.argsort(dist[j])[:1]:
            record_data_plot = plot_sub(i,j,fig,shapelets,X_test,record_data_plot,y_test,ucr_dataset_name)
#    
    caption = """Shapelets learned for the pot volt dataset plotted on top of the best matching time series."""
    pyplot.figtext(0.5, -0.02, caption, wrap=True, horizontalalignment='center', fontsize=20, family='Times New Roman')
    path='results/shapelets_plots/'+str(ucr_dataset_name)+str(X_test.shape[0])+'/'
    if not os.path.exists(path):
        os.mkdir(path)
    pyplot.savefig(path+'.pdf', format='pdf', facecolor=fig.get_facecolor(), bbox_inches="tight")
    #pyplot.savefig(path+'.png', facecolor=fig.get_facecolor(), bbox_inches="tight")
    #pyplot.show()
    #画所有shapelets映射的特征图
    #for shapelet_num in range(0,len(shapelet_transform[1])-1,2):
    #     featrue_map(shapelet_transform, y_test, weights, ucr_dataset_name, X_test, shapelet_num)
    return record_data_plot

