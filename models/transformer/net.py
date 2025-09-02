# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 09:05:05 2023

@author: WXX
"""
from torch import nn

import torch
import numpy as np
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output

    
class MultiHeadAttentionNewClass(nn.Module):
    def __init__(self):
        super(MultiHeadAttentionNewClass, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.Relu = nn.ReLU()
        self.output_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

    def forward(self, cos_similarities):
        # 计算向量1与向量组中每个向量的余弦相似度
        cos_similarities = cos_similarities.unsqueeze(-1)
        concat_similarities = torch.cat([cos_similarities, cos_similarities,cos_similarities], dim=-1)  # 在第二维上拼接
        
        # 将LSTM输出的hidden state变换为适合一维卷积的形状
        
        x=self.Relu(self.conv1(concat_similarities.unsqueeze(1)))
        x = torch.max_pool2d(x, 2)
       #$ x=self.Relu(self.conv1(x))
        #x = torch.max_pool1d(x, 2)
        x = torch.relu(self.conv2(x))
        #print(x.shape,'*****************x')
        x = self.global_avg_pool(x)
        #print(x.shape,'*****************x')
        x = self.output_conv(x)
        # 使用输出层计算输出
        return self.sigmoid(x)
