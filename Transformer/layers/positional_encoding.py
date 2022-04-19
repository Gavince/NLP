# -*- coding: utf-8 -*-
# @Time    : 2022/4/18 上午8:14
# @Author  : gavin
# @FileName: positional encoding.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):

    def __init__(self, num_hiddens, dropout, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 位置编码
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) \
            / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        # 奇偶位置编码
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        """
        使用加法对序列进行位置编码
        :param X: B*T*H
        :return:
        """
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        # 防止过拟合出现
        return self.dropout(X)


if __name__ == "__main__":
    max_len = 10000
    num_hiddens = 32
    P = torch.zeros((1, max_len, num_hiddens))
    print(P.shape)
