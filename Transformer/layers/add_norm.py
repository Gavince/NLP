# -*- coding: utf-8 -*-
# @Time    : 2022/4/18 上午9:08
# @Author  : gavin
# @FileName: add norm.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
from torch import nn
import torch


class AddNorm(nn.Module):

    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(X + self.dropout(Y))


if __name__ == "__main__":
    add_norm = AddNorm([3, 4], 0.5)
    add_norm.eval()
    print(add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4))))
