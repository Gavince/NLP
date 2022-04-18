# -*- coding: utf-8 -*-
# @Time    : 2022/4/18 上午8:59
# @Author  : gavin
# @FileName: positionwise ffn.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
import torch
from torch import nn


class PositionWiseFFN(nn.Module):

    def __init__(self, ffn_num_inputs, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__()

        self.dense1 = nn.Linear(ffn_num_inputs, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


if __name__ == "__main__":
    ffn = PositionWiseFFN(4, 4, 8)
    ffn.eval()
    print(ffn(torch.ones((2, 3, 4)))[0])
