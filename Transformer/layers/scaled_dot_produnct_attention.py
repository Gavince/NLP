# -*- coding: utf-8 -*-
# @Time    : 2022/4/18 上午8:15
# @Author  : gavin
# @FileName: scaled dot produnct attention.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
import math

from matplotlib import pyplot as plt
from torch import nn
import torch
from d2l import torch as d2l


def masked_softmax(X, valid_lens):
    """"""
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            # valid_lens: B*D
            valid_lens = valid_lens.reshape(-1)
        # mask val
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)

        return nn.functional.softmax(X.reshape(shape), dim=-1)


class DotProductAttention(nn.Module):
    """点积注意力机制，计算高效"""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        """

        """

        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)

        return torch.bmm(self.dropout(self.attention_weights), values)
