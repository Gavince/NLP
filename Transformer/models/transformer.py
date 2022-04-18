# -*- coding: utf-8 -*-
# @Time    : 2022/4/18 上午8:18
# @Author  : gavin
# @FileName: transformer.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
from ..layers import AddNorm, DotProductAttention, MultiHeadAttention, PositionalEncoding, PositionWiseFFN
from  d2l import torch as d21
from torch import nn
import torch


class EncoderBlock(nn.Moduled):
    """编码器的基本模块"""

    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape
                 , ffn_num_input, ffn_num_hiddens, num_heads, dropout, use_bias, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout,use_bias)

    def forward(self, X, valid_lens):
        pass