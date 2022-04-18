# -*- coding: utf-8 -*-
# @Time    : 2022/4/18 上午8:13
# @Author  : gavin
# @FileName: mutilheadattention.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch
import math
from d2l import torch as d2l
from seq2seqforatt import DotProductAttention


def transpose_output(X, num_heads):
    """
    聚合头的输出
    :param X: B*num_heads, q, H/num_heads 每一个线性变换输出为大小为:H/num_heads
    :param num_heads:
    :return: B * q * H 等价与将多个不同头的注意力进行聚合输出
    """
    # B, num_heads, q, H/num_heads
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    # B, q, num_heads, H/num_heads
    X = X.permute(0, 2, 1, 3)

    return X.reshape(X.shape[0], X.shape[1], -1)


def transpose_qkv(X, num_heads):
    """
    实现维度转换，以计算不同形式下的仿射变换
    :param X: 输⼊X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)/输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，num_hiddens/num_heads)
    :param num_heads:
    :return:输出形状：B*num_heads, q, H/num_heads
    """
    # B*Q*num_heads, H/num_heads
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)

    return X.reshape(-1, X.shape[2], X.shape[3])


class MultiHeadAttention(nn.Module):

    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        """
        计算多头注意力机制
        :param queries: B*Q*H
        :param keys: B*(k-v)*H
        :param values: B*(k-v)*H
        :param valid_lens: B,
        :return:B*q*num_heads
        """
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        # B*num_heads, q or (k-v), H/num_heads
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
        # B*num_heads, q, H/num_heads
        output = self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_output(output, self.num_heads)

        return self.W_o(output_concat)


if __name__ == "__main__":

    writer = SummaryWriter(log_dir="../logs/graph", comment="MultiAttention", filename_suffix="multiHeadlAttention")
    num_hiddens, num_heads = 100, 5
    attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, dropout=0.5)
    print(attention)
    attention.eval()
    batch_size, num_queries = 2, 4
    num_kvpairs, valid_lens = 6, torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))
    Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
    print(attention(X, Y, Y, valid_lens).shape)
    args = X, Y, Y, valid_lens
    writer.add_graph(attention, input_to_model=args)
    writer.close()