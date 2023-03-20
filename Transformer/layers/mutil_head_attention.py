# -*- coding: utf-8 -*-
# @Time    : 2022/4/18 上午8:13
# @Author  : gavin
# @FileName: mutilheadattention.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import math
from d2l import torch as d2l


def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences.
    Defined in :numref:`sec_seq2seq_decoder`"""

    max_len = X.size(1)
    # 假定：max_len=3, [[1, 2, 3]] < [[2]]
    # 构建mask矩阵为：[[True, False, False]]
    mask = torch.arange((max_len), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    
    return X


def masked_softmax(X, valid_lens):
    """
    :param X:　X的形状[B, T(q), T(k-v pair)]
    :param valid_lens: 有效长度 [B,]
    :return:
    """
    if valid_lens is None:
        return nn.functional.softmax(X, mdim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            # 扩张到时间序列上
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            # valid_lens: [B, T] --> [B*T]
            valid_lens = valid_lens.reshape(-1)
        # 计算带掩码的softmax
        # X: B*T*T' reshape--> (B*T) * T'
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)

        return nn.functional.softmax(X.reshape(shape), dim=-1)


class DotProductAttention(nn.Module):
    """点积注意力机制，计算高效"""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        """

        :param queries: B * q_size * H
        :param keys: B * k_size * H
        :param values: B * v_size * H
        :param valid_lens: 有效的查询长度
        :return: B*q_size * (k or v)_size
        """
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        # 　注意力权重矩阵　
        self.attention_weights = masked_softmax(scores, valid_lens)

        return torch.bmm(self.dropout(self.attention_weights), values)


def transpose_output(X, num_heads):
    """
    聚合头的输出
    :param X: B*num_heads, q, H/num_heads 每一个线性变换输出为大小为:H/num_heads
    :param num_heads:
    :return: B * q * H 等价与将多个不同头的注意力进行聚合输出
    """
    # [B, num_heads, q_T, H/num_heads]
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    # B, q_T, num_heads, H/num_heads
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
    # [B, num_heads, T, H/num_heads]
    X = X.permute(0, 2, 1, 3)

    return X.reshape(-1, X.shape[2], X.shape[3])


class MultiHeadAttention(nn.Module):

    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        # 使用线性投影层将数值隐射到不同子空间内进行学习
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
            # (num_head*B),eg: [1,2 3] --> [1, 1, 1, 2, 2, 2, 3, 3, 3] repeats=2
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
