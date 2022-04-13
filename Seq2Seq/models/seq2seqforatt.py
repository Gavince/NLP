# -*- coding: utf-8 -*-
# @Time    : 2022/4/13 下午3:32
# @Author  : gavin
# @FileName: seq2seqforatt.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
from matplotlib import pyplot as plt
from torch import nn
import torch
from d2l import torch as d2l


def masked_softmax(X, valid_lens):
    """jisua"""
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


class AdditiveAttention(nn.Module):
    """针对query与key长度不一致,使用加性注意力机制"""

    def __init__(self, key_size, query_size, num_hiddens, dropout=0., **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.W_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # broadcast
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # score:  B*Q*K
        score = self.W_v(features).squeeze(-1)
        # mask的softmaxt
        self.attention_weights = masked_softmax(score, valid_lens)

        return torch.bmm(self.dropout(self.attention_weights), values)


if __name__ == "__main__":
    queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
    # values的⼩批量，两个值矩阵是相同的
    print("ds")
    print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3])))
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(
        2, 1, 1)
    valid_lens = torch.tensor([2, 6])
    attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8,
                                  dropout=0.1)
    attention.eval()
    print(attention(queries, keys, values, valid_lens))
    d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                      xlabel='Keys', ylabel='Queries')
    plt.show()
