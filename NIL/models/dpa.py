# -*- coding: utf-8 -*-
# @Time    : 2022/5/17 下午4:45
# @Author  : gavin
# @FileName: nil.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
import torch
from torch import nn
from torch.nn import functional as F


def mlp(num_inputs, num_hiddens, flatten):
    net = []
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_inputs, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))

    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_inputs, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))

    return nn.Sequential(*net)


class Attend(nn.Module):
    """対齐"""

    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B):
        """
        :param A:[B, T_a, H]
        :param B:[B, T_b, H]
        :return:
        """
        # f_A: [B, T_a, H']
        f_A = self.f(A)
        f_B = self.f(B)
        # e的形状：（批量⼤⼩，序列A的词元数，序列B的词元数）
        e = torch.bmm(f_A, f_B.permute((0, 2, 1)))
        # beta的形状：（批量⼤⼩，序列A的词元数， embed_size），
        # 意味着序列B被软对⻬到序列A的每个词元(beta的第1个维度)
        beta = torch.bmm(F.softmax(e, dim=-1), B)
        # beta的形状：（批量⼤⼩，序列B的词元数， embed_size），
        # 意味着序列A被软对⻬到序列B的每个词元(alpha的第1个维度)
        alpha = torch.bmm(F.softmax(e.permute(0, 2, 1), dim=1), A)

        return beta, alpha


class Compare(nn.Module):

    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B, alpha, beta):
        V_A = self.g(torch.cat([A, beta], dim=2))
        V_B = self.g(torch.cat([B, alpha], dim=2))
        return V_A, V_B


class Aggregate(nn.Module):

    def __init__(self, num_inputs, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_inputs, num_hiddens, flatten=True)
        self.linear = nn.Linear(num_hiddens, num_outputs)

    def forward(self, V_A, V_B):
        V_A = V_A.sum(dim=1)
        V_B = V_B.sum(dim=1)
        y_hat = self.linear(self.h(torch.cat([V_A, V_B], dim=1)))

        return y_hat


class DecomposableAttention(nn.Module):

    def __init__(self, vocab, embed_size, num_hiddens, num_inputs_attend=100
                 , num_inputs_compare=200, num_inputs_agg=400, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(num_inputs_attend, num_hiddens)
        self.compare = Compare(num_inputs_compare, num_hiddens)
        self.aggregate = Aggregate(num_inputs_agg, num_hiddens, num_outputs=3)

    def forward(self, X):
        premises, hypotheses = X
        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        y_hat = self.aggregate(V_A, V_B)

        return y_hat


if __name__ == "__main__":
    net = DecomposableAttention(torch.arange(1, 1000), 100, 64)
    print(net)
