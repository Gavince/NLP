# -*- coding: utf-8 -*-
# @Time    : 2022/4/10 上午11:29
# @Author  : gavin
# @FileName: rnn.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
import torch
from torch.nn import functional as F
from d2l import torch as d2l
from torch import nn


def get_params_rnn(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_rnn_state(batch_size, num_hiddens, device):
    """

    :param batch_size:
    :param num_hiddens:
    :param device:
    :return:
    """

    return (torch.zeros((batch_size, num_hiddens), device=device),)


def rnn(inputs, state, params):
    """
    :param inputs: T*B*V
    :param state: B*H
    :param params:
    :return:
    """
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


class RNNModelScratch:

    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        """

        :param X: B*T
        :param state:B*V
        :return:
        """
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


if __name__ == "__main__":
    device = d2l.try_gpu(10)
    num_hiddens = 512
    X = torch.arange(10).reshape((2, 5)).to(device)

    net = RNNModelScratch(28, num_hiddens, device, get_params_rnn,
                          init_rnn_state, rnn)

    state = net.begin_state(X.shape[0], d2l.try_gpu(10))
    Y, new_state = net(X, state)
    print(Y.shape, len(new_state), new_state[0].shape)
