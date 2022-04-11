# -*- coding: utf-8 -*-
# @Time    : 2022/4/10 下午2:38
# @Author  : gavin
# @FileName: basernn.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
import torch
from torch.nn import functional as F
from d2l import torch as d2l
from torch import nn


class RNNModel(nn.Module):

    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.hidden_size = self.rnn.hidden_size
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.hidden_size, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.hidden_size * 2, self.vocab_size)

    def forward(self, input, state):
        X = F.one_hot(input.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 构造T*B个多分类的结果
        output = self.linear(Y.reshape((-1, Y.shape[-1])))

        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            return torch.zeros((self.num_directions * self.rnn.num_layers, batch_size
                                , self.hidden_size), device=device)
        else:
            # LSTM
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))


if __name__ == "__main___":
    device = d2l.try_gpu()
    num_hiddens = 512
    X = torch.arange(10).reshape((2, 5)).to(device)
    rnn_layer = nn.RNN(28, num_hiddens)
    net = RNNModel(rnn_layer, 28).to(device)
    state = net.begin_state(device, 2)
    Y, new_state = net(X, state)
    print(Y.shape, len(new_state), new_state[0].shape)
