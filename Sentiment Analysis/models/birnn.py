# -*- coding: utf-8 -*-
# @Time    : 2022/5/11 下午2:08
# @Author  : gavin
# @FileName: model.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
import torch
from torch import nn


class BiRNN(nn.Module):

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layer, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layer, bidirectional=True)
        self.decoder = nn.Linear(4*num_hiddens, 2)

    def forward(self, x):

        embedddings = self.embedding(x.T)
        self.encoder.flatten_parameters()
        outputs, _ = self.encoder(embedddings)

        # 拼接起始输入和终止输入
        encodings = torch.cat([outputs[0], outputs[-1]], dim=1)
        outs = self.decoder(encodings)

        return outs


if __name__ == "__main__":
    net = BiRNN(100, 20, 10, 2)
    # from torch.utils.tensorboard import SummaryWriter
    #
    # w = SummaryWriter("../log")
    # w.add_graph(net, torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [2, 4, 5, 6, 7, 8, 9, 10]]))
    # print(net)
    # w.close()

