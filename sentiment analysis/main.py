# -*- coding: utf-8 -*-
# @Time    : 2022/5/11 下午2:22
# @Author  : gavin
# @FileName: train.y.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281

import torch
from torch import nn
from d2l import torch as d2l
from model import BiRNN
from utils import TokenEmbedding


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])


if __name__ == "__main__":
    batch_size = 64
    train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
    embed_size, num_hiddens, num_layers = 100, 100, 2
    devices = d2l.try_all_gpus()
    net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
    net.apply(init_weights)
    glove_embedding = TokenEmbedding('glove.6b.100d')
    embeds = glove_embedding[vocab.idx_to_token]
    net.embedding.weight.data.copy_(embeds)
    net.embedding.weight.requires_grad = False
    lr, num_epochs = 0.01, 5
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction="none")
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)
