# -*- coding: utf-8 -*-
# @Time    : 2022/5/17 下午4:46
# @Author  : gavin
# @FileName: train.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
from d2l import torch as d2l
from models.dpa import DecomposableAttention
import torch
from torch import nn

if __name__ == "__main__":
    batch_size, num_steps = 256, 50
    train_iter, test_iter, vocab = d2l.load_data_snli(batch_size, num_steps)
    embed_size, num_hiddens, devices = 100, 200, d2l.try_all_gpus()
    net = DecomposableAttention(vocab, embed_size, num_hiddens)
    glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
    embeds = glove_embedding[vocab.idx_to_token]
    net.embedding.weight.data.copy_(embeds)
    lr, num_epochs = 0.001, 4
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction="none")
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)
