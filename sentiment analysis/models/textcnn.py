# -*- coding: utf-8 -*-
# @Time    : 2022/5/16 上午9:12
# @Author  : gavin
# @FileName: textcnn.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
import torch
from torch import nn


class TextCNN(nn.Module):

    def __init__(self, vocab_size, embed_size, kernel_sizes, num_chanels, **kwargs):
        """

        :param vocab_size:
        :param embed_size:
        :param kernel_sizes:
        :param num_chanels:
        :param kwargs:
        :return:
        """

        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_chanels), 2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.relu = nn.ReLU()
        self.convs = nn.ModuleList()
        for c, k in zip(num_chanels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))

    def forward(self, inputs):
        # 拼接两种不同的的Embedding向量
        # embedding输出:[B, T, embed_size]
        # 拼接：[B, T, embed_size*2]
        embeddings = torch.cat([self.embedding(inputs), self.constant_embedding(inputs)], dim=2)
        # 转换数据维度为：[B, T, embed_size*2] --> [B, embed_size*2, T]
        embeddings = embeddings.permute(0, 2, 1)
        # 计算不同大小的卷积核
        # squeeze: [B, embed_size*2, 1] --> [B, channel_size] --> concat --> [B, channel_size*k_num]
        encoding = torch.cat([torch.squeeze(self.relu(self.pool(conv(embeddings)))) for conv in self.convs], dim=-1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs


if __name__ == "__main__":
    net = TextCNN(100, 20, [2, 3], [10, 10])
    # from torch.utils.tensorboard import SummaryWriter
    #
    # w = SummaryWriter("../log")
    # w.add_graph(net, torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [2, 4, 5, 6, 7, 8, 9, 10]]))
    # print(net)
    # w.close()
