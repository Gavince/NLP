# -*- coding: utf-8 -*-
# @Time    : 2022/5/20 下午5:27
# @Author  : gavin
# @FileName: bert_classifier.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
from torch import nn


class BERTClassifier(nn.Module):

    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.Linear(256, 3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoder_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoder_X[:, 0, :]))