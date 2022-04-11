# -*- coding: utf-8 -*-
# @Time    : 2022/4/11 下午3:55
# @Author  : gavin
# @FileName: seq2seq.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
from basemodel import Encoder
from basemodel import Decoder
from basemodel import EncoderDecoder
from torch import nn
import torch


class Seq2SeqEncoder(Encoder):

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        # X:B*T
        X = self.embedding(X)
        # X:B*T*E
        X = X.permute(1, 0, 2)
        # X:T*B*E
        output, state = self.rnn(X)
        # output:T*B*H
        # state: L*B*H
        return output, state


class Seq2SeqDecoder(Decoder):

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        X = self.embedding(X).permute(1, 0, 2)
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), dim=2)
        output, state = self.rnn(X_and_context, state)

        return output, state


if __name__ == "__main__":

    encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    encoder.eval()
    X = torch.zeros((4, 7), dtype=torch.long)

    decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    decoder.eval()
    state = decoder.init_state(encoder(X))
    d_output, d_state = decoder(X, state)

    print(d_output.shape, d_state.shape)
