# -*- coding: utf-8 -*-
# @Time    : 2022/4/11 下午3:55
# @Author  : gavin
# @FileName: seq2seq.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
import torch
from torch import nn


class Encoder(nn.Module):

    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError


class Decoder(nn.Module):

    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class EncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)

        return self.decoder(dec_X, dec_state)


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
        # output: T*B*H
        output = self.dense(output).permute(1, 0, 2)
        # output: B*T*V
        return output, state


if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter

    W = SummaryWriter("../logs/graph")
    encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    encoder.eval()
    X = torch.zeros((4, 7), dtype=torch.long)

    decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    decoder.eval()
    state = decoder.init_state(encoder(X))
    d_output, d_state = decoder(X, state)
    net = EncoderDecoder(encoder, decoder)
    args = (X, X)
    W.add_graph(net, input_to_model=args)
    Y_hat, _ = net(X, X)
    print(Y_hat.shape, d_state.shape)
