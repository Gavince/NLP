# -*- coding: utf-8 -*-
# @Time    : 2022/4/18 上午8:18
# @Author  : gavin
# @FileName: transformer.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
import math
from d2l import torch as d2l
from Transformer.layers import AddNorm, MultiHeadAttention, PositionalEncoding, PositionWiseFFN
from torch import nn
import torch


class EncoderBlock(nn.Module):
    """编码器的基本模块"""

    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape
                 , ffn_num_input, ffn_num_hiddens, num_heads, dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class TransformerEncoder(d2l.Encoder):
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens, norm_shape
                 , ffn_num_input, ffn_num_hiddens, num_heads, num_Layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_Layers):
            self.blks.add_module("block" + str(i),
                                 EncoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape
                                              , ffn_num_input, ffn_num_hiddens, num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):

        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weighs = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weighs[i] = blk.attention.attention.attention_weights

        return X


class DecoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape
                 , ffn_num_input, ffn_num_hiddens, num_heads, dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.add_norm1 = AddNorm(norm_shape, dropout)

        self.attention2 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.add_norm2 = AddNorm(norm_shape, dropout)

        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.add_norm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练阶段，输出序列的所有词元都在同⼀时间处理，
        # 因此state[2][self.i]初始化为None。

        # 预测阶段，输出序列是通过词元⼀个接着⼀个解码的，
        # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表⽰
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), dim=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_step, _ = X.shape
            # dec_valid_len: (batch_size, num_steps)
            # 每一行为[1, 2, 3.....num_steps]
            # B*num_steps
            dec_valid_lens = torch.arange(1, num_step + 1, device=X.device).repeat(batch_size, 1)
        else:
            # 直接计算相应的注意力权重
            dec_valid_lens = None
        # 自注意力
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.add_norm1(X, X2)
        # 编码器-解码器
        # enc_outputs: bathch_size*num_steps*num_hiddens
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.add_norm2(Y, Y2)

        return self.add_norm3(Z, self.ffn(Z)), state


class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens, norm_shape
                 , ffn_num_input, ffn_num_hiddens, num_heads, num_Layers, dropout, use_bias=False, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layer = num_Layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encodding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_Layers):
            self.blks.add_module("block" + str(i), DecoderBlock(key_size, query_size
                                                                , value_size, num_hiddens, norm_shape, ffn_num_input,
                                                                ffn_num_hiddens, num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        """
        初始化解码器的状态
        :param enc_outputs:
        :param enc_valid_lens:
        :param args:
        :return:
        """
        return [enc_outputs, enc_valid_lens, [None] * self.num_layer]

    def forward(self, X, state):
        X = self.pos_encodding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attentin_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            # 解码器的自注意力权重
            X, state = blk(X, state)
            self._attentin_weights[0][i] = blk.attention1.attention.attention_weights
            # "编码器-解码器"自注意力权重
            self._attentin_weights[1][i] = blk.attention2.attention.attention_weights
        # state 存
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attentin_weights


if __name__ == "__main__":
    X = torch.ones((2, 100, 24))
    valid_lens = torch.tensor([3, 2])
    encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
    encoder_blk.eval()
    print(encoder_blk(X, valid_lens).shape)
    encoder = TransformerEncoder(
        200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5)
    encoder.eval()
    print(encoder(torch.ones((2, 100), dtype=torch.long), valid_lens).shape)
