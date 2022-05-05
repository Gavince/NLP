# -*- coding: utf-8 -*-
# @Time    : 2022/4/26 上午11:07
# @Author  : gavin
# @FileName: bert.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
import torch
from torch import nn
from d2l import torch as d2l
from .mlm import MaskLM
from .nsp import NextSentencePred


class BERTEncoder(nn.Module):
    """BERT编码器"""

    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        # 词编码
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        # 句子编码
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}",
                                 d2l.EncoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape,
                                                  ffn_num_input
                                                  , ffn_num_hiddens, num_heads, dropout, True))
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))

    def forward(self, tokens, segments, valid_lens):

        x = self.token_embedding(tokens) + self.segment_embedding(segments)
        x = x + self.pos_embedding.data[:, :x.shape[1], :]
        for blk in self.blks:
            x = blk(x, valid_lens)

        return x


class BERTModel(nn.Module):
    """BERT模型"""

    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):

        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                                   ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                                   dropout, max_len=max_len, key_size=key_size,
                                   query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens, pred_position=None):

        # encoder_X: [B, T, H]
        encoder_X = self.encoder(tokens, segments, valid_lens)
        if pred_position is not None:
            mlm_Y_hat = self.mlm(encoder_X, pred_position)
        else:
            mlm_Y_hat = None
        # encoder_X: [B, T, H]-->[B, H](只取分类头的输出)
        nsp_Y_hat = self.nsp(self.hidden(encoder_X[:, 0, :]))

        return encoder_X, mlm_Y_hat, nsp_Y_hat


if __name__ == "__main__":
    vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
    norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
    encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                          ffn_num_hiddens, num_heads, num_layers, dropout)
    tokens = torch.randint(0, vocab_size, (2, 8))
    segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
    encoded_X = encoder(tokens, segments, None)
    print(encoded_X.shape)
    print(encoder)
