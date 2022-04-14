# -*- coding: utf-8 -*-
# @Time    : 2022/4/13 下午3:32
# @Author  : gavin
# @FileName: seq2seqforatt.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
import math

from matplotlib import pyplot as plt
from torch import nn
import torch
from d2l import torch as d2l


def masked_softmax(X, valid_lens):
    """jisua"""
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            # valid_lens: B*D
            valid_lens = valid_lens.reshape(-1)
        # mask val
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)

        return nn.functional.softmax(X.reshape(shape), dim=-1)


class AdditiveAttention(nn.Module):
    """针对query与key长度不一致,使用加性注意力机制"""

    def __init__(self, key_size, query_size, num_hiddens, dropout=0., **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.W_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        """

        :param queries:B * queries_size * H
        :param keys: B * (k-v) * H
        :param values: B * (k-V) * H
        :param valid_lens:
        :return: B*(k-q)*值的维度
        """

        queries, keys = self.W_q(queries), self.W_k(keys)
        # broadcast
        # (B * (q) * 1 * H) (B * 1 * (k-v) * H)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        # feature: B * q * (k-v) * H
        features = torch.tanh(features)
        # score:  B * q * (k - v)
        score = self.W_v(features).squeeze(-1)
        # mask的softmaxt
        # B * q * (k - v)
        self.attention_weights = masked_softmax(score, valid_lens)
        # B * q * H
        return torch.bmm(self.dropout(self.attention_weights), values)


class DotProductAttention(nn.Module):
    """点积注意力机制，计算高效"""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        """

        """

        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)

        return torch.bmm(self.dropout(self.attention_weights), values)


class AttentionDecoder(d2l.Decoder):
    """带有注意力机制的解码器"""

    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    def attention_weights(self):
        raise NotImplementedError


class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = AdditiveAttention(num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        """

        :param enc_outputs: B * T * H
        :param enc_valid_lens:
        :param args:
        :return:
        """

        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        enc_outputs, hidden_state, enc_valid_lens = state
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        # 按照时序进行输入
        for x in X:
            # x: B * H
            # query: B * 1 * K
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # context: B*1*H
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)
            # context + embed: B*1*(E+H)
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # out:T*B*H, hidden_state: (layer, bid)*B*H
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # outputs: T*B*V
        outputs = self.dense(torch.cat(outputs, dim=0))
        # B*T*V, state
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights


if __name__ == "__main__":
    # queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
    # # values的⼩批量，两个值矩阵是相同的
    # print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3])))
    # values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(
    #     2, 1, 1)
    # valid_lens = torch.tensor([2, 6])
    # attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8,
    #                               dropout=0.1)
    # attention.eval()
    # print(attention(queries, keys, values, valid_lens))
    # d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
    #                   xlabel='Keys', ylabel='Queries')
    # plt.show()
    encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                                 num_layers=2)
    encoder.eval()
    decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                                      num_layers=2)
    decoder.eval()
    X = torch.zeros((4, 7), dtype=torch.long)  # (batch_size,num_steps)
    state = decoder.init_state(encoder(X), None)
    output, state = decoder(X, state)
    print(output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape)
