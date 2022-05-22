# -*- coding: utf-8 -*-
# @Time    : 2022/4/30 上午8:56
# @Author  : gavin
# @FileName: test.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
from d2l import torch as d2l
import torch
from models.bert import BERTModel


def get_bert_encoding(net, token_a, token_b, vocab, device):
    tokens, segments = d2l.get_tokens_and_segments(token_a, token_b)
    token_ids = torch.tensor(vocab[tokens], device=device).unqueeze(0)
    segments = torch.tensor(segments, device=device).unsqueeze(0)
    valid_len = torch.tensor(len(tokens), device=device).unsqueeze(0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)

    return encoded_X


if __name__ == "__main__":
    batch_size, max_len = 64, 64
    train_iter, vocab = d2l.load_data_wiki(batch_size, max_len)
    net = BERTModel(len(vocab), num_hiddens=128, norm_shape=[128],
                    ffn_num_input=128, ffn_num_hiddens=256, num_heads=2,
                    num_layers=2, dropout=0.2, key_size=128, query_size=128,
                    value_size=128, hid_in_features=128, mlm_in_features=128,
                    nsp_in_features=128
                    )
    net.load_state_dict(torch.load("./checkpoint/bert.pth"))
    devices = d2l.try_gpu(0)
    # 输入数据
    tokens_a = ['a', 'crane', 'is', 'flying']
    encoded_text = get_bert_encoding(net, tokens_a)
    # 词元： '<cls>','a','crane','is','flying','<sep>'
    encoded_text_cls = encoded_text[:, 0, :]
    encoded_text_crane = encoded_text[:, 2, :]
    encoded_text.shape, encoded_text_cls.shape, encoded_text_crane[0][:3]
