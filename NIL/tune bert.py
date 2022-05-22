# -*- coding: utf-8 -*-
# @Time    : 2022/5/21 下午5:39
# @Author  : gavin
# @FileName: tune bert.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
from d2l import torch as d2l
import torch
import os
import json
from models.bert_classifier import BERTClassifier
from datasets.bert_dataset import SNILBERTDataset


def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_layers, dropout, max_len, devices):
    # data_dir = d2l.download_extract(pretrained_model)
    # 定义空词表以加载预定义词表
    data_dir = "../data/bert.small.torch"
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(vocab.idx_to_token)}

    bert = d2l.BERTModel(len(vocab), num_hiddens, norm_shape=[256],
                         ffn_num_input=256, ffn_num_hiddens=ffn_num_hiddens,
                         num_heads=4, num_layers=2, dropout=0.2,
                         max_len=max_len, key_size=256, query_size=256,
                         value_size=256, hid_in_features=256,
                         mlm_in_features=256, nsp_in_features=256)

    # 加载预训练BERT参数
    bert.load_state_dict(torch.load(os.path.join(data_dir,
                                                 'pretrained.params')))

    return bert, vocab


def train(net, train_iter, test_iter, devices):
    net.train()
    lr, num_epochs = 1e-4, 5
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss(reduction="none")
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)


def predict():
    pass


def main():
    devices = d2l.try_all_gpus()
    bert, vocab = load_pretrained_model(
        'bert.small', num_hiddens=256, ffn_num_hiddens=512, num_heads=4,
        num_layers=2, dropout=0.1, max_len=512, devices=devices)
    batch_size, max_len, num_workers = 512, 128, d2l.get_dataloader_workers()
    # data_dir = d2l.download_extract('SNLI')
    data_dir = "../data/snli_1.0"
    train_set = SNILBERTDataset(d2l.read_snli(data_dir, True), max_len, vocab)
    test_set = SNILBERTDataset(d2l.read_snli(data_dir, False), max_len, vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True,
                                             num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                            num_workers=num_workers)
    net = BERTClassifier()
    train(net, train_iter, test_iter, devices)


if __name__ == "__main__":
    main()
