# -*- coding: utf-8 -*-
# @Time    : 2022/5/20 下午4:12
# @Author  : gavin
# @FileName: bert_dataset.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
from torch import nn
import torch
import multiprocessing


def tokenize(lines, token='word'):
    """Split text lines into word or character tokens.
    Defined in :numref:`sec_text_preprocessing`"""

    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unknown token type: ' + token)


def get_tokens_and_segments(tokens_a, tokens_b=None):
    """获取相应的token和句子位置编码"""

    tokens = ["<cls>"] + tokens_a + ["<seq>"]
    # 段嵌入
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ["<seq>"]
        segments += [1] * (len(tokens_b) + 1)

    return tokens, segments


class SNILBERTDataset(nn.Module):
    """构建基于Bert的SNIL数据格式"""

    def __int__(self, dataset, max_len, vocab=None):
        all_premise_hypothesis_tokens = [[p_tokens, h_tokens] for p_tokens, h_tokens in
                                         zip(*[tokenize([s.lower() for s in sentences])
                                               for sentences in dataset[:2]])]
        self.labels = torch.tensor(dataset[2])
        self.vocab = vocab
        self.max_len = max_len
        (self.all_token_ids, self.all_segments, self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)
        print("read " + str(len(self.all_token_ids)) + " examples")

    def _preprocess(self, all_premise_hypothesis_tokens):
        """构建多进程处理数据"""

        # todo:多进程如何工作
        pool = multiprocessing.Pool(4)
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
        all_token_ids = [
            token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        return (torch.tensor(all_token_ids, dtype=torch.long), torch.tensor(all_segments, dtype=torch.long),
                torch.tensor(valid_lens, dtype=torch.long))

    def _mp_worker(self, premise_hypothesis_tokens):
        """单个线程需要处理的数据"""
        p_tokens, h_tokens = premise_hypothesis_tokens
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        # 分割词元序列
        tokens, segments = get_tokens_and_segments(p_tokens, h_tokens)
        # pad填补序列
        token_ids = self.vocab([tokens]) + [self.vocab['<pad>']] * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        """为bert预留词保存: '<cls>', <sep>和<sep>"""

        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def __getitem__(self, idx):

        return self.all_token_ids[idx], self.all_segments[idx], self.valid_lens[idx], self.labels[idx]

    def __len__(self):
        return len(self.all_token_ids)
