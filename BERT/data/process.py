# -*- coding: utf-8 -*-
# @Time    : 2022/4/25 上午9:06
# @Author  : gavin
# @FileName: process.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
import os
import random
import torch
from d2l import torch as d2l


def _read_wiki(data_dir):
    """分割句子"""

    file_name = os.path.join(data_dir, "wiki.train.tokens")
    with open(file_name, "r") as f:
        lines = f.readlines()
    # 段落
    # [["A", "B"], ["C", "D", "E"]] A,B表示单个句子
    paragraphs = [line.strip().lower().split(".")
                  for line in lines if len(line.split(".")) >= 2]
    random.shuffle(paragraphs)

    return paragraphs


def get_tokens_and_segments(tokens_a, tokens_b=None):
    """获取相应的token和句子位置编码"""

    tokens = ["<cls>"] + tokens_a + ["<seq>"]
    # 段嵌入
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ["<seq>"]
        segments += [1] * (len(tokens_b) + 1)

    return tokens, segments


def _get_next_sentence(sentence, next_sentence, paragraphs):
    """二分类句子对数据构造"""

    if random.random() < 0.5:
        is_next = True
    else:
        # 从所有段落里面挑选一个, 再从相应的段落里面挑一个句子出来
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False

    return sentence, next_sentence, is_next


def _get_nxt_data_from_paragraph(paragraph, paragrahs, vocab, max_len):
    """
    下一个句子预测的样本对
    """
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        # 构建句子对
        tokens_a, tokens_b, is_next = _get_next_sentence(paragraph[i], paragraph[i + 1], paragrahs)
        # 设置最大长度限制
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        # tokens:["<cls>", "I", "hate", "<seq>", "beautiful", "day", "seq"]
        # segments:[0, 0, 0, 0, 1, 1, 1]
        # is_next:[1]
        tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))

    return nsp_data_from_paragraph


def _replace_mlm_tokens(tokens, candidate_pred_position, num_mlm_preds, vocab):
    """
    对所有序列tokens数据的百分之15的数据进行位置掩码和标签
    :param tokens: 带特殊标记的token样本
    :param candidate_pred_position:有效的可选择位置
    :param num_mlm_preds: 最大掩码词数
    :param vocab: 词表大小
    :return:
    """
    # mlm_input_tokens: ["<cls>", "I", "hate", "<seq>", "beautiful", "day", "seq"]
    mlm_input_tokens = [token for token in tokens]
    pred_position_and_labels = []
    # 保证位置上的随机抽取
    random.shuffle(candidate_pred_position)
    for mlm_pred_position in candidate_pred_position:
        if len(pred_position_and_labels) >= num_mlm_preds:
            break
        mask_token = None
        # 80%, 替换为mask
        if random.random() < 0.8:
            mask_token = "<mask>"
        else:
            # 10%, 保持不变
            if random.random() < 0.5:
                mask_token = tokens[mlm_pred_position]
            # 10%, 随机选择其它词进行填补
            else:
                mask_token = random.choice(vocab.idx_to_token)
        # mlm_input_tokens: ["<cls>", "I", "<mask>", "<seq>", "beautiful", "day", "seq"]
        mlm_input_tokens[mlm_pred_position] = mask_token
        # (预测位置, 真实标签)
        # (2, "hate")
        pred_position_and_labels.append((mlm_pred_position, tokens[mlm_pred_position]))

    return mlm_input_tokens, pred_position_and_labels


def _get_mlm_data_from_tokens(tokens, vocab):
    """
    获取mlm后的数据
    :param tokens:
    :param vocab: 词表
    :return: 序列, 掩码位置, 掩码标签
    """
    candidate_pred_position = []
    # 舍弃特殊的标记位置，只对有效值进行掩码处理
    for i, token in enumerate(tokens):
        if token in ["<cls>", "<sep>"]:
            continue
        candidate_pred_position.append(i)

    # 随机抽取整个序列15%的数据进行预测任务
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    # eg: ["<cls>", "mask", "i"], [(1, "mask")]
    mlm_input_tokens, pre_position_and_labels = _replace_mlm_tokens(tokens
                                                                    , candidate_pred_position, num_mlm_preds, vocab)

    # 按照时序位置进行排序
    pre_position_and_labels = sorted(pre_position_and_labels, key=lambda x: x[0])
    pred_positions = [v[0] for v in pre_position_and_labels]
    mlm_pred_labels = [v[1] for v in pre_position_and_labels]

    # mlm_pre_labels为掩蔽后的预测值
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]


def _pad_bert_inputs(examples, max_len, vocab):
    """构建bert模型输入，并转为等长的tensor"""
    # TODO:为什么要在最长长度上在选择0.15
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens, = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        # 输入向量维度
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (
                max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (
                max_len - len(segments)), dtype=torch.long))

        # valid_lens不包括'<pad>'的计数
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
                max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        # 填充词元的预测将通过乘以0权重在损失中过滤掉
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                    max_num_mlm_preds - len(pred_positions)),
                         dtype=torch.float32))
        # 真实的类别标签
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
                max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))

    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)


class _WikiTextDataset(torch.utils.data.Dataset):

    def __init__(self, paragraphs, max_len):
        # 对每个句子按照词级别进行划分token
        # input: [["A", "B", "C"], ["E", "F"]]
        # output:[
        # [["a1", "a2", "a3"], ["b1", "b2"], ["c1", "c2"]]，
        # [["e1", "e2"], ["f1", f2]]
        # ]
        paragraphs = [d2l.tokenize(paragraph, token="word") for paragraph in paragraphs]
        # 建立词表
        # [["A"], ["B"], ［"C"］, ［"D"］, ［"E"］]
        sentences = [sentence for paragraph in paragraphs for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=["<pad>" "<mask>", "<cls>", "<sep>"])
        # nsp任务数据构造
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nxt_data_from_paragraph(paragraph, paragraphs, self.vocab, max_len))
        # mlm任务数据构造
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab) + (segments, is_next)) for tokens, segments, is_next
                    in examples]

        # 填充输⼊, token转为id,并进行填补
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)


def load_data_wiki(batch_size, max_len):
    """加载WikiText-2数据集"""
    num_workers = d2l.get_dataloader_workers()
    # data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki("../data/wikitext-2/")
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                             shuffle=True, num_workers=num_workers)
    return train_iter, train_set.vocab


if __name__ == "__main__":
    batch_size, max_len = 1, 64
    train_iter, vocab = load_data_wiki(batch_size, max_len)
    for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
         mlm_Y, nsp_y) in train_iter:
        print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
              pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
              nsp_y.shape)
        for val in (tokens_X, segments_X, valid_lens_x, pred_positions_X
                    , mlm_weights_X, mlm_Y, nsp_y):
            print(val)
        break
