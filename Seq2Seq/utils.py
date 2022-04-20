# -*- coding: utf-8 -*-
# @Time    : 2022/4/11 下午4:51
# @Author  : gavin
# @FileName: utils.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
import torch
from torch import nn


def squence_mask(X, valid_len, value=0):
    """
    序列掩码
    :param X:输入序列 [B, T]
    :param valid_len: 有效的输入序列长度 [B]
    :param value: 填充值，默认为0
    :return:
    """
    max_len = X.size(1)
    mask = torch.arange((max_len), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


class MaskSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    def forward(self, pre, label, valid_len):
        """
        :param pre:pred的形状：(batch_size,num_steps,vocab_size)
        :param label:label的形状：(batch_size,num_steps)
        :param valid_len:valid_len的形状：(batch_size,)
        :return:
        """

        weights = torch.ones_like(label)
        weights = squence_mask(weights, valid_len)
        self.reduction = "none"
        unweighted_loss = super(MaskSoftmaxCELoss, self).forward(pre.permute(0, 2, 1), label)
        weights_loss = (unweighted_loss * weights).mean(dim=1)

        return weights_loss


if __name__ == "__main__":
    X = torch.tensor([[1, 2, 3], [4, 5, 6]])
    valid_len = torch.tensor([1, 2])
    print(squence_mask(X, valid_len))
    loss = MaskSoftmaxCELoss()
    l = loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long),
             torch.tensor([4, 2, 0]))
    print(l)
