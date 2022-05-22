# -*- coding: utf-8 -*-
# @Time    : 2022/4/26 上午11:07
# @Author  : gavin
# @FileName: nsp.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
from torch import nn


class NextSentencePred(nn.Module):
    """预测下一个句子的任务"""

    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        """
        :param X: [B, num_hiddens]
        :return: [B, 2]
        """
        return self.output(X)


if __name__ == "__main__":
    nst = NextSentencePred(728)
    print(nst)
