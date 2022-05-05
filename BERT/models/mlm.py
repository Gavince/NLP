# -*- coding: utf-8 -*-
# @Time    : 2022/4/26 上午11:07
# @Author  : gavin
# @FileName: mlm.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
import torch
from torch import nn


class MaskLM(nn.Module):
    """基于掩蔽语言模型任务"""

    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size)
                                 )

    def forward(self, X, pred_position):
        """
        计算基于掩蔽模型的预测任务，主要实现对掩码区域进行预测
        :param X: shape:[B, T, H]
        :param pred_position: shape[B, C]
        :return:
        """
        # 多少个预测位置
        num_pred_positions = pred_position.shape[1]
        pred_position = pred_position.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # 假定batch_size = 2, num_pred_positions=3
        # batch_idx = [0, 0, 0, 1, 1, 1] 表示每一个每一个batch所获取的有效值
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        # 表示第几个样本的第T个时刻值
        # masked_X  [B*num_pred_positions, H]
        masked_X = X[batch_idx, pred_position]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_y_hat = self.mlp(masked_X)
        # [B, num_pred_positions, V]
        return mlm_y_hat


if __name__ == "__main__":
    mlm = MaskLM(10000, 768)
    mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
    mlm_Y_hat = mlm(torch.randn((2, 8, 768)), mlm_positions)
    print(mlm_Y_hat.shape)
    print(mlm)
    # 测试损失
    mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
    loss = nn.CrossEntropyLoss(reduction="none")
    mlm_l = loss(mlm_Y_hat.reshape((-1, 10000)), mlm_Y.reshape(-1))
    print(mlm_l.shape, mlm_l)
