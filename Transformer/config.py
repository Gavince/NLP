# -*- coding: utf-8 -*-
# @Time    : 2022/4/19 下午5:32
# @Author  : gavin
# @FileName: config.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
from d2l import torch as d2l
import argparse


def get_parser():

    parser = argparse.ArgumentParser(description="参数解析!")
    parser.add_argument("--num_hiddens", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--lr", "--learning_rate", type=float, default=0.005)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--device", type=int, default=d2l.try_gpu(10), help="输入可供运行的GPU数目")
    parser.add_argument("--ffn_num_input", type=int, default=32)
    parser.add_argument("--ffn_num_hiddens", type=int, default=64)
    parser.add_argument("--key_size", type=int, default=32, help="key的维度大小")
    parser.add_argument("--query_size", type=int, default=32, help="query的维度大小")
    parser.add_argument("--value_size", type=int, default=32, help="value的维度大小")
    parser.add_argument("--norm_shape", type=list, default=[32], help="layer norm归一化的参数")
    parser.add_argument("--is_training", type=bool, default=False, help="是否进行训练(默认设置为训练)")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint")

    return parser.parse_args()
