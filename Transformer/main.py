# -*- coding: utf-8 -*-
# @Time    : 2022/4/18 上午8:12
# @Author  : gavin
# @FileName: main.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from Seq2Seq.utils import MaskSoftmaxCELoss
from models import TransformerDecoder
from models import TransformerEncoder
import collections
import math
import torch
from d2l import torch as d2l
from torch import nn
from tqdm import tqdm
import config
import wandb


def train_net(net, date_iter, lr, num_epochs, tgt_vocab, device):
    print("Training......!")

    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_normal_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    loss = MaskSoftmaxCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    wandb.watch(net)
    net.train()
    for epoch in tqdm(range(num_epochs)):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)
        for batch in date_iter:
            optimizer.zero_grad()
            # X:[B, T], valid_len:[B]
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            # 为预测加入句子的起始符: begging of sequence
            # B*1
            bos = torch.tensor([tgt_vocab["<bos>"]] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], dim=1)
            # Decoder:拼接起始符号和原始输入
            # 直接输出T个时刻的预测结果
            Y_hat, _ = net(X, dec_input, X_valid_len)
            # l:[B, 1]
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)

        if (epoch + 1) % 10 == 0:
            wandb.log({"loss": metric[0] / metric[1]})

    torch.save(net.state_dict(), "./checkpoints/net.pth")
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')


@torch.no_grad()
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device, save_attention_weights=False):
    net.load_state_dict(torch.load("./checkpoints/net.pth"))
    net.eval()
    # 处理为num
    src_tokens = src_vocab[src_sentence.lower().split(" ")] + [src_vocab["<eos>"]]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab["<pad>"])
    # 增加B维度
    enx_X = torch.unsqueeze(torch.tensor(src_tokens, device=device, dtype=torch.long), dim=0)
    enc_outputs = net.encoder(enx_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 预测的起始值（第一个预测字符为开始字符编码）
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab["<bos>"]], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []

    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 预测截止
        if pred == tgt_vocab["<eos>"]:
            break
        output_seq.append(pred)

    return " ".join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


def bleu(pred_seq, lab_seq, k):
    pre_tokens, label_tokens = pred_seq.split(" "), lab_seq.split(" ")
    len_pred, len_label = len(pre_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    # 计算N-Gram
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        # 统计词频
        for i in range(len_label - n + 1):
            label_subs[" ".join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[" ".join(pre_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[" ".join(pre_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))

    return score


def evaluate(net, engs, fras, src_vocab, tgt_vocab, num_layers, num_steps, num_heads, device, visiual=True):
    net.load_state_dict(torch.load("./checkpoints/net.pth"))
    net.eval()
    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device, True)
        print(f'{eng} => {translation}, ',
              f'bleu {bleu(translation, fra, k=2):.3f}')

    if visiual:
        enc_attention_weights = torch.cat(net.encoder.attention_weights, 0).reshape((num_layers, num_heads,
                                                                                     -1, num_steps))
        # Encoder注意力权值可视化
        d2l.show_heatmaps(
            enc_attention_weights.cpu(), xlabel='Key positions',
            ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
            figsize=(7, 3.5))

        # Decoder注意力权重可视化

        dec_attention_weights_2d = [head[0].tolist()
                                    for step in dec_attention_weight_seq
                                    for attn in step for blk in attn for head in blk]
        dec_attention_weights_filled = torch.tensor(
            pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
        dec_attention_weights = dec_attention_weights_filled.reshape((-1, 2, num_layers, num_heads, num_steps))
        dec_self_attention_weights, dec_inter_attention_weights = \
            dec_attention_weights.permute(1, 2, 3, 0, 4)
        d2l.show_heatmaps(
            dec_self_attention_weights[:, :, :, :len(translation.split()) + 1],
            xlabel='Key positions', ylabel='Query positions',
            titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))
        # Encoder-Decoder注意力权重可视化
        d2l.show_heatmaps(
            dec_inter_attention_weights, xlabel='Key positions',
            ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
            figsize=(7, 3.5))

        plt.show()


if __name__ == "__main__":
    args = config.get_parser()
    print("正在加载数据集......!")
    train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(args.batch_size, args.num_steps)
    print("数据集加载完成......!")

    encoder = TransformerEncoder(
        len(src_vocab), args.key_size, args.query_size, args.value_size, args.num_hiddens,
        args.norm_shape, args.ffn_num_input, args.ffn_num_hiddens, args.num_heads,
        args.num_layers, args.dropout)

    decoder = TransformerDecoder(
        len(tgt_vocab), args.key_size, args.query_size, args.value_size, args.num_hiddens,
        args.norm_shape, args.ffn_num_input, args.ffn_num_hiddens, args.num_heads,
        args.num_layers, args.dropout)

    net = d2l.EncoderDecoder(encoder, decoder)
    wandb.init(project="Transform", entity="wanyu")
    wandb.config = args
    if args.is_training:
        train_net(net, train_iter, args.lr, args.num_epochs, tgt_vocab, args.device)
    else:
        # 预测
        engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
        fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
        evaluate(net, engs, fras, src_vocab, tgt_vocab, args.num_layers, args.num_steps, args.num_heads, args.device)
