# -*- coding: utf-8 -*-
# @Time    : 2022/4/11 下午2:48
# @Author  : gavin
# @FileName: train.py.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torch
from d2l import torch as d2l
from torch import nn
from utils import MaskSoftmaxCELoss
from tqdm import tqdm
from models import Seq2SeqEncoder, Seq2SeqDecoder
from models import EncoderDecoder


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
    net.train()
    # animator = d2l.Animator(xlabel="epoch", ylabel="loss", xlim=[10, num_epochs])
    writer = SummaryWriter("./logs/loss")
    for epoch in tqdm(range(num_epochs)):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)
        for batch in date_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            # begging of sequence
            # B*1
            bos = torch.tensor([tgt_vocab["<bos>"]] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], dim=1)
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            writer.add_scalar("loss", scalar_value=metric[0] / metric[1], global_step=epoch + 1)
    torch.save(net.state_dict(), "./checkpoints/net.pth")
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')
    writer.close()


@torch.no_grad()
def predict(net, src_sentence, src_vocab, tgt_vocab, num_steps, device, save_attention_weights=False):
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
    # 翻译词
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab["<bos>"]], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []

    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        if pred == tgt_vocab["<eos>"]:
            break
        output_seq.append(pred)

    return " ".join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


if __name__ == "__main__":
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epcohs, device = 0.005, 300, d2l.try_gpu(10)
    print(device)
    train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = EncoderDecoder(encoder, decoder)
    # train_net(net, train_iter, lr, num_epcohs, tgt_vocab, device)
    print(predict(net, "go !", src_vocab, tgt_vocab, num_steps, device))