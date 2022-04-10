# -*- coding: utf-8 -*-
# @Time    : 2022/4/9 下午7:47
# @Author  : gavin
# @FileName: train.py.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
import matplotlib
import matplotlib.pyplot as plt
import torch.optim
from d2l import torch as d2l
from torch import nn
import math
from models import GRUModelScratch
from models.gru import  get_params_gru, init_gru_state, gru
from tqdm import tqdm


def grap_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params

    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def predict(prefix, num_preds, net, vocab, device):
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    # warming up
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    # prediction
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return "".join([vocab.idx_to_token[i] for i in outputs])


def train_epoch(net, train_iter, loss, updater, device, use_random_iter):
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)
    for X, Y in train_iter:
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # GRU,RNN
                state.detach_()
            else:
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grap_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grap_clipping(net, 1)
            updater(batch_size=1)

        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def train(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel="epoch", ylabel="perplexity", legend=["train"], xlim=[10, num_epochs])
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr=lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    pred = lambda prefix: predict(prefix, 50, net, vocab, device)

    for epoch in tqdm(range(num_epochs)):
        ppl, speed = train_epoch(net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10:
            print(pred("time traveller"))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(pred('time traveller'))
    print(pred('traveller'))


if __name__ == "__main__":
    batch_size, num_steps = 32, 35
    num_hiddens = 512
    device = d2l.try_gpu()
    num_epochs, lr = 500, 1
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

    # rnn_layer = nn.RNN(len(vocab), num_hiddens)
    # net = RNNModel(rnn_layer, len(vocab)).to(device)

    net = GRUModelScratch(len(vocab), num_hiddens, device, get_params_gru, init_gru_state, gru)
    print(predict("time traveller", 10, net, vocab, device))
    train(net, train_iter, vocab, lr, num_epochs, device)
    plt.show()

