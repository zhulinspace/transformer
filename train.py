import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math,copy,time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
from tensorboardX import SummaryWriter
import torchsnooper
from model import *

class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        self.src = src # [30,10]
        self.src_mask = (src != pad).unsqueeze(-2)# [30 1 10]
        if trg is not None:
            self.trg = trg[:, :-1]#[30 ,9]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()#sum对布尔型数组中的True值进行计数

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)#[30,1,9]
        tgt_mask = tgt_mask & torch.tensor(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

# @torchsnooper.snoop()
def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        # batch src:[30,10] trg_y:[30,9] ntoken=270
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        # out ： [30 9 512]
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            # print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
            #         (i, loss / batch.ntokens, tokens / elapsed))
            print("\nloss:",loss,"\tbatch.ntokens:",batch.ntokens,'\ttokens:',tokens,'\telapsed:',elapsed)
            start = time.time()
            tokens = 0
    return total_loss / total_tokens

# deposite
# def run_all_epoch(data_iter, model, generator, criterion, opt):
#     losses = []
#     # train
#     start = time.time()
#     total_tokens = 0
#     total_loss = 0
#     tokens = 0
#     for i, batch in enumerate(data_iter):
#         # batch src:[30,10] trg_y:[30,9] ntoken=270
#         opt.optimizer.zero_grad()
#
#         out = model.forward(batch.src, batch.trg,
#                             batch.src_mask, batch.trg_mask)
#         # out ： [30 9 512]
#         x = generator(out)
#         y = batch.trg_y
#         loss = criterion(x.contiguous().view(-1, x.size(-1)),
#                          y.contiguous().view(-1))
#         loss.backward()
#         opt.step()
#
#         losses.append(loss.item())
#
#         total_loss += loss.item()
#         total_tokens += batch.ntokens
#         tokens += batch.ntokens
#         if i % 50 == 1:
#             elapsed = time.time() - start
#             print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
#                   (i, loss.item() / batch.ntokens, tokens / elapsed))
#             start = time.time()
#             tokens = 0
#
#     # eva
#     # model.eval()
#     # total=0
#     # correct=0
#     # with torch.no_grad():
#     #     for i,batch in enumerate(data_iter):
#
#     return total_loss / total_tokens
global max_src_in_batch,max_tgt_in_batch
def batch_size_fn(new,count,sofar):
    "keep augmenting batch and calculate total number of tokens + padding"
    global max_src_in_batch,max_tgt_in_batch
    if count==1:
        max_src_in_batch=0
        max_tgt_in_batch=0
    max_src_in_batch=max(max_src_in_batch,len(new.src))
    max_tgt_in_batch=max(max_tgt_in_batch,len(new.trg)+2)
    src_elements=count*max_src_in_batch
    tgt_elements=count*max_tgt_in_batch
    return max(src_elements,tgt_elements)


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        '''
        :param model_size: model.src_embed[0].d_model
        :param factor: 1
        :param warmup: 400
        :param optimizer: torch.optim.Adam(model.parameters(),lr=0,betas=(0.9,0.98),eps=(1e-9))
        '''
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    """
        With label smoothing,
        KL-divergence between q_{smoothed ground truth prob.}(w)
        and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, size, padding_idx, smoothing=0.0):
        '''

        :param size: tgt_vocab_size
        :param padding_idx:
        :param smoothing:
        '''
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        '''
        :param x: FloatTensor:batch_size x n_classes
        :param target: (LongTensor) :batch_size
        :return:
        '''
        # x =[270,11] y=[270]

        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist

        return self.criterion(x, torch.tensor(true_dist, requires_grad=False))


'''A simple example'''
def data_gen(V,batch,nbatches):# 11 30 20
    'Generate random data for a src-tgt copy task'
    for i in range(nbatches):
        data=np.random.randint(1,V,size=(batch,10))
        data[:,0]=1
        src=torch.LongTensor(data).requires_grad_(False) # [30 10]
        tgt=torch.LongTensor(data).requires_grad_(False) #[30 10]
        yield Batch(src,tgt,0)

# @torchsnooper.snoop()
class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)

        # print(f"x :{x.dtype},target:{y.dtype}")
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm

if __name__=='__main__':

    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    # criterion = nn.KLDivLoss(size_average=False)
    model = make_model(V, V, N=2)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    # with torchsnooper.snoop():
    for epoch in range(10):
        model.train()
        print("epoch",epoch)
        run_epoch(data_gen(V, 30, 20), model,
                  SimpleLossCompute(model.generator, criterion, model_opt))

        model.eval()
        print("eval")
        print(run_epoch(data_gen(V, 30, 5), model,
                        SimpleLossCompute(model.generator, criterion, None)))
