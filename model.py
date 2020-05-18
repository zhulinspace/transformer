import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math,copy,time
import matplotlib.pyplot as plt
import seaborn
from tensorboardX import SummaryWriter
seaborn.set_context(context='talk')
import torchsnooper

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):

    def __init__(self,d_model,vocab):
       super(Generator,self).__init__()
       self.proj=nn.Linear(d_model,vocab)

    def forward(self,x):
        return F.log_softmax(self.proj(x),dim=-1)


# the encoder is composed of a stack of N=6 identical layers
def clones(module,N):
    "produce N in identical layers"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    # copy.deepcopy
    # ModuleList 和 sequential

class LayerNorm(nn.Module):
    "construct a layernorm module"
    def __init__(self,features,eps=1e-6):
        super(LayerNorm,self).__init__()
        self.a_2=nn.Parameter(torch.ones(features))
        self.b_2=nn.Parameter(torch.zeros(features))
        #注册参数到nn
        self.eps=eps

    def forward(self,x):
        mean=x.mean(-1,keepdim=True)
        std=x.std(-1,keepdim=True)
        # -1表示最后一个维度
        return self.a_2*(x-mean)/(std+self.eps)+self.b_2

class SublayerConnection(nn.Module):
    '''
    A residual connection followed by a layer norm
    Note for code simplicity the norm is first as opposed to last
    '''
    def __init__(self,size,dropout):
        super(SublayerConnection,self).__init__()
        self.norm=LayerNorm(size)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,sublayer):
        'apply residual connection to any sublayer with the same size'
        # print("先进入sublayerconnection")
        return x+self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    'encoder is made up of self-attn and feed forward (defined below)'
    def __init__(self,size,self_attn,feed_forward,dropout):
        # size=512
        super(EncoderLayer,self).__init__()
        self.self_attn=self_attn
        self.feed_forward=feed_forward
        self.sublayer=clones(SublayerConnection(size,dropout),2)
        self.size=size

    def forward(self,x,mask):
        # print("未进行attention之前的mask",mask.size())
        x=self.sublayer[0](x,lambda x:self.self_attn(x,x,x,mask))

        return self.sublayer[1](x,self.feed_forward)



class Encoder(nn.Module):
    "core encoder is a satck of N layers"
    def __init__(self,layer,N):
        super(Encoder,self).__init__()
        self.layers=clones(layer,N)
        self.norm=LayerNorm(layer.size)

    def forward(self,x,mask):
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)

class Decoder(nn.Module):
    'generic N layer decoder with masking'
    def __init__(self,layer,N):
        super(Decoder,self).__init__()
        self.layers=clones(layer,N)
        self.norm=LayerNorm(layer.size)

    def forward(self,x,memory,src_mask,tgt_mask):
        for layer in self.layers:
            x=layer(x,memory,src_mask,tgt_mask)

        return self.norm(x)


class DecoderLayer(nn.Module):
    'decoder is made of self-attn,src-attn,feed-forward'
    def __init__(self,size,self_attn,src_attn,feed_forward,dropout):
        super(DecoderLayer,self).__init__()
        self.size=size
        self.self_attn=self_attn
        self.src_attn=src_attn
        self.feed_forward=feed_forward
        self.sublayer=clones(SublayerConnection(size,dropout),3)

    def forward(self,x,memory,src_mask,tgt_mask):
        # input:embed(tgt) memory:encoder的输出
        m=memory
        x=self.sublayer[0](x,lambda x:self.self_attn(x,x,x,tgt_mask))
        x=self.sublayer[1](x,lambda x:self.src_attn(x,m,m,src_mask))
        return self.sublayer[2](x,self.feed_forward)

def subsequent_mask(size):
    '使decoder self-attention只基于以及预测的输出'
    attn_shape=(1,size,size)
    subsequent_mask=np.triu(np.ones(attn_shape),k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask)==0
    '''对于[1,size,size]的matrix,等于0的地方变成True'''

def attention(query,key,value,mask=None,dropout=None):
    "compute 'scaled Dot Product Attention"
    d_k=query.size(-1)# 64
    scores=torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
    if mask is not None:
        scores=scores.masked_fill(mask==0,-1e9)
    p_attn=F.softmax(scores,dim=-1)
    if dropout is not None:
        p_attn=dropout(p_attn)

    return torch.matmul(p_attn,value),p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self,h,d_model,dropout=0.1):
        'take in model size and number of heads'
        super(MultiHeadedAttention,self).__init__()
        assert d_model%h==0
        # we assume d_v = d_k
        self.d_k=d_model//h # 64
        self.h=h

        self.linears=clones(nn.Linear(d_model,d_model),4)
        self.attn=None
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,query,key,value,mask=None):
        # print("先进入到attention!  11111")
        if mask is not None:
            mask=mask.unsqueeze(1)# mask=[30,1,1,10]
            # print('mask_size',mask.size())
        nbatches=query.size(0)# 30

        query,key,value=\
        [
            l(x).view(nbatches,-1,self.h,self.d_k).transpose(1,2)
            for l ,x in zip(self.linears,(query,key,value))
        ]

        # print("query.size",query.size()) [30,8,10,64]

        x,self.attn=attention(query,key,value,mask=mask,dropout=self.dropout)

        x=x.transpose(1,2).contiguous()\
            .view(nbatches,-1,self.h*self.d_k)
        return self.linears[-1](x)# [30 10 512]

class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super(PositionwiseFeedForward,self).__init__()
        self.w_1=nn.Linear(d_model,d_ff)
        self.w_2=nn.Linear(d_ff,d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        # print("进入FeedFowrad")
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    '''和其它序列转化模型类似，利用可以学习的嵌入层来把输入输出的索引转化为向量，维度为（d_model)
    我们也利用线性层和softmax来把decoder的输出变成预测下一个 单词的概率，在我们的模型中，
    我们共享嵌入层和线性层的参数，特别的在嵌入层我们把参数乘上根号下d_model'''
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    '''因为我们的模型没有RNN和CNN,为了让模型能够利用序列的参数，
    我们必须注入一些参数关于序列中元素的绝对或者相对位置信息  
      所以在encoder和decoder栈底，我们加了position encoding到input-embedding
      PE和嵌入层有相同的维度d-model,所以才能被相加'''


    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)# pe=[1,5000,512]


    def forward(self, x):
        x = x + torch.tensor(self.pe[:, :x.size(1)],
                         requires_grad=False)
        # print("after pc",x.size())
        return self.dropout(x)


def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model




if __name__=='__main__':

    #small example model
    with torchsnooper.snoop():
        tmp_model=make_model(11,11,2)
        data=np.random.randint(1,11,size=(30,10))
        data[:,0]=1
        src=torch.LongTensor(data)# [30,10] int32
        src_mask = (src != 0).unsqueeze(-2) # [30,1,10] bool
        trg=torch.LongTensor(data)
        trg=trg[:,:-1]# [30,9] int32
        trg_mask=(trg!=0).unsqueeze(-2)
        trg_mask=trg_mask&torch.tensor(subsequent_mask(trg.size(-1)).type_as(trg_mask.data)).clone().detach()
        # [30,9,9] bool
        out=tmp_model(src,trg,src_mask,trg_mask)


    # print("src",src.size(),src.dtype)
    # print('\nsrc_mask',src_mask.size(),src_mask.dtype)
    # print('\ntrg',trg.size(),trg.dtype)
    # print('\ntrg_mask',trg_mask.size(),trg_mask.dtype)
    # print("out",out.size())# [30 9 512]

    # with SummaryWriter(comment='transformer') as w:
    #     w.add_graph(tmp_model,(dummpy_input,))
    # print(tmp_model)
