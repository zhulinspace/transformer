Attenion is all you need 这篇paper提出了transformer架构，Transformer提出是解决RNN无法并行化运算，同时解决CNN而又无法看到所有的信息。

RNN对于输入序列无法进行并行化运算，即RNN得输出不可以同时得到，RNN运算是依赖于序列的输入顺序，即当前的输出是依赖于之前的运算，如果是对于单向的RNN,当前时间节点的输出是在看过之前输入的部分得到的，而双向RNN当前的输出是在看过所有输入得到的，但是必须在之前时间节点计算出hidden state等才能得到当前的输出，即无法并行化运算是RNN处理序列问题一个计算瓶颈，而transformer则可以同时得到所有输出序列，即计算输出序列的每个元素的操作是并行的，但同时这也意味着输入序列的顺序并未被考虑进去，而输入序列的顺序则代表着语义信息，因此增加position encoding部分来解决transformer的顺序问题。Transformer也是用的encoder和decoder的架构，但是encoder和decoder的内部不再是RNN,而是self-attention.下图是transformer的架构

![](https://github.com/zhulinspace/transformer/blob/master/img/network.png)

大体框架是：

encoder里以encoderlayer 来进行堆叠，堆叠层数一般为8层，而encoderlayer里面包含muti-head attention（self-attention）和feed forward 两个子层sublayer,其中每个子层都要进行残差连接以及layer normalization。

decoder以decoderlayer进行堆叠，其中包含三个子层，masked muti-head attention （self-attention）,muti-head attention(encoder-decoder attention)---这和Loung global attention 类似，feed-forward。同样进行残差连接和layer norm

注意不同层数的参数不同  

# network  part         

## self-attention

简单回顾一下，Loung在2015年提出得global attention，即先保存encoder得所有hidden state,在decoder时，拿当前时间节点得输出Y，和encoder的hidden state比较相似程度得到alignment score，encoder hiddenstates加权aligment score后求和平均得到语义向量c，作为decoder的输入。

self-attention作为attenion的一种机制，it allows the inputs to interact with each other("self"),and find out who they should pay more attention to ('attention'),the outputs are aggregates of these interactions and the attention scores.

如何实现self-attention(以encoder的self-attention layer为例):

input-embedding和position encoding相加后作为self-attention的输入input，对input做三个不同的线性变化操作得到Q,K,V ，Q和K的转置除以根号下k的维度得到 alignment score，进行softmax后与V相乘得到self-attention层的输出

![](https://github.com/zhulinspace/transformer/blob/master/img/self_attention_matrix.png)

![](https://github.com/zhulinspace/transformer/blob/master/img/attention_formulate.png)

而muti-head attention则是将最后结果进行concatenate

![muti-head attention](https://github.com/zhulinspace/transformer/blob/master/img/muti_head_attention.png)

![muti-head operation](https://github.com/zhulinspace/transformer/blob/master/img/muti_head_operation.png)

## Position encoding

如上所说，必须给transformer加上序列的位置信息，解决方案是给序列中每个词加上该词的位置信息，即positional encoding，即对于输入的input-embedd,在加上positional encoding部分。

即一个简单的想法是对于句子 "I  am a student",对于单词I,加上1，对于am，加上2 ......

然而在对于句子过长，则加上的值就会变得很大，其也有可能会降低其泛化能力。在paper里面提出了一种方法，该方法并不是加上一个单个的值，而是一个包含在句子某个特定位置信息的d维的vector，

假设t是输入序列的某个词的特定位置，而pt是要加上的position encoding,其维度为d

则定义可以产生pt的函数f如下：

![ ](https://github.com/zhulinspace/transformer/blob/master/img/positional_encoding_1.png)

其中embeded的维度和position encoding的维度要相同，否则不能相加。

## layer normalization（LN）

LN和BN(batch norm)是两种常见的归一化方法，而LN是较常用于RNN网络中的，两者的区别是

- LN ：it normalize the inputs across the features

- BN: it normalize the input feature across the batch dimension

  ![LN and BN](https://github.com/zhulinspace/transformer/blob/master/img/LN_and_BN.PNG)



## the decoder side :masked muti-head attention

在decoder里，self attention层只能attent to 之前位置的词，attend 不到后面位置的词，因为还没有预测出来，这导致了在decoder self attention在计算alignment score时需要掩码来式后面位置的词为无穷，这样才能保证注意不到，这部分会在代码部分详细讲解

## the decoder side：encoder-decoder attention

在encoder-decoder attention时，需要利用top encoder的K和V的值，可以通过上面框架图进行理解

# Train part

## label smoothing

label smoothing可以解决overfitting和overconfidence的问题

标签平滑所作的事情简单来说，在做图像二分类问题时，如果把猫的图像定义标签为0，狗的图像定义标签为1，则这种标签可以称作是hard label,而如果训练数据出现打错标签（mislabeled）的情况，则那么可能导致学习的分类器学习不到正确的猫和狗的特征，为了避免这种情况，我们首先定义label_smoothing为0.2 ，然后根据以下公式，把标签从【0，1】（hard label）调整到【0.1,0.9】(soft label)，降低了标签得confidence

new_onehot_labels=onehot_labels*(1-label_smoothing)+label_smoothing/num_classes

[0.1, 0.9]=[0 , 1]*(1-0.2)+0.2/2



## optimizer

## greedy decoding

## beam search



### Ref

[ http://nlp.seas.harvard.edu/2018/04/03/attention.html ]( http://nlp.seas.harvard.edu/2018/04/03/attention.html )

[ https://towardsdatascience.com/what-is-label-smoothing-108debd7ef06 ]( https://towardsdatascience.com/what-is-label-smoothing-108debd7ef06 )

[ http://jalammar.github.io/illustrated-transformer/ ]( http://jalammar.github.io/illustrated-transformer/ )

[ https://kazemnejad.com/blog/transformer_architecture_positional_encoding/ ]( https://kazemnejad.com/blog/transformer_architecture_positional_encoding/ )

### note

对于transformer学习过程，建议如下：

- 可以先看[李宏毅老师的视频]( https://www.bilibili.com/video/av48285039?p=92 )有一个模糊概念

- 在认真看一遍[illustrated-transformer]([ http://jalammar.github.io/illustrated-transformer/)掌握整体框架

- 再通过[the annotated transformer]( http://nlp.seas.harvard.edu/2018/04/03/attention.html )看代码，对论文以及代码不懂得地方细看

- 推荐使用hook或者torchsnooper进行debug

  