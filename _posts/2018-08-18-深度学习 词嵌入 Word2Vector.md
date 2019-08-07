---
layout:     post   				    # 使用的布局（不需要改）
title:      深度学习 / 正则化
subtitle:      #副标题
date:       2018-08-19 				# 时间
author:     zhu.xinwei 		    	# 作者
header-img: img/post-bg-desk.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - Linear Regression
    - Logistic Regression
    - Universal approximation theorem
---


## 图像领域的预训练


## NLP领域的预训练

### 语言模型


### Word2Vec

#### CBOW

#### Skip-Gram


Word Embedding 等价于把 Onehot 层到 embedding 层的网络用预训练好的参数矩阵 Q 初始化。这和图像领域的低层预训练过程其实是一样的，区别是 Word Embedding 只能初始化第一层网络参数，再高层的参数就无能为力了。下游 NLP 任务在使用 Word Embedding 的时候也类似图像有两种做法：一种是 Frozen，就是 Word Embedding 那层网络参数固定不动；另外一种是 Fine-Tuning，就是 Word Embedding 这层参数使用新的训练集合训练也需要跟着训练过程更新掉。


**Word Embedding 存在什么问题？ 无法区分多义词的不同语义，无法解决NLP常见的歧义性现象，这就是它的一个比较严重的问题。**


### ELMO 语言模型嵌入

>  ELMo(Embedding from Language Models)模型是在NAACL 2018 最佳论文[Deep contextualized word representation](https://aclweb.org/anthology/N18-1202)中提出的。简单来，其核心思想是根据当前上下文对Word Embedding做动态调整，这样做可以缓解多义性问题。

> Word Embedding 本质上是个静态的方式，所谓静态指的是训练好之后每个单词的表达就固定住了，以后使用的时候，不论新句子上下文单词是什么，这个单词的 Word Embedding 不会跟着上下文场景的变化而改变。


ELMO 的本质思想是：先用语言模型学好一个单词的 Word Embedding，此时多义词无法区分。在实际使用 Word Embedding 的时候，单词已经具备了特定的上下文，这个时候ELMO可以根据上下文单词的语义去调整单词的 Word Embedding 表示，这样经过调整后的 Word Embedding 更能表达在这个上下文中的具体含义，自然也就解决了多义词的问题了。所以 ELMO 本身是个根据当前上下文对 Word Embedding 动态调整的思路。

ELMO存在的问题：
- LSTM提取特征能力远弱于Transformer
- 拼接方式双向融合特征融合能力偏弱

> Transformer 是谷歌在 17 年做机器翻译任务的“Attention is all you need”的论文中提出的，是个叠加的“自注意力机制（Self Attention）”构成的深度网络，是目前 NLP 里最强的特征提取器。很多研究已经证明了 Transformer 提取特征的能力是要远强于 RNN、CNN传统特征提取器 。RNN 一直受困于其并行计算能力，这是因为它本身结构的序列性依赖导致的，CNN 的最大优点是易于做并行计算，所以速度快，但是在捕获 NLP 的序列关系尤其是长距离特征方面天然有缺陷，不是做不到而是做不好


[深度学习中的注意力机制(2017版)](https://blog.csdn.net/malefactor/article/details/78767781)

### GPT(Generative Pre-Training)生成式预训练

GPT 也采用两阶段过程，第一个阶段是利用语言模型进行预训练，第二阶段通过 Fine-tuning 的模式解决下游任务。 GPT 的预训练过程和 ELMO 是类似的，主要不同在于两点：
- 特征抽取器不是用的 RNN，而是用的 Transformer，上面提到过它的特征抽取能力要强于 RNN，这个选择很明显是很明智的；
- GPT 的预训练虽然仍然是以语言模型作为目标任务，但是采用的是单向的语言模型,只采用 Context-before 这个单词的上文来进行预测，而抛开了下文。这个选择现在看不是个太好的选择


GPT存在的问题？
- 要是把语言模型改造成双向的就好了
- 不太会炒作，GPT也是非常重要的工作



### Bert

Bert 采用和 GPT 完全相同的两阶段模型，首先是语言模型预训练；其次是使用 Fine-Tuning 模式解决下游任务。和 GPT 的最主要不同在于在预训练阶段采用了类似 ELMO 的双向语言模型，当然另外一点是语言模型的数据规模要比 GPT 大。第二阶段，Fine-Tuning 阶段，这个阶段的做法和 GPT 是一样的。当然，它也面临着下游任务网络结构改造的问题，在改造任务方面 Bert 和 GPT 有些不同，下面简单介绍一下。



- Bert这种两阶段的模式（预训练+Finetuning）必将成为NLP领域研究和工业应用的流行方法；
- 从NLP领域的特征抽取器角度来说，Transformer会逐步取代RNN成为最主流的的特征抽取器。

NLP四大类任务：
- 序列标注：分词、POS Tag、命名实体识别、语义标注...特点是句子中每个单词要求模型根据上下文都要给出一个分类类别
- 分类任务：文本分类、情感计算...特点是不管文章有多长，总体给出一个分类类别即可
- 句子关系判断：蕴含关系、问答、推理...特点是给定两个句子，模型判断出两个句子是否具备某种语义关系
- 生成式任务：机器翻译、文本摘要...特点是输入文本内容后，需要自主生成另外一段文字