---
layout:     post   				    # 使用的布局（不需要改）
title:      机器学习 / 词嵌入 Word Embedding				# 标题 
subtitle:      #副标题
date:       2018-09-11 				# 时间
author:     zhu.xinwei 		    	# 作者
header-img: img/post-bg-desk.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - Linear Regression
    - Logistic Regression
    - Universal approximation theorem
---

### 词频向量

### 词嵌入


### TF-IDF向量

Word Embedding, ELMo, Transformer, Attention, Bert

近年来，研究人员通过文本上下文信息分析获得更好的词向量。ELMo是其中的翘楚，在多个任务、多个数据集上都有显著的提升。所以，它是目前最好用的词向量，the-state-of-the-art的方法。这篇文章发表在2018年的NAACL上，outstanding paper award。下面就简单介绍一下这个“神秘”的词向量模型。


#### `基于统计的词嵌入方法`和`基于预测的词嵌入方法`（如Word2Vec）有什么区别，为什么深度学习中要使用后者
个人观点是，一个方面，基于统计的词嵌入方法一旦在数据确定之后，词的向量便就确定，在不同的NLP任务中都要使用相同的词向量，无法根据不同任务进行调整。而基于预测的词嵌入可以在不同的任务和模型中更新词向量，使词向量逐步地适应这项任务；另一方面则是基于统计的词嵌入方法提取的语义信息实在太少，无法取到比较好的结果。
