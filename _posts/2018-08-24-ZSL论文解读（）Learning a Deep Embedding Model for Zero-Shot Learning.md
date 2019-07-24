---
layout:     post   				    # 使用的布局（不需要改）
title:      ZSL论文集解读 (二) / Learning a Deep Embedding Model for Zero-Shot Learning Embedding
subtitle:      #副标题
date:       2018-09-26 				# 时间
author:     zhu.xinwei 		    	# 作者
header-img: img/post-bg-desk.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - Zero-shot Learning 
    - DEM
---

论文链接：https://arxiv.org/abs/1703.06870

开源代码：https://github.com/TuSimple/mx-maskrcnn

录用信息：CVPR2017


核心思想：

1. 使用RoIAlign代替RoIPooling，得到更好的定位效果。

2. 在Faster R-CNN基础上加上mask分支，增加相应loss，完成像素级分割任务。

缺点：


作者提出的一个核心观点是：视觉特征空间作为嵌入空间（embedding space）要比语义空间作为嵌入空间的效果好很多。而我第一次尝试的模型就是讲语义空间作为嵌入空间。

贡献：
- 提出了一种新的端对端的深度嵌入模型；
- 能够很自然地衍生到多模态的融合算法中；
- 在多个数据集上取得了最好的效果。

___
### 论文要点
零样本要做的是，学习一个共享**嵌入空间**（Joint Embedding Space）。 在这个空间里，语义表述（Semantic Representation）和视觉表述（Visual Representation）可以做**最近邻搜索**。
> Zero-shot learning (ZSL) models rely on learning a joint embedding space where both textual/semantic description of object classes and visual representation of object images can be projected to for nearest neighbour search.

作者认为，选择一个正确的**嵌入空间**是ZSL模型成功的关键。之前的相关工作都是将**语义空间**或者**中间共享空间**作为嵌入空间，然后再这个空间中做最近邻搜索。而本篇论文，作者提出使用视觉空间作为嵌入空间，效果会非常好。
> We argue that the key to the success of a deep embedding model for ZSL is the choice of the embedding space.


把视觉空间作为嵌入空间，会减少hubness问题。 由于嵌入空间是一个高维空间，所以很容易出现hubness problem。该问题是指：在高维空间中，一部分未见类别的原型prototypes可能会成为很多数据点的K近邻(KNN)，但其类别之间却没什么关系。
> However, since the embedding space is of high dimension and NN search is to be performed there, the hubness problem is inevitable, that is, a few unseen class prototypes will become the NNs of many data points, i.e. hubs.

当我们使用语义空间(semantic space)作为嵌入空间时，需要将视觉特征映射到语义空间中，这样会使得空间发生萎缩，点与点之间更加稠密，从而加重hubness problem。
> Using the semantic space as the embedding space means that the visual feature vectors need to be projected into the semantic space which will shrink the variance of the projected data points and thus aggravate the hubness problem

DEM模型的设计结构够很自然地衍生到多模态的融合算法中，将属性、词向量、语义描述拼接起来。
> Third, when multiple semantic spaces are available, this model can provide a natural mechanism for fusing the multiple modalities.


___
### 相关工作

#### 语义空间Semantic Space

使用单个语义模态
- 属性：效果最好，但是还是依靠人工标注
- 词向量：几乎完全free
- 文本标注：使用相对较少

融合多个模态：
- score-level fusion
- multi-view
DEM直接拼接起来；

#### 嵌入模型

第一类是直接学习一个映射函数，不过是将语义空间作为嵌入空间；
> The first group learns a mapping function by regression from the visual feature space to the semantic space with pre-computed features or deep neural network regression.

第二类是在一个公共中间空间隐式地学习视觉-语义映射；
> The second group of models implicitly learn the relationship between the visual and semantic space through a common intermediate space, again either with a neural network formulation or without.

DEM直接使用单层全连接层，如果使用多模态语义表示，可能需要tanh激活函数
```python

# 	........

	# Placeholder
    # define placeholder for inputs to network
    sementic_features = tf.placeholder(tf.float32, [None, 500])
    visual_features = tf.placeholder(tf.float32, [None, feat_size])
    # # Network
    W_left_w1 = weight_variable([500, feat_size])
    b_left_w1 = bias_variable([feat_size])
    left_w1 = tf.nn.relu(tf.matmul(sementic_features, W_left_w1) + b_left_w1)

    # # loss 
    loss_w = tf.reduce_mean(tf.square(left_w1 - visual_features))

    # L2 regularisation for the fully connected parameters.
    regularisers_w = (tf.nn.l2_loss(W_left_w1) + tf.nn.l2_loss(b_left_w1))

    # Add the regularisation term to the loss.
    loss_w += 1e-3 * regularisers_w

#   ........

```

#### Loss函数

- margin-based losses
- euclidean distance loss
- three training losses:  binary cross entropy loss, hinge loss and Euclidean distance loss

本文使用最小二乘损失，均方差
> In our model, we find that the least square loss between the two embedded vectors is very effective and offers an easy theoretical justification as for why it copes with the hubness problem better.

用一些高大上的名词：
多模态融合multi-modality fusion: 其实就是把不同的语义表示拼接起来；
端到端模型：也就是直接进行映射，可能更容易优化。



