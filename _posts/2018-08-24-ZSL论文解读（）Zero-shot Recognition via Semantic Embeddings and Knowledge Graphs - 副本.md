---
layout:     post   				    # 使用的布局（不需要改）
title:      ZSL论文集解读 (四) / Zero-shot Recognition via Semantic Embeddings and Knowledge Graphs
subtitle:      #副标题
date:       2018-08-22 				# 时间
author:     zhu.xinwei 		    	# 作者
header-img: img/post-bg-desk.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - Zero-shot Learning 
    - 图卷积网络
---


模型使用LeakyReLU作为激活函数，因为作者发现，对于回归任务，使用LeakyReLU收敛的更快。
> For activation functions, instead of using ReLU after each convolutional layer, we apply
LeakyReLU [27, 51] with the negative slope of 0.2. Empirically, we find that LeakyReLU leads to faster convergence for our regression problem.



“激活函数”能分成两类——“饱和激活函数”和“非饱和激活函数“

为什么需要激活函数


### Sigoid 函数

> sigmoid(x) = 1 / (1 + e^-x)

一个实值输入压缩至[0,1]的范围

sigmoid函数的图像

[]()

我们优化参数时会用到误差反向传播算法，需要对激活函数求导，得到sigmoid函数的瞬时变化率，其导数表达式为：
> sigmoid(x)(1 - sigmoid(x))

[]()

由图可知，导数从0开始很快就又趋近于0了，易造成“梯度消失”现象


### Tanh 函数

### ReLU激活函数

> relu(x) = max(0, x)

AlexNet中提出用ReLU来替代传统的激活函数(Sigmoid等)是深度学习的一大进步， 此后，深度学习中，我们一般使用ReLU作为中间隐层神经元的激活函数。

