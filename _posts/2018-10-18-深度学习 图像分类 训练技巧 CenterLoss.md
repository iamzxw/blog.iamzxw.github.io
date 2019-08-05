---
layout:     post   				    # 使用的布局（不需要改）
title:     	深度学习 / 图像分类 / 各种Loss总结
subtitle:      #副标题
date:       2018-10-17 				# 时间
author:     zhu.xinwei 		    	# 作者
header-img: img/post-bg-desk.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - CenterLoss
    - TrepletLoss
    - 
---

### Softmax Loss


### Cross Entropy Loss


### CenterLoss

为了提升图像分类的性能，除了设计各种网络结构外，还可以从改进损失函数的角度入手。

center loss的原理主要是在softmax loss的基础上，通过对训练集的每个类别在特征空间分别维护一个类中心，在训练过程，增加样本经过网络映射后在特征空间与类中心的距离约束，从而兼顾了类内聚合与类间分离。

center loss意思即为：为每一个类别提供一个类别中心，最小化min-batch中每个样本与对应类别中心的距离，这样就可以达到缩小类内距离的目的。

![](/img/cnn/loss/center_loss_1.PNG)

同一类里的样本差异不是太大的情况下应该用CenterLoss效果应该会好.

在Cifar10和Cifar100上测试过Center Loss，发现效果并不是很好，准确率明显低于单独的Softmax；

在Mnist上测试加上Center Loss的Softmax好于单独的Softmax；

所以个人理解之所以Center Loss是针对人脸识别的Loss是有原因的，个人认为人脸的中心性更强一些，也就是说一个人的所有脸取平均值之后的人脸我们还是可以辨识是不是这个人，所以Center Loss才能发挥作用


### Triplet Loss


### Contrastive loss


[损失函数整理](https://zhuanlan.zhihu.com/p/35027284)