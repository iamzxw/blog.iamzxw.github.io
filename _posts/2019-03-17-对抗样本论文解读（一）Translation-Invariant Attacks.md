---
layout:     post   				    # 使用的布局（不需要改）
title:      对抗样本论文集解读 (三) / Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks

subtitle:   CVPR 2019 Oral   #副标题
date:       2019-04-22 				# 时间
author:     zhu.xinwei 		    	# 作者
header-img: img/post-bg-desk.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - 对抗样本
    - Translation-Invariant Attacks
---

![](/img/adversarial_examples/different_discriminative_regions.PNG)

正常训练的CNN模型，尽管模型架构不同，但是对于同一张图片类激活图class activation mapping关注的区域都是同一个地方。
不同的防御模型，对于同一张图片类激活图class activation mapping关注的区域各不相同。

传统的攻击方法，都是只攻击单张原始图像，所以生成的对抗样本很可能与CAM关注的区域和梯度相关。对于其他防御模型，它们的CAM关注区域和梯度不同，所以生成的对抗样本对防御模型无效。


为了减少对抗样本与CAM关注区域的相关性，作者提出一种平移不变形攻击（Translation-Invariant Attacks）方法。

传统攻击方法，在给定约束条件下s.t. ||x_adv - x_real|| < ε, 最大化损失函数 argmax J(x_adv, y)；损失函数只涉及一个对抗样本x_adv;
TI_ATTACK方法要求，生成的对抗样本x_adv, 不仅要使得J(x_adv, y)最大化，还要在对x_adv变换后的图像T(x_adv)最大化，即，最大化损失函数 argmax J(T(x_adv), y)；

这种变换函数T可以是rotation、scaling、translation等。如果采用梯度优化策略，传统攻击方法只需计算损失函数J()一张图片计算梯度，而TI方法需要计算损失函数J（）对所有变换后的图片T(x)计算梯度，计算量非常大。

作者采用平移变化的方法，因为他们可以一种有效的方法进行TI_attack，只需计算损失函数J（）对原始一张图片计算梯度，就可以完成攻击。

Are adversarial examples have local ...?


生成的对抗样本都具有斑纹形状
- Regional Homogeneity
- DCT 变换 *
- TI_FGSM  *
- ATN
- mini-batch
- Synthesizing

要么找到直接生成斑纹形状噪音的方法；
要么指出虽然这些攻击方法，攻击能力强，可迁移性强，但是违背了噪音imperceptible的初衷。

防御方法，加上超分辨率；