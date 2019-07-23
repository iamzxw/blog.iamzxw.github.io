---
layout:     post   				    # 使用的布局（不需要改）
title:      对抗样本论文集解读 (二) / Transferable Adversarial Perturbations

subtitle:      #副标题
date:       2019-03-28 				# 时间
author:     zhu.xinwei 		    	# 作者
header-img: img/post-bg-desk.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - 对抗样本
    - Translation-Invariant Attacks
---

![](/img/adversarial_examples/different_discriminative_regions.PNG)

最大化原始图片和对抗样本在卷积网络中的中间特征，可以提高白盒、黑盒的攻击成功率；
在目标函数中，加入平滑梯度正则项，可以提高黑盒攻击的迁移能力；