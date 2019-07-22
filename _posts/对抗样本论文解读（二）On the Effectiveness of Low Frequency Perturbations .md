---
layout:     post   				    # 使用的布局（不需要改）
title:      对抗样本论文集解读 (二) / On the Effectiveness of Low Frequency Perturbations

subtitle:      #副标题
date:       2019-03-22 				# 时间
author:     zhu.xinwei 		    	# 作者
header-img: img/post-bg-desk.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - 对抗样本
    - Translation-Invariant Attacks
---

![](/img/adversarial_examples/different_discriminative_regions.PNG)

最近一些工作发现，将攻击空间限制在低频子空间会提高对抗样本的攻击成功率。目前，不清楚起作用的是哪里部分：
- 将攻击搜索空间限制在低频子空间内，有助于寻优
- 低频空间的噪音更具有攻击性

离散余弦变换Discrete cosine transform: DCT是从DFT推导出来的另一种变换。 大多数自然信号（声音、图像）的能量都集中在离散余弦变换后的低频部分，因而DCT在（声音、图像）数据压缩中得到了广泛的使用。