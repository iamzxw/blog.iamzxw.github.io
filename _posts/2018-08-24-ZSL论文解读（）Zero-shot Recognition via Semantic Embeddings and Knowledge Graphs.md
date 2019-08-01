---
layout:     post   				    # 使用的布局（不需要改）
title:      ZSL论文集解读 (四) / Rethinking Knowledge Graph Propagation for Zero-Shot Learning
subtitle:      #副标题
date:       2019-07-22 				# 时间
author:     zhu.xinwei 		    	# 作者
header-img: img/post-bg-desk.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - Zero-shot Learning 
    - 图卷积网络
---

GCN 多层结构存在的问题。
> However, multi-layer architectures, which are required to propagate knowledge to distant nodes in the graph, dilute the knowledge by performing extensive Laplacian smoothing at each layer and thereby consequently decrease performance.

为了借助图结构信息的同时，防止知识减退
>In order to still enjoy the benefit brought by the graph structure while preventing dilution of knowledge from distant nodes, we propose a Dense Graph Propagation (DGP) module with carefully designed direct links among distant nodes