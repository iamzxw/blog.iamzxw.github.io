---
layout:     post   				    # 使用的布局（不需要改）
title:      ZSL论文集解读 (一) / Semantically Consistent Regularization for Zero-Shot Recognition
subtitle:   CVPR 2017   #副标题
date:       2018-08-22 				# 时间
author:     zhu.xinwei 		    	# 作者
header-img: img/post-bg-desk.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - Zero-shot Learning 
    - 图卷积网络
---

ZSL早期工作：致力于找到更好的语义表示
> Historically, early efforts were devoted to the identification of good semantics for ZSL

随后工作：致力于设计更好的语义映射; 作者分为两个大类；
- **使用独立语义RIS**：直接预测每个属性，多个SVM或一个多分类模型，监督信号是属性
- **使用语义空间RULE**：
> Subsequent works addressed the design of the semantic space S, using one of two strategies previously proposed in the semantic representation literature.
> - The first, recognition using independent semantics (RIS), consists of learning an independent
classifier per semantic
> - This motivated a shift to the second strategy, which ties the design of S to the goal of recognition, by learning a single multi-class classifier that optimally discriminates between all training classes

通过分析这两类方法的利弊，作者提出了一种基于深度网络，并结合RIS和RULE的方法，得到了在多个数据集上最好的结果。
> While some learn semantics independently, others only supervise the semantic subspace explained by training classes. Thus, the former is able to constrain the whole space but lacks the ability to model semantic correlations. The latter addresses this issue but leaves part of the semantic space unsupervised

**使用独立语义RIS**：在ZSL领域，这是最早采用的方法。简单来说就是对每个属性进行独立地学习，**需要独立训练多个SVM模型，或者一个多分类CNN模型**，属性与属性之间没有任何的关系。这个方法的代表就是ZSL领域的开山方法DAP[1], 为每个语义属性学一个SVM模型。
> One of the most popular among these is the direct attribute prediction (DAP) method [31], which learns attributes independently using SVMs and infers ZS predictions by a maximum a posteriori rule that assumes attribute independence.

**使用语义空间RULE**：为了使得定义的属性能够有更加丰富的表达，出现了RULE的方法，它的核心思想是学习一个语义属性空间，**只需训练一个多分类模型**，将类别和样本都映射到这个语义属性空间中，这使得事先定义属性向量在语义空间中有了相互关系，即距离越近的属性向量之间的相似度越大，这样能够使得属性向量的表达能够更强。
> RULE is an alternative strategy that exploits the one-toone relationship between semantics and object classes. The central idea is to define an embedding φ(·) that maps each class y into a Q-dimensional vector of attribute states φ(y) that identifies it.

**使用独立语义RIS**能够对每一个样本到每一种属性向量的映射进行监督学习，但它没有对属性之间的关系进行建模；**使用语义空间RULE**能够对属性之间的关系进行建模，但是，它对于建立的语义空间往往约束不够，会使得semantic domain shift（ZSL中最常见的共同问题）的问题更加严重
> On one hand, RIS supervises all attributes but cannot model their dependencies. On the other, RULE models dependencies but leaves a large number of dimensions of A unconstrained.

需要注意一点： 标签和属性的映射是固定的，不需要学习。
> Note that φ(y) is a fixed mapping from the space of attributes to the space of class labels
___
#### SCoRe模型


问题：
在DEM论文中指出，


需要梳理一下ZSL发展时间线：

ZSL早期工作：致力于找到更好的语义表示
> Historically, early efforts were devoted to the identification of good semantics for ZSL

随后工作：致力于设计更好的语义嵌入空间; 作者分为两个大类；
- 为每个属性学习一个分类器
- 
> Subsequent works addressed the design of the semantic space S, using one of two strategies previously proposed in the semantic representation literature.
> - The first, recognition using independent semantics (RIS), consists of learning an independent
classifier per semantic
> - This motivated a shift to the second strategy, which ties the design of S to the goal of recognition, by learning a single multi-class classifier that optimally discriminates between all training classes


对抗样本发展时间线：

分类模型发展时间线：