---
layout:     post   				    # 使用的布局（不需要改）
title:      ZSL论文集解读 (三) / A Generative Model For Zero Shot Learning Using Conditional Variational Autoencoders
subtitle:      #副标题
date:       2018-08-22 				# 时间
author:     zhu.xinwei 		    	# 作者
header-img: img/post-bg-desk.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - Linear Regression
    - Logistic Regression
    - Universal approximation theorem
---


研究背景

近几年，深度学习(Deep Learning)的火热主要归功于监督学习(Supervised Learning)的成功。在拥有大量标记数据和高性能计算平台(GPU)的情况下，监督学习在某些任务中可以达到(甚至超过)人类水平。如，借助Imagenet上的大量标记图片，各种CNN模型可以在图片分类任务中超过人类水平。

然而，在实际应用中，人工标记大量数据会有许多局限，如，人工成本，数据收集等问题。与人类能够识别的3万多种类别相比，Imagenet上具有大量标记图片的类别也只不过有几千种而已(其他两万多物体类别只有很少量的标记图片)。而且，与人类的学习能力相比，机器(如，CNN模型)显得有点太笨了~。你需要进行千万次级别的训练(一边又一遍的告诉它，这个是猫，那个是狗，还要举大量栗子，囧)，才能得到一个分类器。

所以，近两年，机器学习社区将研究重点转向了非监督学习(Unsupervised Learning)和半监督学习(Semi-supervised Learning) —— 在只有少量标记数据，甚至没有标记数据的情况下，如何训练一个有效的模型。

对于非监督学习任务，一种解决方案是采用生成模型(Generative Model)。在只有大量未标记数据的情况下，生成模型可以学得真实数据的隐含分布信息(True Data Latent Distribution)，并且生成与真实数据类似的生成数据。如，给定大量未标记的图片数据，生成模型可以生成一些与真实图片(Real Images)相似的生成图片(Generated Images)。

你可能会问，已经有真实、清晰的图片了，干嘛还要生成一些不太像的生成图片。额...我们好像是在解读ZSL论文，这个..这个..扯的有点远了，有时间我们在专门的Generative Model文章中探讨。

我们现在只需了解到，在生成模型领域，又分出好几类方法，如，生成对抗网络(Generative Adversarial Networks)，变分自编码器(Variational Auto-Encoder),  受限玻尔兹曼机(Restricted Boltzmann Machine)，自回归模型（Autoregressive models），常规流模型(Normalizing Flow-based Model)。本篇文章解读的论文使用是变分自编码器(VAE)。

另外多说一句，生成模型派系研究人员有一个终极目标：使用真实世界中的大量数据(图片，文本，语音，视频等等)训练一个生成模型，使得模型能够理解真实世界的数据分布，并能根据模型自己的'理解'生成对应的数据。进而使机器能够理解真实世界~，人类将会进入全新的AI时代。(既然是终极目标，那就应该是不可能实现的目标！！)

不过，现阶段，生成模型在许多短期应用( Short term Applications )中取得了非常好的效果。如，图像、视频的去燥(Denoising)、修复(Inpainting)，超分辨率成像(Super-resolution imaging)等。当然，我们今天要讲的零样本学习(Zero Shot Learning)也是其中之一。 

零样本图片分类(Zero Shot Image Recognition)是零样本学习(ZSL)的一个子领域。现阶段的ZSL研究几乎都焦距在图片分类任务上，所以可以将ZSL等同于ZSR(无奈~)。 简单来说，ZSL问题就是，在给定已知类别的训练集上，训练一个模型，使得模型不仅能够识别训练集中出现的类别(seen class)，而且还能识别一些训练集中未出现的类别(unseen class)。后期我们在专门的ZSL科普文章中再深入探讨。

## 引言
2013年，Diederik P Kingma等人在一篇名为Auto-Encoding Variational Bayes的论文中首次提出变分自编码器(Variational Auto-Encoder)模型。作为一类重要的无监督生成模型(Unsupervised Generative Model)，变分自编码器VAE在图像生成领域有着出色的性能。与自编码器(Auto-Encoder)不同之处在于，VAE在编码过程中增加了一些限制，迫使其生成的隐含向量(latent vector)粗略地遵循标准正态分布。这样我们只需给解码器一个服从标准正态分布的随机隐含向量，就能够得到对应的生成数据。

对VAE不太熟悉的同学可以参看OpenAI实习生Kevin Frans的一篇博文Variational Autoencoders Explained(感受一下被高中生碾压的心情)，中文参考花式解释AutoEncoder与VAE，还有苏老师的系列博文变分自编码器（一）：原来是这么一回事。

由于VAE只利用训练数据\left\{ x \right\} _{i=1}^n (没有对应的标签，也没有额外辅助信息y)进行模型训练，在生成数据时，模型无法生成某一特定条件下的数据。如，你只想生成猫(限制条件)的图片，VAE做不到。VAE能做的是，给定一个随机正态分布的隐含向量，VAE可以生成对应图片。但是这个生成图片可能是猫，也可能是桌子，或者是MNIST中0~9任意一个数字。

随后，2015年，Kihyuk Sohn等人提出一种改进的VAE —— 条件变分自编码器(Conditional Variational Auto Encoder)。简单来说，CVAE在训练数据\left\{ x \right\} _{i=1}^n 和辅助信息y(不一定是标签)上进行模型训练。在生成数据时，CVAE可以在给定限制条件(辅助信息y)下生成某一特定类型的数据。如果加入辅助信息y，CVAE就不是那么纯粹的无监督学习了，即，半监督学习(Semi-supervised Learning)。不过，它可以满足你的要求 —— 生成给定限制条件(猫)下的图片。CVAE模型图解参看下文。

根据CVAE模型的特性，我们很容易想到将CVAE应用到ZSL问题上。

针对ZSL问题，一般有两种解决方案。一种是基于映射的方法：在训练集上，学习语义空间(semantic space)和视觉空间的映射关系，然后将学得的映射函数应用(迁移)于测试集类别。对于这一类方法，提升映射函数的迁移能力很重要。我们在基于映射方法的相关论文中再做探讨。

另一种解决方案是基于生成模型的方法。在训练ZSL模型时，虽然unseen类别的数据不能使用，但是我们可以利用seen类别的数据训练一个生成模型，然后生成unseen类别的图片(或者图片特征)，将ZSL问题转化为了传统的监督学习分类问题。如何有效地生成unseen类别的图片，是这类方法的关键所在。上文中，我们有提到过各种主流的生成模型，如，GAN，VAE，RBM，Flow-based Model等。也有最新一些ZSL研究成果使用的是其他生成模型，特别是GAN。后期我们会陆续讲解，so just enjoy it ~。