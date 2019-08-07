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

> CenterLoss是论文[A Discriminative Feature Learning Approach for Deep Face Recognition](https://kpzhang93.github.io/papers/eccv2016.pdf)提出的一种辅助训练loss，期望达到缩小类内距离的目的。论文被2016ECCV录用。


> In most of the available CNNs, the softmax loss function is used as the supervision signal to train the deep model. In order to enhance the discriminative power of the deeply learned features, this paper proposes a new supervision signal, called center loss, for face recognition task.

为了提升图像分类的性能，除了设计各种网络结构外，还可以从改进损失函数的角度入手。

center loss的原理主要是在softmax loss的基础上，通过对训练集的每个类别在特征空间分别维护一个类中心，在训练过程，增加样本经过网络映射后在特征空间与类中心的距离约束，从而兼顾了类内聚合与类间分离。

center loss意思即为：为每一个类别提供一个类别中心，最小化min-batch中每个样本与对应类别中心的距离，这样就可以达到缩小类内距离的目的。

![](/img/cnn/loss/center_loss_1.PNG)


**同一类里的样本差异不是太大的情况下应该用CenterLoss效果应该会好.**

在Cifar10和Cifar100上测试过Center Loss，发现效果并不是很好，准确率明显低于单独的Softmax；

在Mnist上测试加上Center Loss的Softmax好于单独的Softmax；

所以个人理解之所以Center Loss是针对人脸识别的Loss是有原因的，个人认为人脸的中心性更强一些，也就是说一个人的所有脸取平均值之后的人脸我们还是可以辨识是不是这个人，所以Center Loss才能发挥作用

```python
class CenterLoss(torch.nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        # 为每个类维护一个原形,即中心向量
        if self.use_gpu:
            self.centers = torch.nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = torch.nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        # print(x.shape)
        # print(labels.shape)

        batch_size = x.size(0)
        
        # distmat就是计算每个特征x与标签对应的center向量的均方差 (x - center)^2

        # 对特征x每个值求平均，然后相加x1^2 + x2^2 + ....,编程（batch_size, 1）
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        # distmat矩阵加上 x*centers ： β mat+α (mat1 * mat2)
        # distmat - 2x*centers [batch_size, feat]*[feat, num_classes]
        distmat.addmm_(1, -2, x, self.centers.t())

        # [0,1,2,3,...,365]
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        
        # classes.expand(batch_size, self.num_classes)
        # [0,1,2,3,...,365]
        # [0,1,2,3,...,365]
        # mask
        # [batch_size, num_classes]
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

```

### Triplet Loss


### Contrastive loss


[损失函数整理](https://zhuanlan.zhihu.com/p/35027284)