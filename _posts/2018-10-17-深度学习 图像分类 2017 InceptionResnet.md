---
layout:     post   				    # 使用的布局（不需要改）
title:     	深度学习 / 图像分类 / Inception ResNet 论文笔记
subtitle:      #副标题
date:       2018-9-25 				# 时间
author:     zhu.xinwei 		    	# 作者
header-img: img/post-bg-desk.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - 图像分类s
    - Inception v4
    - Inception ResNet v1
    - Inception ResNet v2
---

> 2016年，在论文[Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/pdf/1602.07261.pdf)中，作者Szegedy对Inception网络做了进一步改进，提出了inception v4, inception resnet v1 v2等网络结构。

### Inception-V4

Inception-v4是对原来的版本进行了梳理，因为原始模型是采用分区方式训练，而迁移到TensorFlow框架后可以对Inception模块进行一定的规范和简化。Inception-v4整体结构下图所示。

![](/img/cnn/inception_4_all.PNG)

```python

	# Modules
    self.features = nn.Sequential(
        
        # Stem 
        BasicConv2d(3, 32, kernel_size=3, stride=2),
        BasicConv2d(32, 32, kernel_size=3, stride=1),
        BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
        Mixed_3a(),
        Mixed_4a(),
        Mixed_5a(),
       	
       	# Inception 堆叠
       	Inception_A(),
        Inception_A(),
        Inception_A(),
        Inception_A(),
        Reduction_A(), # Mixed_6a

        Inception_B(),
        Inception_B(),
        Inception_B(),
        Inception_B(),
        Inception_B(),
        Inception_B(),
        Inception_B(),
        Reduction_B(), # Mixed_7a
        
        Inception_C(),
        Inception_C(),
        Inception_C()
    )
    # ...
    # 收尾部分 并没有使用Dropout
    x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
    x = x.view(x.size(0), -1)
    x = self.last_linear(x) 
    # self.last_linear = nn.Linear(1536, num_classes)

```

网络的输入是299x299大小。在使用Inception模块之前，有一个stem模块，这个模块在Inception-v3网络也是存在的，它将输出35x35大小的特征图。

![](inception_4_stem.PNG)

```python
	# Modules
   	self.features = nn.Sequential(
		BasicConv2d(3, 32, kernel_size=3, stride=2),
        BasicConv2d(32, 32, kernel_size=3, stride=1),
        BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
        Mixed_3a(),
        Mixed_4a(),
        Mixed_5a(),

```

Inception-v4中的Inception模块分成3组，基本上与Inception-v3网络是一致的，但有细微的变化，如下图所示：

![](/img/cnn/inception_4_A.PNG)

```python

class Inception_A(nn.Module):

    def __init__(self):
        super(Inception_A, self).__init__()
        self.branch0 = BasicConv2d(384, 96, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(384, 96, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


```

![](/img/cnn/inception_4_B.PNG)

```python

class Inception_B(nn.Module):

    def __init__(self):
        super(Inception_B, self).__init__()
        self.branch0 = BasicConv2d(1024, 384, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(224, 256, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(192, 224, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(224, 224, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(224, 256, kernel_size=(1,7), stride=1, padding=(0,3))
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(1024, 128, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


```


![](/img/cnn/inception_4_C.PNG)

```python

class Inception_C(nn.Module):

    def __init__(self):
        super(Inception_C, self).__init__()

        self.branch0 = BasicConv2d(1536, 256, kernel_size=1, stride=1)

        self.branch1_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch1_1a = BasicConv2d(384, 256, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch1_1b = BasicConv2d(384, 256, kernel_size=(3,1), stride=1, padding=(1,0))

        self.branch2_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch2_1 = BasicConv2d(384, 448, kernel_size=(3,1), stride=1, padding=(1,0))
        self.branch2_2 = BasicConv2d(448, 512, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch2_3a = BasicConv2d(512, 256, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch2_3b = BasicConv2d(512, 256, kernel_size=(3,1), stride=1, padding=(1,0))

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(1536, 256, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)

        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)

        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)

        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        return out


```


### Inception ResNet

Inception-ResNet网络是在Inception模块中引入ResNet的残差结构。

Inception-ResNet-v1的Inception模块如图所示，与原始Inception模块对比，增加shortcut结构，而且在add之前使用了线性的1x1卷积对齐维度。对于Inception-ResNet-v2模型，与v1比较类似，只是参数设置不同。

![](/img/cnn/inception_resnet_a_block.PNG)

```python
class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super(Block35, self).__init__()
        self.scale = scale

        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1),
            BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out
```
![](/img/cnn/inception_resnet_b_block.PNG)

```python

class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super(Block17, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 160, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(160, 192, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out

```
![](/img/cnn/inception_resnet_c_block.PNG)

```python

class Block8(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(2080, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=(1,3), stride=1, padding=(0,1)),
            BasicConv2d(224, 256, kernel_size=(3,1), stride=1, padding=(1,0))
        )

        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out

```

Inception-ResNet-v1对标Inception-v3，两者计算复杂度类似，而Inception-ResNet-v2对标Inception-v4，两者计算复杂度类似。Inception-ResNet网络结构如下图，整体架构与Inception类似。

![](/img/cnn/inception_resnet_all.PNG)

不同Inception网络的在ImageNet上的对比结果如下表所示，可以看到加入残差结构，并没有很明显地提升模型效果。但是作者发现残差结构有助于加速收敛。所以作者说没有残差结构照样可以训练出很深的网络。

![](/img/cnn/inception_resnet_table.PNG)


下图分别是Inception-ResNet-v1和Inception-ResNet-v2网络的stem模块结构，也即是Inception-v3和Inception-v4网络的stem模块。

![](/img/cnn/inception_resnet_stem.PNG)
![](/img/cnn/inception_4_stem.PNG)

```python
	# Inception-ResNet-v1 inception v3 Stem
	self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
	self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
	self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
	self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
	self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)

	# Inception-ResNet-v2 inception v4 Stem
    self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
    self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
    self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
    self.maxpool_3a = nn.MaxPool2d(3, stride=2)
    self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
    self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
    self.maxpool_5a = nn.MaxPool2d(3, stride=2)
```



