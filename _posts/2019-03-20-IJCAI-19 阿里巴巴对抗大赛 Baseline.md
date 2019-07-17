---
layout:     post
title:      【IJCAI-19 阿里巴巴人工智能对抗算法竞赛】 Baseline (MI_FGSM + Ensemble + Training DenseNet from scratch)
subtitle:      #副标题
date:       2019-3-20 				# 时间
author:     zhu.xinwei 		    	# 作者
header-img: img/post-bg-desk.jpg	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - 对抗样本
    - DenseNet
    - 
---

## IJCAI-19 阿里巴巴人工智能对抗算法竞赛

先来看一下使用本篇文章的Baseline可以达到的线上测评分数(online score)，与官方Baseline做个对比


| _ | Defense Track | Targeted Attack Track | Non-targeted Attack Track |
| -------- | -------- | -------- | -------- |
| Baseline(ours) |  **11.5665**    |   **92.6502**   |  **47.1709**    |
| Baseline(official) |   2.9525   |   126.6780   | 63.1448 |

从表中可以看出，使用本篇文章的Baseline，可以达到非常好的线上测评。

我们的解决方案，主要参考[NIPS 2017: Adversarial Attack and Defense Competition](https://www.kaggle.com/benhamner/adversarial-learning-challenges-getting-started)的Top 1 Submission。

来自清华大学的TSAIL团队，同时获得了NIPS 2017对抗攻防赛的三个赛道的三个冠军。赛后，他们不仅公开了竞赛代码，还撰写了两篇论文进行说明。下面连接可以直达TSAIL竞赛代码。  
[Non-Targeted-Adversarial-Attacks](https://github.com/dongyp13/Non-Targeted-Adversarial-Attacks)、[Targeted-Adversarial-Attack](https://github.com/dongyp13/Targeted-Adversarial-Attack)、[Guided-Denoise](https://github.com/lfz/Guided-Denoise)

___
## 科普一秒钟
2013年，Christian Szegedy等人在一篇名为[Intriguing properties of neural networks](https://arxiv.org/abs/1312.6199)的论文中首次提出**对抗样本**(Adversarial examples)的概念。简单来说，对抗样本就是在一张**正常图片**(Benign Image)上刻意加入一些人类不易察觉的**扰动**(Perturbation)，使得CNN分类器在对这个带有噪音图片(Perturbed image)进行分类时，出现非常离谱的错误。  

![]()

作为攻击方Attacker，我们的目标是想办法找到这些对抗样本....     


> 算了，我们还是直接看代码，跑数据吧....   
> 笔者QQ: `953831895` 欢迎交流^_^  


____
## Defense Track

| Defense Method | inception_v1 | resnet_v1_50 | vgg_16 |
| --------- | --------- | --------- | --------- |
| None | 2.9525 | 1.9176 | 3.4122 |

首先我们将官方给出的三个CNN网络权重部署到Defense Track，没有任何防御措施。从线上评测结果可以看出，线上5个基础攻击模型生成的550张图片，并没有将这三个CNN网络的分类Accuracy降为0。 

### 去燥(Denoising)

防御对抗样本最直观的方法就是去燥了，可以从pixel-level、feature-level等多个层面进行去燥。

比较简单的防御方法是进行Input Transformation C(t(x')) = C(x), 也就是在像素级别(pixel-level)进行图片去燥(Denoising)、变换(Transfomation)。方法有好多啦。

- Wavelet Denoising
- Median Filter
- Pixel Deflection
- JPEG Compression
- Image Quilting
- Total Variation Denoising
- Super Resolution
- Randomization
- ......

| Defense Method | inception_v1 | resnet_v1_50 | vgg_16 |
| --------- | --------- | --------- | --------- |
| None | 2.9525 | 1.9176 | 3.4122 |
| jpeg_0 | 4.9026 |  |  |
| jpeg_5 | 6.6222 | 4.6178 | 5.6355 |
| jpeg_15 | 4.1571 |  | 4.1126 |
| jpeg_35 | 3.3325 |  |  |
| jpeg_55 | 3.3325 |  |  |
| Wavelet Denoising |  |  |  |
| Pixel Deflection |  |  |  |
| Image Quilting |  |  |  |
| Total Variation Denoising |  |  |  |
| ... |  |  |  |

结果不尽如人意。所以，我们尝试训练了两个CNN模型，然后直接将训练好的权重部署到Defense Track，没有任何防御措施。  

| Defense Method | inception_v3(ours) | densenet121(ours) |
| --------- | --------- | --------- |
| None | 10.0888 | 11.5665 |


使用我们自己训练的CNN模型，线上分数有一个很大的提升。可以推测，线上5个基础攻击模型，对公开的三个网络权重进行了**百盒攻击**(White-box Attack)，所以直接部署三个公开CNN网络权重会有非常低的分数。而使用我们自己的CNN模型，由于5个基础攻击模型不可能访问到我们的网络权重，属于黑盒攻击(Black-box Attack)，攻击强度减弱，相应的防御分数会有很大幅度提升。

然后，我们在使用自己训练的CNN权重基础上，加入一些去燥操作，线上分数有了一些提高，但是提升效果并没有太多。


### 对抗训练(Adversarial Training)

**对抗训练**(Adversarial Training )，**集成对抗训练**(Ensemble Adversarial Training)，是另外一类防御对抗样本方法，C'(x') = C(x)。

如果在**集成攻击**(Ensemble Attack)方法中，加入这些对抗训练模型，生成的扰动图片会有更强的攻击能力。这类方法需要有很好的GPU支持，我们还在尝试中 ...   当然，我们也可以尝试使用TSAIL团队的Guided-Denoised解决方案。

下面是我们训练CNN的代码(pytorch version), 仅供参考。

### Dependencies

- python 3.6
- pytorch 0.4.1
- torchvision
- scikit-learn
- progressbar
- pillow
- pandas

### densenet.py：DenseNet

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()
        # Special attributs
        self.input_space = None
        self.input_size = (224 , 224, 3)
        self.mean = None
        self.std = None

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.last_linear = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def logits(self, features):
        x = F.relu(features, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7, stride=1)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    def get_features(self, input):
        x = self.features(input)
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7, stride=1)
        x = x.view(x.size(0), -1)
        return x

def densenet121(pretrained=False, **kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    return model

def densenet169(pretrained=False, **kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     **kwargs)
    return model

def densenet201(pretrained=False, **kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                     **kwargs)
    return model

def densenet161(pretrained=False, **kwargs):
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),
                     **kwargs)
    return model




### Note :

官方给出的训练图片有些无法读取，在开始训练之前，需要对训练数据进行预处理。

首先删除两张损坏图片: `/00012/1e19e7fa7da1641e786b69dc8eed9daa.jpg`,`/00092/bdc7be7063d7e99953bbaee2cc99888c.jpg`。  

然后运行下面代码，使用PIL.Image.convert('RGB')循环处理一下整个训练集，大概需要十几分钟。遍历过程会出现一些警告，不用理会，训练CNN模型时，使用处理过后的图片即可(保存在`IJCAI_2019_AAAC_train_processed`目录下)

import os
import glob
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from progressbar import *

class ImageSet_preprocess(Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, item):
        image_path = self.df.iloc[item]['image_path']
        image = Image.open(image_path).convert('RGB')
        image_path = image_path.replace('/IJCAI_2019_AAAC_train/', '/IJCAI_2019_AAAC_train_processed/')
        _dir, _filename = os.path.split(image_path)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        image.save(image_path)
        return image_path
    
def load_data_jpeg_compression(batch_size=16):
    all_imgs = glob.glob('/path/to/IJCAI_2019/IJCAI_2019_AAAC_train/*/*.jpg')
    train_data = pd.DataFrame({'image_path':all_imgs})
    datasets = {
        'train_data': ImageSet_preprocess(train_data),
    }
    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       num_workers=8,
                       shuffle=False) for ds in datasets.keys()
    }
    return dataloaders

if __name__ == '__main__':

#     dataloader = load_data_jpeg_compression()
#     widgets = ['jpeg :',Percentage(), ' ', Bar('#'),' ', Timer(),
#        ' ', ETA(), ' ', FileTransferSpeed()]
#     pbar = ProgressBar(widgets=widgets)
#     for batch_data in pbar(dataloader['train_data']):
#         pass
    pass




____
## Non-targeted Attack Track

![](http://jupter-oss.oss-cn-hangzhou.aliyuncs.com/public/files/image/1095279296084/1553051754641_Fb8Qe8fo2g.jpg)


直接抛出结论。   

针对**百盒模型**（white-box），FGSM具有较弱的**攻击能力**(attack strength)。但是使用FGSM生成的扰动图片具有较好的**可转移性**(Transferability)，即，可以使用FGSM针对A模型进行百盒攻击，生成的扰动图片有很大概率也会将B模型出错。  

基于迭代的攻击方法，如，I-FGSM，正好相反，针对百盒模型具有非常强的攻击能力，生成的扰动图片往往可以将被攻击模型的分类正确率降维0。不过，由于I-FGSM进行多次梯度计算，可能会'过拟合'被攻击模型的参数，从而使得生成的扰动图片具有很差的可转移性，即对**黑盒模型**(black-box)具有很差的攻击能力。

TSAIL提出的动量迭代FGSM(Momentum-Iterative FGSM)，既具有FGSM的可转移性，又具有I-FGSM的攻击能力。MI_FGSM针对百盒模型生成的扰动图片，不仅可以将百盒模型的分类正确率降到最低，同时可以将无法访问到的黑盒模型出现很大错误。

另外，如果我们同时攻击多个百盒模型，生成的饶动图片可以同时将所有被攻击的百盒模型分类正确率降到最低。这些扰动图片很可能学习到了一些通用的对抗扰动，可转移性会有很大提升。这类方法称为集成攻击策略(Ensemble Attack)。

| Attack Method | online score | 
| -------------- | -------------- | 
| inception_v1_fsgm_32 | 63.1448 |
| inception_v1_mi_fgsm_eps_32 | 50.5378 |
| ensemble_3_mi_fgsm_eps_32 | 47.1709 |

我们尝试了使用Mi_FGSM 对inception_v1进行百盒攻击，相较于官方Baseline使用FGSM针对inception_v1进行攻击的方法, 线上评分有了13个提升点。

而后，我们又尝试使用MI_FGSM针对inception_v1、resnet_v1_50、vgg_16同时进行百盒攻击，相较于单独攻击inception_v1，线上评分提上了3个点，似乎没有提升太多。。。


### non_target_mi_fgsm_attack.py

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.stats as st
from PIL import Image
from scipy.misc import imread
from scipy.misc import imresize
from tensorflow.contrib.slim.nets import resnet_v1, inception, vgg
slim = tf.contrib.slim

# 声明一些攻击参数
CHECKPOINTS_DIR = './data/checkpoints/'
model_checkpoint_map = {
    'inception_v1': os.path.join(CHECKPOINTS_DIR,'inception_v1', 'inception_v1.ckpt'),
    'resnet_v1_50': os.path.join(CHECKPOINTS_DIR, 'resnet_v1_50','model.ckpt-49800'),
    'vgg_16': os.path.join(CHECKPOINTS_DIR, 'vgg_16', 'vgg_16.ckpt')}

input_dir = ''
output = ''
max_epsilon = 32.0
num_iter = 20
batch_size = 11
momentum = 1.0

# 在图片数据输入模型前，做一些预处理
def preprocess_for_model(images, model_type):
    if 'inception' in model_type.lower():
        images = tf.image.resize_bilinear(images, [224,224],align_corners=False)
        # tensor-scalar operation
        images = (images / 255.0) * 2.0 - 1.0
        return images

    if 'resnet' in model_type.lower() or 'vgg' in model_type.lower():
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        images = tf.image.resize_bilinear(images, [224,224],align_corners=False)
        tmp_0 = images[:,:,:,0] - _R_MEAN
        tmp_1 = images[:,:,:,1] - _G_MEAN
        tmp_2 = images[:,:,:,2] - _B_MEAN
        images = tf.stack([tmp_0,tmp_1,tmp_2],3)
        return images
# 加载评测图片
def load_images_with_true_label(input_dir):
    images = []
    filenames = []
    true_labels = []
    idx = 0
    dev = pd.read_csv(os.path.join(input_dir, 'dev.csv'))
    filename2label = {dev.iloc[i]['filename'] : dev.iloc[i]['trueLabel'] for i in range(len(dev))}
    for filename in filename2label.keys():
        image = imread(os.path.join(input_dir, filename), mode='RGB')
        images.append(image)
        filenames.append(filename)
        true_labels.append(filename2label[filename])
        idx += 1
        if idx == 11:
            images = np.array(images)
            yield filenames, images, true_labels
            filenames = []
            images = []
            true_labels = []
            idx = 0
    if idx > 0:
        images = np.array(images)
        yield filenames, images, true_labels

def save_images(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        image = (((images[i] + 1.0) * 0.5) * 255.0).astype(np.uint8)
        # resize back to [299, 299]
        image = imresize(image, [299, 299])
        Image.fromarray(image).save(os.path.join(output_dir, filename), format='PNG')

def check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        
# 定义MI_FGSM迭代攻击的计算图
def non_target_graph(x, y, i, x_max, x_min, grad):

  eps = 2.0 * max_epsilon / 255.0
  alpha = eps / num_iter
  num_classes = 110

  with slim.arg_scope(inception.inception_v1_arg_scope()):
    logits_inc_v1, end_points_inc_v1 = inception.inception_v1(
      x, num_classes=num_classes, is_training=False, scope='InceptionV1')

  # rescale pixle range from [-1, 1] to [0, 255] for resnet_v1 and vgg's input
  image = (((x + 1.0) * 0.5) * 255.0)
  processed_imgs_res_v1_50 = preprocess_for_model(image, 'resnet_v1_50')
  with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    logits_res_v1_50, end_points_res_v1_50 = resnet_v1.resnet_v1_50(
      processed_imgs_res_v1_50, num_classes=num_classes, is_training=False, scope='resnet_v1_50')

  end_points_res_v1_50['logits'] = tf.squeeze(end_points_res_v1_50['resnet_v1_50/logits'], [1, 2])
  end_points_res_v1_50['probs'] = tf.nn.softmax(end_points_res_v1_50['logits'])

  # image = (((x + 1.0) * 0.5) * 255.0)#.astype(np.uint8)
  processed_imgs_vgg_16 = preprocess_for_model(image, 'vgg_16')
  with slim.arg_scope(vgg.vgg_arg_scope()):
    logits_vgg_16, end_points_vgg_16 = vgg.vgg_16(
      processed_imgs_vgg_16, num_classes=num_classes, is_training=False, scope='vgg_16')

  end_points_vgg_16['logits'] = end_points_vgg_16['vgg_16/fc8']
  end_points_vgg_16['probs'] = tf.nn.softmax(end_points_vgg_16['logits'])

  ########################
  # Using model predictions as ground truth to avoid label leaking
  pred = tf.argmax(end_points_inc_v1['Predictions'] + end_points_res_v1_50['probs'] + end_points_vgg_16['probs'], 1)
  first_round = tf.cast(tf.equal(i, 0), tf.int64)
  y = first_round * pred + (1 - first_round) * y
  one_hot = tf.one_hot(y, num_classes)
  ########################
  logits = (end_points_inc_v1['Logits'] + end_points_res_v1_50['logits'] + end_points_vgg_16['logits']) / 3.0
  cross_entropy = tf.losses.softmax_cross_entropy(one_hot,
                                                  logits,
                                                  label_smoothing=0.0,
                                                  weights=1.0)
  noise = tf.gradients(cross_entropy, x)[0]
  noise = noise / tf.reduce_mean(tf.abs(noise), [1,2,3], keep_dims=True)
  noise = momentum * grad + noise
  x = x + alpha * tf.sign(noise)
  x = tf.clip_by_value(x, x_min, x_max)
  i = tf.add(i, 1)
  return x, y, i, x_max, x_min, noise

def stop(x, y, i, x_max, x_min, grad):
  return tf.less(i, num_iter)

# Momentum Iterative FGSM
def non_target_mi_fgsm_attack(input_dir, output_dir):

  # some parameter
  eps = 2.0 * max_epsilon / 255.0
  batch_shape = [batch_size, 224, 224, 3]

  _check_or_create_dir(output_dir)

  with tf.Graph().as_default():
    # Prepare graph
    raw_inputs = tf.placeholder(tf.uint8, shape=[None, 299, 299, 3])

    # preprocessing for model input,
    # note that images for all classifier will be normalized to be in [-1, 1]
    processed_imgs = preprocess_for_model(raw_inputs, 'inception_v1')

    x_input = tf.placeholder(tf.float32, shape=batch_shape)
    x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
    x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

    y = tf.constant(np.zeros([batch_size]), tf.int64)
    # y = tf.placeholder(tf.int32, shape=[batch_size])
    i = tf.constant(0)
    grad = tf.zeros(shape=batch_shape)
    x_adv, _, _, _, _, _ = tf.while_loop(stop, non_target_graph, [x_input, y, i, x_max, x_min, grad])

    # Run computation
    s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV1'))
    s2 = tf.train.Saver(slim.get_model_variables(scope='resnet_v1_50'))
    s3 = tf.train.Saver(slim.get_model_variables(scope='vgg_16'))

    with tf.Session() as sess:
      s1.restore(sess, model_checkpoint_map['inception_v1'])
      s2.restore(sess, model_checkpoint_map['resnet_v1_50'])
      s3.restore(sess, model_checkpoint_map['vgg_16'])

      for filenames, raw_images, true_labels in load_images_with_true_label(input_dir):
        processed_imgs_ = sess.run(processed_imgs, feed_dict={raw_inputs: raw_images})
        adv_images = sess.run(x_adv, feed_dict={x_input: processed_imgs_})
        save_images(adv_images, filenames, output_dir)
        
if __name__=='__main__':
#     input_dir = '/path/to/dev_data'
#     output_dir = '/path/to/output'
#     non_target_mi_fgsm_attack(input_dir, output_dir)
    pass



## Targeted Attack Track

与Non-targeted Attack不同的是，Targeted Attack基本没有什么可转移性。即，Targeted Attack生成的扰动图片只对被攻击的百盒模型有效，对黑盒模型几乎没有太大影响。 

TSAIL团队采用的策略是，针对常见的几个CNN模型进行百盒攻击，不追求生成的扰动图片具有较强的可转移性。

| Attack Method | online score |
| -------------- | -------------- |
| inception_v1_mi_fgsm_eps_32 | 126.6780 |
| ensemble_3_mi_fgsm_eps_32_iter_30 | 98.8626 |
| ensemble_3_mi_fgsm_eps_32_iter_15 | 92.6502 |

官方Baseline使用MI_FGSM方法针对inception_v1权重进行百盒攻击，线上测评分数为126.6780。

我们尝试集成攻击策略，使用MI_FGSM方法同时攻击inception_v1, resnet_v1_50, vgg_16三个网络权重，并设置了不同的迭代参数。从表格中，我们可以看出，迭代次数为15的情况下，效果最好。


### target_mi_fgsm_attack.py

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.stats as st
from PIL import Image
from scipy.misc import imread
from scipy.misc import imresize
from tensorflow.contrib.slim.nets import resnet_v1, inception, vgg
slim = tf.contrib.slim

# 声明一些攻击参数
CHECKPOINTS_DIR = './data/checkpoints/'
model_checkpoint_map = {
    'inception_v1': os.path.join(CHECKPOINTS_DIR,'inception_v1', 'inception_v1.ckpt'),
    'resnet_v1_50': os.path.join(CHECKPOINTS_DIR, 'resnet_v1_50','model.ckpt-49800'),
    'vgg_16': os.path.join(CHECKPOINTS_DIR, 'vgg_16', 'vgg_16.ckpt')}

input_dir = ''
output = ''
max_epsilon = 32.0
num_iter = 20
batch_size = 11
momentum = 1.0

# 在图片数据输入模型前，做一些预处理
def preprocess_for_model(images, model_type):
    if 'inception' in model_type.lower():
        images = tf.image.resize_bilinear(images, [224,224],align_corners=False)
        # tensor-scalar operation
        images = (images / 255.0) * 2.0 - 1.0
        return images

    if 'resnet' in model_type.lower() or 'vgg' in model_type.lower():
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        images = tf.image.resize_bilinear(images, [224,224],align_corners=False)
        tmp_0 = images[:,:,:,0] - _R_MEAN
        tmp_1 = images[:,:,:,1] - _G_MEAN
        tmp_2 = images[:,:,:,2] - _B_MEAN
        images = tf.stack([tmp_0,tmp_1,tmp_2],3)
        return images
    
def load_images_with_target_label(input_dir):
    images = []
    filenames = []
    target_labels = []
    idx = 0

    dev = pd.read_csv(os.path.join(input_dir, 'dev.csv'))
    filename2label = {dev.iloc[i]['filename'] : dev.iloc[i]['targetedLabel'] for i in range(len(dev))}
    for filename in filename2label.keys():
        image = imread(os.path.join(input_dir, filename), mode='RGB')
        images.append(image)
        filenames.append(filename)
        target_labels.append(filename2label[filename])
        idx += 1
        if idx == 11:
            images = np.array(images)
            yield filenames, images, target_labels
            filenames = []
            images = []
            target_labels = []
            idx = 0
    if idx > 0:
        images = np.array(images)
        yield filenames, images, target_labels

def save_images(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        image = (((images[i] + 1.0) * 0.5) * 255.0).astype(np.uint8)
        # resize back to [299, 299]
        image = imresize(image, [299, 299])
        Image.fromarray(image).save(os.path.join(output_dir, filename), format='PNG')

def check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def target_graph(x, y, i, x_max, x_min, grad):

  eps = 2.0 * max_epsilon / 255.0
  alpha = eps / num_iter
  num_classes = 110

  with slim.arg_scope(inception.inception_v1_arg_scope()):
    logits_inc_v1, end_points_inc_v1 = inception.inception_v1(
      x, num_classes=num_classes, is_training=False, scope='InceptionV1')

  # rescale pixle range from [-1, 1] to [0, 255] for resnet_v1 and vgg's input
  image = (((x + 1.0) * 0.5) * 255.0)
  processed_imgs_res_v1_50 = preprocess_for_model(image, 'resnet_v1_50')
  with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    logits_res_v1_50, end_points_res_v1_50 = resnet_v1.resnet_v1_50(
      processed_imgs_res_v1_50, num_classes=num_classes, is_training=False, scope='resnet_v1_50')

  end_points_res_v1_50['logits'] = tf.squeeze(end_points_res_v1_50['resnet_v1_50/logits'], [1, 2])
  end_points_res_v1_50['probs'] = tf.nn.softmax(end_points_res_v1_50['logits'])

  # image = (((x + 1.0) * 0.5) * 255.0)#.astype(np.uint8)
  processed_imgs_vgg_16 = preprocess_for_model(image, 'vgg_16')
  with slim.arg_scope(vgg.vgg_arg_scope()):
    logits_vgg_16, end_points_vgg_16 = vgg.vgg_16(
      processed_imgs_vgg_16, num_classes=num_classes, is_training=False, scope='vgg_16')

  end_points_vgg_16['logits'] = end_points_vgg_16['vgg_16/fc8']
  end_points_vgg_16['probs'] = tf.nn.softmax(end_points_vgg_16['logits'])

  ########################
  one_hot = tf.one_hot(y, num_classes)
  ########################
  logits = (end_points_inc_v1['Logits'] + end_points_res_v1_50['logits'] + end_points_vgg_16['logits']) / 3.0
  cross_entropy = tf.losses.softmax_cross_entropy(one_hot,
                                                  logits,
                                                  label_smoothing=0.0,
                                                  weights=1.0)
  noise = tf.gradients(cross_entropy, x)[0]
  noise = noise / tf.reshape(
    tf.contrib.keras.backend.std(tf.reshape(noise, [batch_size, -1]), axis=1),
      [batch_size, 1, 1, 1])
  noise = momentum * grad + noise
  noise = noise / tf.reshape(
    tf.contrib.keras.backend.std(tf.reshape(noise, [batch_size, -1]), axis=1),
    [batch_size, 1, 1, 1])
  x = x - alpha * tf.clip_by_value(tf.round(noise), -2, 2)
  x = tf.clip_by_value(x, x_min, x_max)
  i = tf.add(i, 1)
  return x, y, i, x_max, x_min, noise

def stop(x, y, i, x_max, x_min, grad):
  return tf.less(i, num_iter)

# Momentum Iterative FGSM
def target_mi_fgsm_attack(input_dir, output_dir):

  # some parameter
  eps = 2.0 * max_epsilon / 255.0
  batch_shape = [batch_size, 224, 224, 3]

  _check_or_create_dir(output_dir)

  with tf.Graph().as_default():
    # Prepare graph
    raw_inputs = tf.placeholder(tf.uint8, shape=[None, 299, 299, 3])

    # preprocessing for model input,
    # note that images for all classifier will be normalized to be in [-1, 1]
    processed_imgs = preprocess_for_model(raw_inputs, 'inception_v1')

    x_input = tf.placeholder(tf.float32, shape=batch_shape)
    x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
    x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

#     y = tf.constant(np.zeros([batch_size]), tf.int64)
    y = tf.placeholder(tf.int32, shape=[batch_size])
    i = tf.constant(0)
    grad = tf.zeros(shape=batch_shape)
    x_adv, _, _, _, _, _ = tf.while_loop(stop, target_graph, [x_input, y, i, x_max, x_min, grad])

    # Run computation
    s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV1'))
    s2 = tf.train.Saver(slim.get_model_variables(scope='resnet_v1_50'))
    s3 = tf.train.Saver(slim.get_model_variables(scope='vgg_16'))

    with tf.Session() as sess:
      s1.restore(sess, model_checkpoint_map['inception_v1'])
      s2.restore(sess, model_checkpoint_map['resnet_v1_50'])
      s3.restore(sess, model_checkpoint_map['vgg_16'])

      for filenames, raw_images, target_labels in load_images_with_target_label(input_dir):
        processed_imgs_ = sess.run(processed_imgs, feed_dict={raw_inputs: raw_images})
        adv_images = sess.run(x_adv, feed_dict={x_input: processed_imgs_, y: target_labels})
        save_images(adv_images, filenames, output_dir)
        
if __name__ == '__main__':
#     input_dir = '/path/to/dev_data'
#     output_dir = '/path/to/output'
#     target_mi_fgsm_attack(input_dir, output_dir)
    pass