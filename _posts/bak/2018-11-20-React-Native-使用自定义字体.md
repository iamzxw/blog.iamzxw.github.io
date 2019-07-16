---
layout:     post   				    # 使用的布局（不需要改）
title: React Native 使用自定义字体
subtitle:      #副标题
date:       2018-11-20				# 时间
author:     liangping 						# 作者
header-img: img/react.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - React
---

## 使用

关于如何在 `react native` 引入自定义字体其实已经有很多的教程，比如 [React Native Custom Fonts](https://medium.com/react-native-training/react-native-custom-fonts-ccc9aacf9e5e) 讲的就比较明白了。

总结一下就是以下四步：

1. 将字体引入项目中，比如放在 `src/static/fonts` 目录下；

2. 在 `package.json` 文件中插入下面这段代码：

   ```json
   "rnpm": {
       "assets": [
   	".src/static/fonts/"
       ]
   }
   ```

3. 使用 `react-native link` 命令将字体 link 到 iOS 项目和 Android 项目中；

4. 检查 iOS 项目中的 `info.plist` 文件和 Android 的 `/app/src/main/assets/fonts` 文件夹下有没有你引入的这些字体，有就说明你引入成功了；

## 错误

但是这里有一个问题，就是虽然你成功地引入了自定义字体，但是在使用时，可能还是会出现 `Unrecognized font family xxxx` 的错误，这很有可能是因为你看到并使用的字体的名字和字体内部定义的不一致，这就导致了系统不能识别你所使用的字体。

解决方案如下：

1. 在 iOS 的 "AppDelegate.m" 插入下列代码，运行，然后在控制台中查看字体正确的名称：

   ```objective-c
   for (NSString* family in [UIFont familyNames])  {
       NSLog(@"%@", family);
       
       for (NSString* name in [UIFont fontNamesForFamilyName: family])    {
         NSLog(@"  %@", name);
       }
     }
   ```

2. 在 react native 中使用正确的名称；

> 参见 https://github.com/facebook/react-native/issues/18269 ；