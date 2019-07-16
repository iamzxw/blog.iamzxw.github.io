---
layout:     post   				    # 使用的布局（不需要改）
title:      var,let,const			# 标题 
subtitle:      #副标题
date:       2019-5-15				# 时间
author:     liangping 						# 作者
header-img: img/post-bg-js-version.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - JavaScript
    - ES6
---

# var,let,const的区别

## let 与 var

* var声明的变量仅具有函数级别作用域

* let声明的变量具有块级作用域{} 

* let不允许重复声明同一个变量 // Uncaught SyntaxError: Identifier 'a' has already been declared

* 暂时性死区：let、const 所在块级作用域，在变量声明之前就访问变量的话，会直接提示 ReferenceError，而不像 var 那样使用默认值 undefined。

## const

声明一个只读常量。一旦声明，常量的值就不改变。

const实质上保证的是常量指向的那个内存地址不变。对于简单的数据值（数值、字符串、布尔值）而言，值就保存在变量指向的内存地址中，因此等同于常量。但对于复合类型的数值（主要是数组和对象）而言，变量指向的地址只是一个指针，const只能保证这个指针是固定的，至于他指向的数据结构是不是不可变的。这完全不能控制。**因此将一个复合数据类型声明为常量必须非常小心**
