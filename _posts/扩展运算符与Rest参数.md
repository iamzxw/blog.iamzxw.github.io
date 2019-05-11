---
layout:     post   				    # 使用的布局（不需要改）
title:      数组方法总结				# 标题 
subtitle:      #副标题
date:       2019-5-11				# 时间
author:     liangping 						# 作者
header-img: img/post-bg-js-version.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - JavaScript
    - ES6
---

# 扩展运算符与Rest参数
## rest 参数
* ES6 引入了rest参数（形式为“...变量名”）,用于获取函数的多余参数，这样就不需要使用arguments对象了。rest参数搭配的是一个数组，该变量将多余的参数放入其中。

```javascript
    function add(...values) {
        let sun = 0
        for(var value of values){
            sum += value
        }
        return sum
    }

    add(2,5,3) // 10
```

* 可以用来替代arguments

```javascript
    // arguments的写法
    function sortNumbers() {
        return Array.prototype.slice.call(arguments).sort()
    }

    // rest 参数的写法
    const sortNumbers = (...numbers) => numbers.sort()
```

**rest参数只能作为最后一位，否则会报错**

```javascript
    // 报错
    function f(a,...b,c){
        // sth
    }
```

* 函数的length属性不包括rest参数。
```javascript
    (function(a){}).length // 1
    (function(...a){}).length // 0
    (function(a,...b){}).length // 1
```

## 扩展运算符
### 含义
* 扩展运算符号（spread）是三个点（...），它如同rest参数的逆运算，将一个数组转化为用逗号分隔的参数序列。

```javascript
    // console 输入多个参数需要逗号，...返回的是带逗号的 
    console.log(...[1,2,3]) //1 2 3
    console.log(1,...[2,3,4],5) // 1 2 3 4 5
    [...doucument.querySelectorAll('div')]
    // [<div>,<div>,<div>]
```

* 该运算符主要用于函数调用,下面两个例子，将数组变为参数序列

```javascript
    function push(array,...item){
        array.push(...items)
    }

    function add(x,y) {
        return x + y
    }

    var numbers = [4, 38]
    add(...numbers)
```

* 扩展运算符可以与正常函数参数混合使用

```javascript
    function f(v,w,x,y,z){}
    var args = [0,1]
    f(-1,...args,2,...[3])
```

* 扩展运算符后拜年可以放置表达式
```javascript
    const arr = [(x > 0 ? ['a']:[]),'b']
```

* 如果扩展运算符后边是一个空数组，则不产生仍和效果。
```javascript
    [...[],1]
    // [1]
```

### 替代数组apply方法

* 求最大值

```javascript
    // ES5
    Math.max.apply(null,[14, 3, 77])

    // ES6
    Math.max(...[14, 3, 77])

    // 等价于
    Math.max(14, 3, 77)
```

* 数组push
```javascript
    // ES5
    var arr1 = [0, 1, 2]
    var arr2 = [3, 4, 5]

    Array.prototype.push.apply(arr1 , arr2)

    // ES6
    var arr1 = [0, 1, 2]
    var arr2 = [3, 4, 5]

    arr1.push(...arr2)
```

### 扩展运算符应用

* 合并数组
```javascript
    // ES5
    var arr1 = [0, 1, 2]
    var arr2 = [3, 4, 5]
    var arr3 = [3, 4, 5]

    arr1.concat(arr2,arr3)

    // ES6
    [...arr1,...arr2,...arr3]
```

* 与解构赋值结合使用，用于生成数组
  **当rest解构使用时，只可以将rest放在最后一位**
```javascript
    // ES5
    a = list[0],rest = list.slice(1)
    
    // ES6
    [a,...rest] = list
```

* 函数的返回值
  javascript只能返回一个参数，如需要返回多个参数，只能返回数组或对象。扩展运算符提供了另一种变通方法
```javascript
    var dateFields = readDateFields(database)
    var d = new Date(...dateFields)
```
上面的代码从数据库取出一行数据，通过扩展运算符，直接将其传入构造函数Date。

* 字符串
  扩展运算符可以将字符转化为真正的数组。
```javascript
    [...'hello']
    // ["h","e","l","l","o"]
```

* 可以正确识别超过\FFFFF的32位Unicode字符
```javascript
    'x\uD83D\uDE80y'.length // 4 wrong
    [...'x\uD83D\uDE80y'].length //3 right

    let str = 'x\uD83D\uDE80y'

    str.split("").reverse().join("") 
    // 'y\uDE80\uD83Dx' wrong

    [...str].reverse.join("")
    // 'y\uD83D\uDE80x' right
```

* 实现了 Iterator接口的对象
任何实现了Iterator接口的对象都可以扩展运算符转化为真正的数组。

