---
layout:     post   				    # 使用的布局（不需要改）
title:      React 的优势				# 标题 
subtitle:      #副标题
date:       2018-12-27 				# 时间
author:     liangping 						# 作者
header-img: img/react.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - React
---

# React优势
1. 声明式的写法，不需要关心如何渲染，只需要声明渲染什么
2. React.js 相对于直接操作原生DOM有很大的性能优势， 很大程度上都要归功于virtual DOM的batching 和diff。batching把所有的DOM操作搜集起来，一次性提交给真实的DOM。diff算法时间复杂度也从标准的的Diff算法的O(n\^3)降为O(n)。没有任何框架可以比纯手动的优化 DOM 操作更快，因为框架的 DOM 操作层需要应对任何上层 API 可能产生的操作，react在不需要手动优化的情况下提供了不错的性能
3. 组件化的写法，使得代码更容易，可维护性高

# diff算法理解
**react diff算法的3个策略**
1. Web UI 中DOM节点跨层级的移动操作特别少，可以忽略不计
2. 拥有相同类的两个组件将会生成相似的树形结构，拥有不同类的两个组件将会生成不同的树形结构，这里也是抓前者放后者的思想。
3. 对于同一层级的一组子节点，它们可以通过唯一id进行区分，即没事就warn的key。
对于以上三个策略，react分别对tree diff, component diff, element diff进行算法优化。

    ## tree diff 树分层比较
    ![](https://segmentfault.com/img/remote/1460000010686588)
    两棵树只会对同一层次的节点进行比较，忽略DOM节点跨层级的移动操作。React只会对相同颜色方框内的DOM节点进行比较，即同一个父节点下的所有子节点。当发现节点已经不存在，则该节点及其子节点会被完全删除掉，不会用于进一步的比较。这样只需要对树进行一次遍历，便能完成整个DOM树的比较。由此一来，最直接的提升就是复杂度变为线型增长而不是原先的指数增长。
    值得一提的是，如果真的发生跨层级移动(如下图)，例如某个DOM及其子节点进行移动挂到另一个DOM下时，React是不会机智的判断出子树仅仅是发生了移动，而是会直接销毁，并重新创建这个子树，然后再挂在到目标DOM上。从这里可以看出，在实现自己的组件时，保持稳定的DOM结构会有助于性能的提升。
    
    ## component diff 组件层比较
    1. 如果是同类型组件，则按照原策略继续比较virtual DOM树
    2. 如果不是，则将该组件判断为dirty component，然后整个unmount这个组件下的子节点对其进行替换
    3. 对于同类型组件，virtual DOM可能并没有发生任何变化，这时我们可以通过shouldCompoenentUpdate钩子来告诉该组件是否进行diff，从而提高大量的性能。或者使用PureComponent*https://cn.vuejs.org/v2/guide/comparison.html*使用 PureComponent 和 shouldComponentUpdate 时，需要保证该组件的整个子树的渲染输出都是由该组件的 props 所决定的。如果不符合这个情况，那么此类优化就会导致难以察觉的渲染结果不一致。这使得 React 中的组件优化伴随着相当的心智负担。
    
    ## element diff 元素间的比较
    设置key
    
# React单项数据流
父组件或子组件都不能知道某个组件是有状态还是无状态，并且它们不应该关心某组件是被定义为一个函数还是一个类。

这就是为什么状态通常被称为局部或封装。 除了拥有并设置它的组件外，其它组件不可访问。

组件可以将其状态作为属性传递给子组件。任何状态始终由某些特定组件所有，并且从该状态导出的任何数据或 UI 只能影响树中下方的组件。

添加**反向数据流**
通过回调函数，加入setState，子组件可以设置父组件的state
>与Angular和Vue的比较
>* Vue:单向数据流并非‘单项绑定’，单项数据流与绑定没有关系。Vue也是单项数据流，但是MVVM框架使得视图(View)和模型(Model)之间，双向绑定，改变一个另外一个会相应的改变，这借助的事ViewModel这中间的一层
>![](https://img-blog.csdn.net/20180313195124568?watermark/2/text/Ly9ibG9nLmNzZG4ubmV0L0JvbmpvdXJqdw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
>* Angular: 单项数据流，流向单一，追踪问题的产生方便，缺点是要多写一些代码。双向数据流值与UI绑定，更方便，但是数据相互依赖，产生问题难以被跟踪到。

# React组件通信
1. 父组件向子组件传值，通过props传递
2. 子组件向父组件传值
    * 利用回调函数
    * 利用自定义事件
3. 跨级组建通信
    * 层层组件传递props
    * 使用context
        * 如果你对状态管理库如Redux或Mobx不太熟悉，那就别用context了。在很多实际应用中，这些库及其React绑定是管理与许多组件相关的state的不错选择。Redux可能是你更好的选择，而不是context。*https://react.docschina.org/docs/legacy-context.html*
4. 没有嵌套关系组件之间通信：自定义事件

# prop和state的区别
* 传入组件的值被称为props，它来自父组件或者祖先组件通过单向数据流传递来，是不可改变的，组件根据props渲染。
* state代表的是一个组件的局部状态，其他组件不可访问，但可作为属性传递给其他组件
    
