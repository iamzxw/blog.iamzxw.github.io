---
layout:     post   				    # 使用的布局（不需要改）
title:      Vuex、Flux、Redux、Redux-saga、Dva、MobX(一) 				# 标题 
subtitle:      #副标题
date:       2019-2-16 				# 时间
author:     liangping 						# 作者
header-img: img/post-bg-keybord.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - Redux
---

# Vuex、Flux、Redux、Redux-saga、Dva、MobX
[原文链接](https://mp.weixin.qq.com/s/T3UeN2-RjSNP0mGjJr0PDw)
> 什么是状态共享？比如一个组件需要使用另一个组件的状态，或者一个组件需要改变另一个组件的状态，都是状态共享。
> 父组件质检，兄弟组件之间共享状态往往需要写很多没有必要的代码，比如把状态提升到父组件里，或者给兄弟组件写一个父组件，这很麻烦。
> 如果不对状态进行有效的管理，状态什么时候，什么怨因，如何变化就不会受到控制，就很难追踪和测试。
> 
> 在软件开发里，有些通用的思想，比如隔离变化，约定优于配置等，隔离变化就是说做好抽象，把一些容易变化的地方找到共性，隔离出来，不要去影响其他的代码。约定优于配置就是很多东西我们不一定要写一大堆的配置，比如我们几个人约定，view 文件夹里只能放视图，不能放过滤器，过滤器必须放到 filter 文件夹里，那这就是一种约定，约定好之后，我们就不用写一大堆配置文件了，我们要找所有的视图，直接从 view 文件夹里找就行。
> 
> 根据这些思想，对于状态管理的解决思路就是：把组件之间需要共享的状态抽取出来，遵循特定的约定，统一来管理，让状态的变化可以预测。根据这个思路，产生了很多的模式和库，我们来挨个聊聊。

## Store 模式
最简单的处理就是把状态存到一个外部变量里面，比如：this./\$root./\$data，当然也可以是一个全局变量。
**缺点：数据改变后，不会留下变更过的记录，不利于调试。**

所以我们稍微搞得复杂一点，用一个简单的 Store 模式：
```javascript
var store = {
  state: {
    message: 'Hello!'
  },
  setMessageAction (newValue) {
    // 发生改变记录点日志啥的
    this.state.message = newValue
  },
  clearMessageAction () {
    this.state.message = ''
  }
}
```
store 的 state 来存数据，store 里面有一堆的 action，这些 action 来控制 state 的改变，也就是不直接去对 state 做改变，而是通过 action 来改变，因为都走 action，我们就可以知道到底改变（mutation）是如何被触发的，出现错误，也可以记录记录日志啥的。

![](https://ws3.sinaimg.cn/large/006tKfTcly1g08baq53qgj30ef0i9gmg.jpg)

不过这里没有限制组件里面不能修改 store 里面的 state，万一组件瞎胡修改，不通过 action，那我们也没法跟踪这些修改是怎么发生的。所以就需要规定一下，组件不允许直接修改属于 store 实例的 state，组件必须通过 action 来改变 state，也就是说，组件里面应该执行 action 来分发 (dispatch) 事件通知 store 去改变。这样约定的好处是，我们能够记录所有 store 中发生的 state 改变，同时实现能做到记录变更 (mutation)、保存状态快照、历史回滚/时光旅行的先进的调试工具。

这样进化了一下，一个简单的 Flux 架构就实现了。

## Flux
Flux其实是一种思想，就像MVC，MVVM之类的，他给出了一些基本概念，所有的框架都可以根据他的思想来做一些实现。

Flux把一个应用分成了4个部分： View Action Dispatcher Store

![](https://ws4.sinaimg.cn/large/006tKfTcly1g08byevrm2j30u0093wf2.jpg)

比如我们搞一个应用，显而易见，这个应用里面会有一堆的 View，这个 View 可以是Vue的，也可以是 React的，啥框架都行，啥技术都行。

View 肯定是要展示数据的，所谓的数据，就是 Store，Store 很容易明白，就是存数据的地方。当然我们可以把 Store 都放到一起，也可以分开来放，所以就有一堆的 Store。但是这些 View 都有一个特点，就是 Store 变了得跟着变。

View 怎么跟着变呢？一般 Store 一旦发生改变，都会往外面发送一个事件，比如 change，通知所有的订阅者。View 通过订阅也好，监听也好，不同的框架有不同的技术，反正 Store 变了，View 就会变。

View 不是光用来看的，一般都会有用户操作，用户点个按钮，改个表单啥的，就需要修改 Store。Flux 要求，View 要想修改 Store，必须经过一套流程，有点像我们刚才 Store 模式里面说的那样。视图先要告诉 Dispatcher，让 Dispatcher dispatch 一个 action，Dispatcher 就像是个中转站，收到 View 发出的 action，然后转发给 Store。比如新建一个用户，View 会发出一个叫 addUser 的 action 通过 Dispatcher 来转发，Dispatcher 会把 addUser 这个 action 发给所有的 store，store 就会触发 addUser 这个 action，来更新数据。数据一更新，那么 View 也就跟着更新了。

这个过程有几个需要注意的点： Dispatcher 的作用是接收所有的 Action，然后发给所有的 Store。这里的 Action 可能是 View 触发的，也有可能是其他地方触发的，比如测试用例。转发的话也不是转发给某个 Store，而是所有 Store。 Store 的改变只能通过 Action，不能通过其他方式。也就是说 Store 不应该有公开的 Setter，所有 Setter 都应该是私有的，只能有公开的 Getter。具体 Action 的处理逻辑一般放在 Store 里。

听听描述看看图，可以发现，Flux的最大特点就是数据都是单向流动的。

## Redux
Flux 有一些缺点（特点），比如一个应用可以拥有多个 Store，多个Store之间可能有依赖关系；Store 封装了数据还有处理数据的逻辑。

所以大家在使用的时候，一般会用 Redux，他和 Flux 思想比较类似，也有差别。

![](https://ws2.sinaimg.cn/large/006tKfTcly1g08c9u9t5cj30rs0crta0.jpg)

### store
Redux 里面只有一个 Store，整个应用的数据都在这个大 Store 里面。Store 的 State 不能直接修改，每次只能返回一个新的 State。Redux 整了一个 createStore 函数来生成 Store。
```javascript
import { createStore } from 'redux';
const store = createStore(fn);
```
Store 允许使用 store.subscribe 方法设置监听函数，一旦 State 发生变化，就自动执行这个函数。这样不管 View 是用什么实现的，只要把 View 的更新函数 subscribe 一下，就可以实现 State 变化之后，View 自动渲染了。比如在 React 里，把组件的render方法或setState方法订阅进去就行。

### Action
和 Flux 一样，Redux 里面也有 Action，Action 就是 View 发出的通知，告诉 Store State 要改变。Action 必须有一个 type 属性，代表 Action 的名称，其他可以设置一堆属性，作为参数供 State 变更时参考。
```javascript
const action = {
  type: 'ADD_TODO',
  payload: 'Learn Redux'
};
```
Redux 可以用 Action Creator 批量来生成一些 Action。

### Reducer
Redux 没有 Dispatcher 的概念，Store 里面已经集成了 dispatch 方法。store.dispatch()是 View 发出 Action 的唯一方法。
```javascript
import { createStore } from 'redux';
const store = createStore(fn);
store.dispatch({
  type: 'ADD_TODO',
  payload: 'Learn Redux'
});
```
Redux 用一个叫做 Reducer 的纯函数来处理事件。Store 收到 Action 以后，必须给出一个新的 State（就是刚才说的Store 的 State 不能直接修改，每次只能返回一个新的 State），这样 View 才会发生变化。这种 State 的计算过程就叫做 Reducer。

> 什么是纯函数呢，就是说没有任何的副作用，比如这样一个函数：
> ```javascript
> function getAge(user) {
  user.age = user.age + 1;
  return user.age;
}
> ```
> 这个函数就有副作用，每一次相同的输入，都可能导致不同的输出，而且还会影响输入 user 的值，再比如：
> ```javascript
> let b = 10;
function compare(a) {
  return a >= b;
}
> ```
> 这个函数也有副作用，就是依赖外部的环境，b 在别处被改变了，返回值对于相同的 a 就有可能不一样。

而 Reducer 是一个纯函数，对于相同的输入，永远都只会有相同的输出，不会影响外部的变量，也不会被外部变量影响，不得改写参数。它的作用大概就是这样，根据应用的状态和当前的 action 推导出新的 state：
```javascript
(previousState, action) => newState
```
类比 Flux，Flux 有些像：
```javascript
(state, action) => state
```

> 为什么叫做 Reducer 呢？reduce 是一个函数式编程的概念，经常和 map 放在一起说，简单来说，map 就是映射，reduce 就是归纳。映射就是把一个列表按照一定规则映射成另一个列表，而 reduce 是把一个列表通过一定规则进行合并，也可以理解为对初始值进行一系列的操作，返回一个新的值。
> 比如 Array 就有一个方法叫 reduce，Array.prototype.reduce(reducer, ?initialValue)，把 Array 整吧整吧弄成一个 newValue。
> ```javascript
> const array1 = [1, 2, 3, 4];
const reducer = (accumulator, currentValue) => accumulator + currentValue;
// 1 + 2 + 3 + 4
console.log(array1.reduce(reducer));
// expected output: 10
// 5 + 1 + 2 + 3 + 4
console.log(array1.reduce(reducer, 5));
// expected output: 15
> ```

看起来和 Redux 的 Reducer 是不是好像好像，Redux 的 Reducer 就是 reduce 一个列表（action的列表）和一个 initialValue（初始的 State）到一个新的 value（新的 State）。
把上面的概念连起来，举个例子：
下面的代码声明了 reducer：
```javascript
const defaultState = 0;
const reducer = (state = defaultState, action) => {
  switch (action.type) {
    case 'ADD':
      return state + action.payload;
    default: 
      return state;
  }
};
```
createStore接受 Reducer 作为参数，生成一个新的 Store。以后每当store.dispatch发送过来一个新的 Action，就会自动调用 Reducer，得到新的 State。
```javascript
import { createStore } from 'redux';
const store = createStore(reducer);
```
createStore 内部干了什么事儿呢？通过一个简单的 createStore 的实现，可以了解大概的原理（可以略过不看）：
```javascript
const createStore = (reducer) => {
  let state;
  let listeners = [];

  const getState = () => state;

  const dispatch = (action) => {
    state = reducer(state, action);
    listeners.forEach(listener => listener());
  };

  const subscribe = (listener) => {
    listeners.push(listener);
    return () => {
      listeners = listeners.filter(l => l !== listener);
    }
  };

  dispatch({});

  return { getState, dispatch, subscribe };
};
```

Redux 有很多的 Reducer，对于大型应用来说，State 必然十分庞大，导致 Reducer 函数也十分庞大，所以需要做拆分。Redux 里每一个 Reducer 负责维护 State 树里面的一部分数据，多个 Reducer 可以通过 combineReducers 方法合成一个根 Reducer，这个根 Reducer 负责维护整个 State。

```javascript
import { combineReducers } from 'redux';
// 注意这种简写形式，State 的属性名必须与子 Reducer 同名
const chatReducer = combineReducers({
  Reducer1,
  Reducer2,
  Reducer3
})
```
combineReducers 干了什么事儿呢？通过简单的 combineReducers 的实现，可以了解大概的原理（可以略过不看）：
```javascript
const combineReducers = reducers => {
  return (state = {}, action) => {
    return Object.keys(reducers).reduce(
      (nextState, key) => {
        nextState[key] = reducers[key](state[key], action);
        return nextState;
      },
      {} 
    );
  };
};
```

### 流程

![](https://ws2.sinaimg.cn/large/006tKfTcly1g08c9u9t5cj30rs0crta0.jpg)

再回顾一下刚才的流程图，尝试走一遍 Redux 流程：

1、用户通过 View 发出 Action：
```javascript
store.dispatch(action);
```
2、然后 Store 自动调用 Reducer，并且传入两个参数：当前 State 和收到的 Action。 Reducer 会返回新的 State 。
```javascript
let nextState = xxxReducer(previousState, action);
```
3、State 一旦有变化，Store 就会调用监听函数。
```javascript
store.subscribe(listener);
```
4、listener可以通过 store.getState() 得到当前状态。如果使用的是 React，这时可以触发重新渲染 View。
```javascript
function listerner() {
  let newState = store.getState();
  component.setState(newState);   
}
```
