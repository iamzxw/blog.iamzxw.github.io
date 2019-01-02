---
layout:     post   				    # 使用的布局（不需要改）
title:      React 要点记录 				# 标题 
subtitle:      #副标题
date:       2018-12-27 				# 时间
author:     liangping 						# 作者
header-img: img/react.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - React
---

# JSX
1. 用括号括起来，防止在写多行时，自动插入；
2. 使用花括号在属性中嵌入JavaScript表达式，使用引号将字符串文字指定为属性
3. 默认情况下，React DOM会在渲染之前转义 JSX中嵌入的任何值。在渲染之前，所有东西都被转换为字符串。这有助于防止XSS（跨站点脚本）攻击。
4. 首字母必须大写大写，否则必须将其必须将其赋值给大写开头的变量。因为小写会被误认为原生的html标签
5. 不能使用表达式作为标签名，但是可以先将表达式赋值给大写开头的变量

    ```javascript
    function Story(props) {
      // 正确！JSX 标签名可以为大写开头的变量。
      const SpecificStory = components[props.storyType];
      return <SpecificStory story={props.story} />;
    }
    ```

6. 可以将字符串常量作为属性值传递

    ```javascript
        //以下是等价且正确的：
        <MyComponent message="hello world" />
        <MyComponent message={'hello world'} />
    ```
7. ~~如果某个人属性在父组件写了，但是没有赋值，则默认赋值为rue，不推荐做法~~
8. 扩展属性，可以使用...在传递整个props中的属性<Name  {...props}>


# 渲染元素（Render Element）
1. 通过ReactDOM.render()
2. 只有内容已更新的节点才会被react更新

# 组件和Props（components and props）
1. 当React遇到的元素是用户自定义的组件，它会将JSX属性作为单个对象传递给该组件，这个对象称之为“props”。

    ```javascript
    const element = <Welcome name="Sara" />
    ```
     
    官网示例中使用的属性为name，说明，只要不是标准的标签，传入属性，都会优先jsx解释
1. 提取组件一开始看起来像是一项单调乏味的工作，但是在大型应用中，构建可复用的组件完全是值得的。
2. props具有只读性，无论是使用函数或是类来声明一个组件，它决不能修改它自己的props。
3. 定义一个组件最简单的方式是使用JavaScript函数：

    ```javascript
    function Welcome(props) {
      return <h1>Hello, {props.name}</h1>;
    }
    // 在这之后即可使用<Welcome>标签
    // 也可以使用class来定义
    class Welcome extends React.Component {
      	render() {return <h1>Hello, {this.props.name}</h1>}
    }
    ```

# State和生命周期（State and Lifecycle）
1. 当组件输出到 DOM 后会执行 componentDidMount() 钩子
2. 虽然 this.props 由React本身设置以及this.state 具有特殊的含义，但如果需要存储不用于视觉输出的东西，则可以手动向类中添加其他字段。（构造函数中的this.state中没有相应的属性，但是后边的代码中直接使用并设定相关属性的值）
3. 使用 this.setState() 来更新组件局部状态
    * 不要直接更新状态（this.state.comment='hello',不会重新渲染组件），构造函数是唯一能够初始化this.state的地方
    * 状态更新可能是异步的，多个setState可能被react合并并调用为一个来提高性能，因此当要以state为参数时，应当使用prevState和props作为setstate的参数

    ```javascript
    // Wrong
    this.setState({
      counter: this.state.counter + this.props.increment,
    });
    // Correct
    this.setState(function(prevState, props) {
      return {
        counter: prevState.counter + props.increment
      };
    });
    ```

# 事件处理（Handling Events）
1. React事件绑定属性的命名采用驼峰式写法，而不是小写。
2. 在 React 中另一个不同是你不能使用返回 false 的方式阻止默认行为。你必须明确的使用 preventDefault。例如，传统的 HTML 中阻止链接默认打开一个新页面，你可以这样写：

    ```javascript
    <a href="#" onclick="console.log('The link was clicked.'); return false">
      Click me
    </a>
    ```
    在 React，应该这样来写：
    
    ```javascript
    function ActionLink() {
      function handleClick(e) {
        e.preventDefault();
        console.log('The link was clicked.');
      }
    
      return (
        <a href="#" onClick={handleClick}>
          Click me
        </a>
      );
    }
    ```
    *注：event.preventDefault()取消事件的默认动作。e 是一个合成事件。React 根据 W3C spec 来定义这些合成事件。*
3. 如果采用 JSX 的语法你需要传入一个函数作为事件处理函数，而不是一个字符串(DOM元素的写法)，传统的html中onclick="funcname()",在react中应该为onClick={funcname} 
4. 类的方法默认不会绑定this，这时如果调用相关方法this.funcname,this为undefined，推荐使用第一种或第二种解决方法，第三种解决方法可能会导致额外的重新渲染
    * 解决方法一：bind，例如：
    
        ```javascript
        this.handleClick = this.handleClick.bind(this)
        ```
        
    * 解决方法二：属性初始化器，但是属性初始化器尚处在实验性质
  
        ```javascript
        handleClick = () => {}
        ```

    * 解决方法三：毁掉函数中使用箭头函数
    
        ```javascript
        <button onClick={(e) => this.handleClick(e)}>
            Click me
        </button>  
        ```

5. 传递参数，例子：

    ```javascript
    <button onClick={(e) => this.deleteRow(id, e)}>Delete Row</button>
    <button onClick={this.deleteRow.bind(this, id)}>Delete Row</button>
    ```
    
    *值得注意的是，通过 bind 方式向监听函数传参，在类组件中定义的监听函数，事件对象 e 要排在所传递参数的后面，preventPop(name, e){ }   //事件对象e要放在最后*

# 条件渲染（Conditional Rendering）
1. if
2. 元素变量{元素变量}
3. 与运算符&&
    * {true && expression}总是返回 expression，而 {false && expression} 总是返回 false。因此，如果条件是 true，&& 右侧的元素就会被渲染，如果是 false，React 会忽略并跳过它。
4. 三目运算符{condition?true:false}
5. 阻止组件渲染 让render方法返回 return null，组件的 render 方法返回 null 并不会影响该组件生命周期方法的回调。例如，componentWillUpdate 和 componentDidUpdate 依然可以被调用。
    * 只要条件变得过于复杂，就可能是提取组件的好时机。	
# 列表和keys（lists&keys）
1. Keys可以在DOM中的某些元素被增加或删除的时候帮助React识别哪些元素发生了变化。因此应当给数组中的每一个元素赋予一个确定的标识。一个元素的key最好是这个元素在列表中拥有的一个独一无二的字符串.然而，它们不需要是全局唯一的。当我们生成两个不同的数组时，我们可以使用相同的键
    * 通常，使用来自数据的id作为元素的key.（推荐做法）
    * ~~当元素没有确定的id时，可以使用索引index作为key（不推荐做法，因为如果要对列表进行重新排序，这将导致渲染变慢）~~
2. 当在map()方法的内部调用元素时，你最好随时记得为每一个元素加上一个独一无二的key。
3. key为react保留字，不要作为组件的属性名
    * 如果一个map()嵌套了太多层级，那可能就是你提取出组件的一个好时机。

# 表单（from）
1. 当你有处理多个受控的input元素时，你可以通过给每个元素添加一个name属性，来让处理函数根据 event.target.name的值来选择做什么。

# 状态提升（Lifting State Up）
1. 在React应用中，对应任何可变数据理应只有一个单一“数据源”。通常，状态都是首先添加在需要渲染数据的组件中。此时，如果另一个组件也需要这些数据，你可以将数据提升至离它们最近的父组件中。你应该在应用中保持 自上而下的数据流，而不是尝试在不同组件中同步状态。

# 组合vs继承（Composition vs Inheritance）
1. 一些组件不能提前知道它们的子组件是什么。这对于 Sidebar 或 Dialog 这类通用容器尤其常见。我们建议这些组件使用 children 属性将子元素直接传递到输出。{props.children}

# 使用 PropTypes 进行类型检查
1. import PropTypes from 'prop-types' ,  React.PropTypes 自 React v15.5 起已弃用。请使用 prop-types 库代替。
2. 要检查组件的属性，你需要配置特殊的 propTypes属性:
    ```javascript
    Greeting.propTypes = {
      name: PropTypes.string
    };
    ```

3.
```javascript
import PropTypes from 'prop-types';

MyComponent.propTypes = {
  // 你可以将属性声明为以下 JS 原生类型
  optionalArray: PropTypes.array,
  optionalBool: PropTypes.bool,
  optionalFunc: PropTypes.func,
  optionalNumber: PropTypes.number,
  optionalObject: PropTypes.object,
  optionalString: PropTypes.string,
  optionalSymbol: PropTypes.symbol,

  // 任何可被渲染的元素（包括数字、字符串、子元素或数组）。
  optionalNode: PropTypes.node,

  // 一个 React 元素
  optionalElement: PropTypes.element,

  // 你也可以声明属性为某个类的实例，这里使用 JS 的
  // instanceof 操作符实现。
  optionalMessage: PropTypes.instanceOf(Message),

  // 你也可以限制你的属性值是某个特定值之一
  optionalEnum: PropTypes.oneOf(['News', 'Photos']),

  // 限制它为列举类型之一的对象
  optionalUnion: PropTypes.oneOfType([
    PropTypes.string,
    PropTypes.number,
    PropTypes.instanceOf(Message)
  ]),

  // 一个指定元素类型的数组
  optionalArrayOf: PropTypes.arrayOf(PropTypes.number),

  // 一个指定类型的对象
  optionalObjectOf: PropTypes.objectOf(PropTypes.number),

  // 一个指定属性及其类型的对象
  optionalObjectWithShape: PropTypes.shape({
    color: PropTypes.string,
    fontSize: PropTypes.number
  }),

  // 你也可以在任何 PropTypes 属性后面加上 `isRequired` 
  // 后缀，这样如果这个属性父组件没有提供时，会打印警告信息
  requiredFunc: PropTypes.func.isRequired,

  // 任意类型的数据
  requiredAny: PropTypes.any.isRequired,

  // 你也可以指定一个自定义验证器。它应该在验证失败时返回
  // 一个 Error 对象而不是 `console.warn` 或抛出异常。
  // 不过在 `oneOfType` 中它不起作用。
  customProp: function(props, propName, componentName) {
    if (!/matchme/.test(props[propName])) {
      return new Error(
        'Invalid prop `' + propName + '` supplied to' +
        ' `' + componentName + '`. Validation failed.'
      );
    }
  },

  // 不过你可以提供一个自定义的 `arrayOf` 或 `objectOf` 
  // 验证器，它应该在验证失败时返回一个 Error 对象。 它被用
  // 于验证数组或对象的每个值。验证器前两个参数的第一个是数组
  // 或对象本身，第二个是它们对应的键。
  customArrayProp: PropTypes.arrayOf(function(propValue, key, componentName, location, propFullName) {
    if (!/matchme/.test(propValue[key])) {
      return new Error(
        'Invalid prop `' + propFullName + '` supplied to' +
        ' `' + componentName + '`. Validation failed.'
      );
    }
  })
// This must be exactly one element or it will warn.
    	const children = this.props.children;
};
```

4. 属性默认值，可以通过配置 defaultProps 为 props定义默认值

    ```javascript
    // 为属性指定默认值:
    Greeting.defaultProps = {
      name: 'Stranger'
    };
    ```
    
# Ref&DOM
1. 下面是几个适合使用 refs 的情况：
    * 处理焦点、文本选择或媒体控制。
    * 触发强制动画。
    * 集成第三方 DOM 库
    *如果可以通过声明式实现，则尽量避免使用 refs。*
2. 创建Refs

```javascript
class MyComponent extends React.Component {
  constructor(props) {
    super(props);
    this.myRef = React.createRef();
  }
  render() {
    return <div ref={this.myRef} />;
  }
}
```

3. 访问Refs
    * 当一个 ref 属性被传递给一个 render 函数中的元素时，可以使用 ref 中的 current 属性对节点的引用进行访问。const node = this.myRef.current;

1. ref的值取决于节点的类型:
    * 当 ref 属性被用于一个普通的 HTML 元素时，React.createRef() 将接收底层 DOM 元素作为它的 current属性以创建 ref 。
    * 当 ref 属性被用于一个自定义类组件时，ref 对象将接收该组件已挂载的实例作为它的 current 。
    * 你不能在函数式组件上使用 ref 属性，因为它们没有实例。
2. React 会在组件加载时将 DOM 元素传入 current 属性，在卸载时则会改回 null。ref 的更新会发生在componentDidMount 或 componentDidUpdate 生命周期钩子之前。
3. 不能在函数式组件上使用ref属性，因为函数是组件没有实例。但是可以在函数式组件内部使用ref，只要它只想一个DOM元素或者class组件

    ```javascript
    function CustomTextInput(props) {
      // 这里必须声明 textInput，这样 ref 回调才可以引用它
      let textInput = null;
    
      function handleClick() {
        textInput.focus();
      }
      return (
        <div>
          <input
            type="text"
            ref={(input) => { textInput = input; }} />
          <input
            type="button"
            value="Focus the text input"
            onClick={handleClick}
          />
        </div>
      );  
    }
    ```
    
# 自定义事件


# context
**Context 通过组件树提供了一个传递数据的方法，从而避免了在每一个层级手动的传递 props 属性。**
**不要仅仅为了避免在几个层级下的组件传递 props 而使用 context，它是被用于在多个层级的多个组件需要访问相同数据的情景。**

1. React.createContext
    创建一对 { Provider, Consumer }。当 React 渲染 context 组件 Consumer 时，它将从组件树的上层中最接近的匹配的 Provider 读取当前的 context 值。如果没有匹配值，可以用到defaultValue。
    
```javascript
const {Provider, Consumer} = React.createContext(defaultValue);
```
2. Provider
    接收一个 value 属性传递给 Provider 的后代 Consumers。一个 Provider 可以联系到多个 Consumers。Providers 可以被嵌套以覆盖组件树内更深层次的值。
    
```javascript
<Provider value={/* some value */}>
```

3. Consumer

```javascript
<Consumer>
  {value => /* render something based on the context value */}
</Consumer>
```
    每当Provider的值发送改变时, 作为Provider后代的所有Consumers都会重新渲染。 从Provider到其后代的Consumers传播不受shouldComponentUpdate方法的约束，因此即使祖先组件退出更新时，后代Consumer也会被更新.
    