我们将在本书的剩余部分中详细介绍 Lox 语言中的每一个黑暗和阴暗的角落，但是让你立即开始为解释器编写代码，而不知道我们最终会得到什么，这似乎是残酷的。

同时，在你接触文本编辑器之前，我不想让你通过大量的语言律师和规范。所以这将是一个温和，友好的介绍 Lox。它会遗漏很多细节和边缘情况。我们以后有的是时间。

# [**Hello, Lox  你好，洛克斯**](https://craftinginterpreters.com/the-lox-language.html#hello-lox)

```python
print "Hello, world!";
```

正如 `//` 行注释和结尾的注释所暗示的那样，Lox 的语法是一个 C 家族的成员。（字符串周围没有括号，因为 `print` 是一个内置语句，而不是库函数

现在，我不会声称 C 有一_个伟大_的语法。如果我们想要一些优雅的东西，我们可能会模仿 Pascal 或 Smalltalk。如果我们想充分发挥斯堪的纳维亚家具的极简主义，我们会做一个计划。他们都有自己的优点。

相反，类 C 语法所具有的是你经常会发现在一种语言中更有价值的东西： _熟悉度_ 。我知道你已经习惯了这种风格，因为我们将用来_实现_ Lox 的两种语言 --Java 和 C-- 也继承了它。

# [**A High-Level Language  高级语言**](https://craftinginterpreters.com/the-lox-language.html#a-high-level-language)

虽然这本书的篇幅比我预期的要大，但它仍然不足以容纳像 Java 这样的大型语言。为了在这些页面中容纳两个完整的 Lox 实现，Lox 本身必须非常紧凑。

当我想到小而有用的语言时，我想到的是高级“脚本”语言，如 JavaScript、Scheme 和 Lua。在这三种语言中，Lox 看起来最像 JavaScript，主要是因为大多数 C 语法语言都是这样。正如我们后面将要学习的，Lox 的作用域方法与 Scheme 非常接近。我们将在[第三部分](https://craftinginterpreters.com/a-bytecode-virtual-machine.html)中构建的 Lox 的 C 风格在很大程度上归功于 Lua 的干净、高效的实现。

Lox 与这三种语言有两个共同点：

# 动态类型

**Lox 是动态类型的。变量可以存储任何类型的值，单个变量甚至可以在不同时间存储不同类型的值。**
如果您尝试对错误类型的值执行操作 - 例如，将数字除以字符串 - 则会在运行时检测并报告错误。

喜欢静态类型有很多原因，但它们并没有超过为 Lox 选择动态类型的实际原因。静态类型系统需要大量的学习和实现。跳过它会给你一个更简单的语言和一本更短的书。如果我们将类型检查推迟到运行时，我们将更快地启动解释器并执行代码

# [_**自动内存管理**_](https://craftinginterpreters.com/the-lox-language.html#automatic-memory-management)

高级语言的存在是为了消除容易出错的低级苦差事，还有什么比手动管理存储的分配和释放更乏味的呢？没有人会在迎接早晨的太阳时说：“我迫不及待地想知道今天分配的每个字节在哪里调用 `free（）`！”

有两种主要的内存管理技术 ： **引用计数**和**跟踪垃圾收集** （通常称为 **垃圾收集**或 **GC**）。Ref 计数器实现起来要简单得多 -- 我想这就是为什么 Perl、PHP 和 Python 都开始使用它们的原因。但是，随着时间的推移，引用计数的局限性变得太麻烦了。所有这些语言最终都添加了一个完整的跟踪 GC，或者至少有足够的跟踪 GC 来清理对象周期

跟踪垃圾收集有一个可怕的名声。 在原始内存的级别上工作有点令人痛苦。GGC 有时会让你在梦中看到十六进制转储。但是，请记住，这本书是关于驱散魔法和杀死那些怪物的，**所以我们_要_写我们自己的垃圾收集器。**我想你会发现这个算法很简单，而且实现起来很有趣。

# 数据类型

在 Lox 的小宇宙中，构成所有物质的原子是内置的数据类型。只有几个：

**布尔值 。** 没有逻辑就不能编码，没有布尔值就不能进行逻辑。“真”与“假”，软件的阴阳。与一些重新利用现有类型来表示真和假的古代语言不同，Lox 有一个专用的布尔类型。我们这次探险可能很艰苦，但我们不是_野蛮人_ 。

显然，有两个布尔值，每个值都有一个文字。

```
true;  // Not false.
false; // Not *not* false.
```

**Numbers.** Lox 只有一种数字：双精度浮点数。由于浮点数也可以表示各种整数，因此在保持简单的同时，它涵盖了很多领域。

全功能语言有很多数字语法 -- 十六进制、科学记数法、八进制，各种有趣的东西。我们将满足于基本的整数和十进制文字。

**字符串.** 我们已经在第一个例子中看到了一个字符串字面量。像大多数语言一样，它们都用双引号括起来。
```python
"I am a string";
"";    // The empty string.
"123"; // This is a string, not a number.
```
正如我们在实现它们时所看到的，在这个无害的字符序列中隐藏着相当多的复杂性。

**Nil 还有最后一个内在价值，**
他从未被邀请参加派对，但似乎总是出现。它代表“没有价值”。它在许多其他语言中被称为“null”。在 Lox 中，我们拼写为 `nil`。(When 我们开始实现它，这将有助于区分 Lox 的 `nil` 与 Java 或 C 的 `null`。

在语言中没有空值有很好的理由，因为空指针错误是我们行业的祸害。如果我们正在开发一种静态类型的语言，那么禁止它是值得的，但是在动态类型的语言中，消除它往往比拥有它更令人讨厌

# [**表达式**](https://craftinginterpreters.com/the-lox-language.html#expressions)

如果内置数据类型及其文字是原子，那么**表达式**必须是分子。其中大部分将是熟悉的。

# [_**算术**_](https://craftinginterpreters.com/the-lox-language.html#arithmetic)

Lox 提供了你所熟悉和喜爱的 C 语言和其他语言中的基本算术运算符：

```
add + me;
subtract - me;
multiply * me;
divide / me;
```

运算符两边的子表达式都是**操作数** 。因为有_两_个，所以它们被称为**二元**运算符。(It 与二进制的 1 和 0 用法无关。）因为操作符固定_在_操作数的中间，所以这些操作符也称为**中缀**操作符（与操作符位于操作数之前的前缀操作符和**操作符位于操作**数之后的后缀操作符相反）。

一个算术运算符实际上_既是_中缀又是前缀。那个 `-` 运算符也可以用来求反一个数。
`-negateMe;` 所有这些运算符都处理数字，向它们传递任何其他类型都是错误的。**例外是 `+` 运算符 - 你也可以传递两个字符串来连接它们。**

# [_Comparison and equality 比较与平等_](https://craftinginterpreters.com/the-lox-language.html#comparison-and-equality)

沿着，我们还有一些总是返回布尔结果的运算符。我们可以使用 Ye Olde Comparison Operators 比较数字（而且只能比较数字）。

```
less < than;
lessThan <= orEqual;
greater > than;
greaterThan >= orEqual;
```

我们可以测试任何类型的两个值是否相等或不相等。

```
1 == 2;         // false.
"cat" != "dog"; // true.
```

Even different types.  甚至是不同的类型
`314 == "pi"; // false.`

不同类型的值_永远不_相等。

`123 == "123"; // false.`

我通常反对隐式转换

# [_**3 . 4 . 3Logical operators  逻辑运算符**_](https://craftinginterpreters.com/the-lox-language.html#logical-operators)

not 运算符，前缀 `！` 如果其操作数为真，则返回 `false`，反之亦然。

```
!true;  // false.
!false; // true.
```

另外两个逻辑操作符实际上是以表达式为幌子的控制流构造。`and` 表达式确定两个值是否_都_为 true。如果为假，则返回左操作数，否则返回右操作数。

```
true and false; // false.
true and true;  // true.
```

`或`表达式确定两个值中的_任一个_ （或两个）是否为真。如果为真，则返回左操作数，否则返回右操作数。

```
false or false; // false.
true or false;  // true.
```

and 和 `or` 类似于控制流结构的原因是它们 短路如果左边的操作数为假，不仅返回，而且在这种情况下，它甚至不_计算_相反地（contrapositively？），如果 OR 的左操作数为真，则跳过右操作数。

# [_**3 . 4 . 4Precedence and grouping  优先和分组**_](https://craftinginterpreters.com/the-lox-language.html#precedence-and-grouping)

所有这些运算符都具有相同的优先级和结合性，这是你期望从 C 中得到的。(When 我们会进行解析，我们会得到更精确_的方法_ 。）如果优先级不是您想要的，您可以使用 `（）` 团体活动 

`var average = (min + max) / 2;`

由于它们在技术上不是很有趣，我已经从我们的小语言中删除了典型操作符的其余部分。无位、移位、模或条件运算符。我不会给你打分，但是如果你用它们来增强你自己的 Lox 实现，你会在我心中得到加分。

这些是表达形式（除了与我们稍后将讨论的特定功能相关的几个），所以让我们向上移动一个级别

# [**Statements  报表**](https://craftinginterpreters.com/the-lox-language.html#statements)

现在我们在录口供。**表达式的主要工作是产生一个_值_ ，语句的工作是产生一个_效果_** 。因为，根据定义，语句不计算值，所以要有用，它们必须以某种方式改变世界 - 通常是修改某些状态，阅读输入或产生输出。 你们已经看过几种声明了。第一个是 `print "Hello, world!";`

`print` 语句计算单个表达式并将结果显示给用户。你也看到了一些声明，比如： `"some expression";`

**表达式后跟一个小字符串（`;`）将表达式提升为语句级**。这被称为（足够简洁地） **表达式语句** 。 如果你想打包一系列的语句，而只需要一个语句，你可以把它们包装在一个**块**中。

```
{
  print "One statement.";
  print "Two statements.";
}
```

块也会影响作用域，这将引导我们进入下一节 。. .& nbsp;你好

# [**3 . 6Variables  变量**](https://craftinginterpreters.com/the-lox-language.html#variables)

使用 `var` 语句声明变量。**如果省略初始化器，则变量的值默认为 `nil`。**

```
var imAVariable = "here is my value";
var iAmNil;
```

一旦声明了变量，您就可以自然地使用变量名访问和赋值变量。

```
var breakfast = "bagels";
print breakfast; // "bagels".
breakfast = "beignets";
print breakfast; // "beignets".
```

我不会在这里讨论变量作用域的规则，因为我们将在后面的章节中花费大量的时间来映射规则的每一寸。在大多数情况下，它的工作方式就像你期望来自 C 或 Java 一样。

# [**3 . 7Control Flow  控制流**](https://craftinginterpreters.com/the-lox-language.html#control-flow)

如果你不能跳过某些代码或者多次执行某些代码，那么就很难写出有用的程序。这意味着控制流。除了我们已经介绍过的逻辑运算符之外，Lox 还直接从 C 语言中提取了三个语句。

`if` 语句根据某些条件执行两个语句之一。

```
if (condition) {
  print "yes";
} else {
  print "no";
}
```

只要条件表达式的计算结果为 true`，while` 循环就会重复执行主体。

```
var a = 1;
while (a < 10) {
  print a;
  a = a + 1;
}
```

最后，我们有 `for` 循环

```
for (var a = 1; a < 10; a = a + 1) {
  print a;
}
```

这个循环和前面的 `while` 循环做的是一样的。大多数现代语言也有某种 `for-in` 或 `foreach` 循环，用于显式迭代各种序列类型。在真实的语言中，这比我们在这里得到的粗糙的 C 风格`的 for` 循环要好。Lox 保持基本。

# [**Functions](https://craftinginterpreters.com/the-lox-language.html#functions)  函数**

函数调用表达式看起来与 C 中的一样 

`makeBreakfast(bacon, eggs, toast);`

你也可以调用一个函数而不向它传递任何东西。 `makeBreakfast();`

与 Ruby 等不同，在这种情况下括号是强制性的。如果你不使用它们，名字就不会_调用_函数，它只是引用它。 如果你不能定义自己的函数，一门语言就不是很有趣。在 Lox，你这样做很`有趣` 。

```
fun printSum(a, b) {
  print a + b;
}
```

现在是澄清一些术语的好时机。有些人把“parameter”和“argument”抛在一边，好像它们是可以互换的，对许多人来说，它们确实是。我们会花很多时间来分析语义上的细微差别，所以让我们来提高我们的词汇。从现在开始：

**argument** 是在调用函数时传递给函数的实际值。因此，函数_调用_有一个_参数_列表。有时你会听到**实际的参数**用于这些。

**parameter** 是一个变量，它将参数的值保存在函数体中。因此，函数_声明_有一_个参数_ 名单其他人称这些**为形式参数**或简单**的形式**  函数的主体总是一个块。在它里面，你可以使用 `return` 语句返回一个值。

```
fun returnSum(a, b) {
  return a + b;
}
```

如果执行到达块的末尾而没有`返回` ， 隐式返回 `nil`。

# [_**关闭**_](https://craftinginterpreters.com/the-lox-language.html#closures)

**函数在 Lox 中是一种类型**，这意味着它们是真实的值**，你可以得到一个引用，存储在变量中，传递等等
print 是原语，内置语句

```
fun addPair(a, b) {
  return a + b;
}

fun identity(a) {
  return a;
}

print identity(addPair)(1, 2); // Prints "3".
```

由于函数声明是语句，因此可以在另一个函数中声明局部函数。

```
fun outerFunction() {
  fun localFunction() {
    print "I'm local!";
  }

  localFunction();
}
```

如果你联合收割机局部函数、一级函数和块作用域，你会遇到这种有趣的情况

```
fun returnFunction() {
  var outside = "outside";

  fun inner() {
    print outside;
  }

  return inner;
}

var fn = returnFunction();
fn();
```

在这里，`inner（）` 访问一个在其函数体外部声明的局部变量。这是犹太教的吗现在很多语言都从 Lisp 借用了这个特性，你可能知道答案是肯定的。

要做到这一点，`inner（）` **必须“保持”对它使用的任何周围变量的引用，以便即使在外部函数返回后它们也保持不变**。我们称之为**闭包**函数。这些天来，这个词经常用于_任何_ 一级函数，尽管如果函数不 会关闭任何变量。

正如你所想象的，实现这些增加了一些复杂性，因为我们不能再假设变量作用域严格地像堆栈一样工作，在堆栈中，局部变量在函数返回时蒸发。我们将有一个有趣的时间学习如何使这些工作正确和有效。

# [**Classes  类**](https://craftinginterpreters.com/the-lox-language.html#classes)

由于 Lox 具有动态类型、词法（大致为“块”）作用域和闭包， 它离函数式语言只有一半的距离。但正如你所见 _也_是一种面向对象语言的一半。这两种范式都有很多优点，所以我认为值得介绍其中的一些。

由于类已经受到抨击，不辜负他们的炒作，让我首先解释为什么我把他们到 Lox 和这本书。实际上有两个问题：

# [_**为什么任何语言都希望是面向对象的？**_](https://craftinginterpreters.com/the-lox-language.html#why-might-any-language-want-to-be-object-oriented)

现在，像 Java 这样的面向对象语言已经销售一空，只能在竞技场表演，喜欢它们就不再酷了。为什么会有人做一个_新的_ 语言与对象？这不就像在8轨上发布音乐吗？

的确，90 年代的“所有的继承”热潮产生了一些可怕的类层次结构，但**面向对象编程** （**OOP**）仍然是非常棒的。数十亿行成功的代码都是用 OOP 语言编写的，为快乐的用户提供了数百万个应用程序。今天，可能大多数程序员都在使用面向对象语言。他们不可能_全都_错了。 特别是，对于动态类型语言，对象非常方便。我们需要_一些_定义复合数据类型的方法来将 blob 捆绑在一起

如果我们也可以将方法挂在这些函数上，那么我们就不需要在所有函数的前面加上它们所操作的数据类型的名称，以避免与不同类型的类似函数发生冲突。比如说，在 Racket 中，你最终不得不将函数命名为 `hash-copy`（复制哈希表）， `vector-copy`（复制矢量），这样它们就不会互相踩踏。方法的作用域是对象，所以问题就解决了。

# [_**为什么 Lox 是面向对象的？**_](https://craftinginterpreters.com/the-lox-language.html#why-is-lox-object-oriented)

我可以声称对象是 groovy，但仍然超出了本书的范围。大多数编程语言书籍，尤其是那些试图实现整个语言的书籍，都没有提到对象。对我来说，这意味着这个话题没有被很好地覆盖。有了这样一个广泛的范例，这种遗漏让我感到难过 考虑到我们中有多少人整天都在_使用_ OOP 语言，似乎这个世界可以使用一个关于如何_创建_一个 OOP 语言的小文档。正如你所看到的，这是非常有趣的。不像你担心的那么难，但也不像你想象的那么简单。

# [_**3 . 9 . 3Classes or prototypes  类或原型**_](https://craftinginterpreters.com/the-lox-language.html#classes-or-prototypes)

当涉及到对象时，实际上有两种,类和[原型](https://en.wikipedia.org/wiki/Prototype-based_programming) 。类首先出现，并且由于 C++，Java，C#和朋友而更加常见。原型实际上是一个被遗忘的分支，直到 JavaScript 意外地接管了世界。

在基于类的语言中，有两个核心概念：实例和类。实例存储每个对象的状态，并具有对实例类的引用。类包含方法和继承链。要调用实例上的方法，总是存在一定程度的间接。查找实例的类，然后找到方法 _其中：_

![](Pasted%20image%2020250527232811.png)

基于原型的语言融合了这两个概念。 只有对象-没有类-每个单独的对象可以包含状态和方法。 对象可以直接从彼此继承（或者用原型术语来说是"委托给"）
![](Pasted%20image%2020250527232822.png)

但是我已经看过很多用原型语言编写的代码，包括 [一些我自己设计的东西](http://finch.stuffwithstuff.com/)你知道人们通常用原型的所有功能和灵活性做什么吗？. . .&他们用它们来重新发明类。
这意味着在某些方面，原型语言比类更基本。它们的实现非常简洁，因为它们非常简单。 此外，它们可以表达许多不寻常的模式，类引导你远离。
我不知道为什么，但人们似乎自然更喜欢基于类（经典？ 优雅？）样式的原型在语言中更简单，但它们似乎只能通过将复杂性推给用户来实现这一点。 因此，对于 Lox，我们将为用户省去麻烦，直接在其中添加烘焙类。

# [_**3 . 9 . 4Classes in Lox  Lox 的课程**_](https://craftinginterpreters.com/the-lox-language.html#classes-in-lox)

有足够的理由，让我们看看我们实际上有什么。在大多数语言中，类包含一系列特征。对于 Lox，我选择了我认为最亮的星星。你可以像这样声明一个类和它的方法：

```
class Breakfast {
  cook() {
    print "Eggs a-fryin'!";
  }

  serve(who) {
    print "Enjoy your breakfast, " + who + ".";
  }
}
```

类的主体包含它的方法。它们看起来像函数声明，但没有 `fun` 关键字 。
**当执行类声明时，Lox 创建一个类对象，并将其存储在一个以类命名的变量中。就像函数一样，类在 Lox 中也是头等的**

```
// Store it in variables.
var someVariable = Breakfast;

// Pass it to functions.
someFunction(Breakfast);
```

```
// Store it in variables.
var someVariable = Breakfast;

// Pass it to functions.
someFunction(Breakfast);
```

接下来，我们需要一种创建实例的方法。我们可以增加`一些` 关键字，但为了简单起见，在 Lox 中，类**本身就是一个工厂 功能为例。像函数一样调用一个类，它会产生一个新的 本身的实例。**

```
var breakfast = Breakfast();
print breakfast; // "Breakfast instance".
```

# [_**实例化和初始化**_](https://craftinginterpreters.com/the-lox-language.html#instantiation-and-initialization)

只有行为的类不是超级有用的。面向对象编程背后的思想是将行为_和状态_封装在一起。为此，您需要字段。Lox 和其他动态类型语言一样，允许您自由地向对象添加属性。

```
breakfast.meat = "sausage";
breakfast.bread = "sourdough";
```

如果字段不存在，则将其复制到字段。

如果你想从**一个方法中访问当前对象上的一个字段或方法，**你可以使用老的 `this`。

```
class Breakfast {
  serve(who) {
    print "Enjoy your " + this.meat + " and " +
        this.bread + ", " + who + ".";
  }

  // ...
}
```

在对象中封装数据的一部分是确保对象在创建时处于有效状态。要做到这一点，你可以定义一个初始化器。如果你的类有一个名为 `init（）的`方法，那么在构造对象时会自动调用它。传递给类的任何参数都被转发给它的初始化器。

```
class Breakfast {
  init(meat, bread) {
    this.meat = meat;
    this.bread = bread;
  }

  // ...
}

var baconAndToast = Breakfast("bacon", "toast");
baconAndToast.serve("Dear Reader");
// "Enjoy your bacon and toast, Dear Reader."
```

# [_**Inheritance  继承**_](https://craftinginterpreters.com/the-lox-language.html#inheritance)

每一种面向对象的语言都允许你不仅定义方法，而且重用它们 跨多个类或对象。因此，Lox 支持单一继承。 声明类时，可以使用小于 （`<`） 操作员。

```
class Brunch < Breakfast {
  drink() {
    print "How about a Bloody Mary?";
  }
}
```

在这里，Brunch 是**派生类**或**子类** ，Breakfast 是 **基类**或**超类** 。 在**超类中定义的每个方法也可用于其子类。**

```
var benedict = Brunch("ham", "English muffin");
benedict.serve("Noble Reader");
```

`init（）` 方法也会被继承 。在实践中，子类通常也希望定义自己`的 init（）` 方法。但是也需要调用原来的类，这样超类才能保持它的状态。我们需要一些方法来调用我们自己_的实例_上的方法，而不影响我们自己_的方法_ 。

在 Java 中，你可以使用 `super`。

```
class Brunch < Breakfast {
  init(meat, bread, drink) {
    super.init(meat, bread);
    this.drink = drink;
  }
}
```

这就是面向对象。我试图保持功能集最小化。这本书的结构确实迫使一个妥协。Lox 不是一种_纯_ 面向对象语言在真正的 OOP 语言中，每个对象都是 一个类，甚至像数字和布尔值这样的原始值。

因为我们直到开始使用内置类型之后才实现类，这将是困难的。因此，基本类型的值不是真实的对象，因为它们是类的实例。它们没有方法或属性。如果我想让 Lox 成为真实的用户的真实的语言，我会解决这个问题。

# [**The Standard Library  标准库**](https://craftinginterpreters.com/the-lox-language.html#the-standard-library)

我们快好了这就是整个语言，所以剩下的就是“核心”或“标准”库 - 直接在解释器中实现的功能集，所有用户定义的行为都建立在其上。

这是 Lox 最悲哀的部分。它的标准库超越了极简主义，接近于彻底的虚无主义。对于本书中的示例代码，我们只需要演示代码正在运行并做它应该做的事情。为此，我们已经有了内置的 `print` 语句。

稍后，当我们开始优化时，我们将编写一些基准测试，看看执行代码需要多长时间。这意味着我们需要跟踪时间，所以我们将定义一个内置函数 `clock（）`，它返回程序启动后的秒数。

和 . . .&就这样。我知道，对吧？这太尴尬了。

如果你想把 Lox 变成一种真正有用的语言，你应该做的第一件事就是充实它。字符串操作，三角函数，文件 I/O，网络，甚至_阅读用户的输入_都会有所帮助。但我们在这本书中不需要这些，加上这些也不会教你什么有趣的东西，所以我把它们去掉了。 别担心，我们会有很多令人兴奋的东西在语言本身，让我们忙碌。