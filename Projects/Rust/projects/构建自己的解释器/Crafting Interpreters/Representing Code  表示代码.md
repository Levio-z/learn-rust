在[上一章](https://craftinginterpreters.com/scanning.html)中，**我们将原始源代码作为字符串，并将其转换为稍微高级的表示：一系列标记。**我们将[在下一章](https://craftinginterpreters.com/parsing-expressions.html)中编写的解析器将接受这些标记，并再次将它们转换为更丰富、更复杂的表示。.
在我们产生这种表示之前，我们需要定义它，这是本章的主题。沿着，我们将涵盖 一些关于形式语法的理论，感觉功能语法和 面向对象编程，复习几种设计模式，并做一些 元编程

在我们做所有这些之前，让我们关注一下主要目标 -- 代码的表示。解析器应该很容易生成，解释器也应该很容易使用。如果您还没有编写过解析器或解释器，那么这些需求并不具有启发性。也许你的直觉能帮上忙。当你扮演_人类_翻译时，你的大脑在做什么？你如何在心里评估这样一个算术表达式：
1 + 2 * 3 - 4
因为你理解了运算的顺序 -- 古老的“ [请原谅我亲爱的萨莉阿姨](https://en.wikipedia.org/wiki/Order_of_operations#Mnemonics) “的东西 -- 你知道乘法在加法或减法之前计算。将优先顺序可视化的一种方法是使用树。叶节点是数字，内部节点是每个操作数都有分支的运算符。
为了计算一个算术节点，你需要知道它的子树的数值，所以你必须先计算这些数值。这意味着从叶子到根的遍历 - 一个_后序_遍历：
![](Pasted%20image%2020250601205336.png)

如果我给你一个算术表达式，你可以很容易地画出这些树。给定一棵树，你可以毫不费力地评估它。因此，直观地看起来，我们代码的一个可行表示是一个与语言的语法结构 （ 操作符嵌套 ） 相匹配的树 。

我们需要更精确地了解这种语法是什么。就像上一章的词汇语法一样，关于句法语法有大量的理论。我们会比在扫描时更深入地探讨这个理论，因为它在解释器的大部分内容中都是一个有用的工具。我们先从乔姆斯基的等级结构上移一级 [](https://en.wikipedia.org/wiki/Chomsky_hierarchy)。. .& nbsp;你好

### ## [Context-Free Grammars  上下文无关文法](https://craftinginterpreters.com/representing-code.html#context-free-grammars)
在上一章中，我们用来定义词汇语法的形式体系 -- **字符如何被分组为标记的规则** -- 被称为_正则语言_ 。这对我们的扫描器来说是好的，它发出一个平坦的令牌序列。但是正则语言没有强大到足以处理可以任意嵌套的表达式。

我们需要一个更大的锤子，而那个锤子就是**上下文无关文法** （**CFG**）。它是下一个最重的工具， **[形式](https://en.wikipedia.org/wiki/Formal_grammar)**语法形式文法采用一组原子片段，它称之为“字母表”。然后它定义了一组（通常是无限的）语法中的“字符串”。每个字符串都是字母表中的“字母”序列。

我之所以使用这些引号，是因为当你从词汇语法转向句法语法时，这些术语会有点令人困惑。在我们的扫描器的语法中，字母表由单个字符组成，字符串是有效的词素 - 大致为“单词”。在我们现在讨论的句法语法中，我们处于不同的粒度级别。现在，字母表中的每个“字母”都是一个完整的符号，而“字符串”是一_个符号_序列 -- 一个完整的表达式。
### **术语对照表**

| Terminology (术语)              | Lexical grammar (词法语法) | Syntactic grammar (句法语法) |
| ----------------------------- | ---------------------- | ------------------------ |
| The “alphabet” is . . .       | Characters（字符）         | Tokens（记号、词法单元）          |
| A “string” is . . .           | Lexeme or token（词素或记号） | Expression（表达式）          |
| It’s implemented by the . . . | Scanner（扫描器/词法分析器）     | Parser（解析器/语法分析器）        |

形式语法的工作是指定哪些字符串是有效的，哪些不是。如果我们要为英语句子定义一个语法，“eggs are tasty for breakfast”应该在语法中，但“tasty breakfast for are eggs”可能不在语法中。

### ### [Rules for grammars  语法规则](https://craftinginterpreters.com/representing-code.html#rules-for-grammars)
1️⃣ **形式语法（Formal Grammar）**
![](Pasted%20image%2020250601205944.png)
2️⃣ **什么是推导（derivation）？**
推导就是从 SSS 出发，**反复应用产生式规则**，一步步替换非终结符，直到得到完全由终结符组成的字符串。

比如：
S → aSb | ε
S ⇒ aSb ⇒ aaSbb ⇒ aaεbb ⇒ aabb

这个过程每一步都称作 **派生步骤（derivation step）**。

3️⃣ **为什么叫“产生式”？**  
因为这些规则本质上就是：
- **从非终结符产生新字符串的指令。**
- 每次应用产生式，就把语法中潜在的某个表达式“生成”出来。
所以：
- 语法规则 ≈ 产生器（generator）。
- 推导出来的字符串 ≈ 语法所定义的语言中的句子。


**如果您从规则开始，则可以使用它们_生成_语法中的字符串**。以这种方式创建的字符串称为**派生，** 因为每个字符串都是 从语法规则中_衍生出来的_ 。在游戏的每一步，你选择一个 统治并遵循它告诉你要做的事情。大多数正式场合的行话 语法是从这个方向来的。规则被称为 因为它们在语法中_产生_字符串。
上下文无关文法中的每个产生式都有一个**头** （ 它的名字 ） 和一个**体** （描述它生成的内容）。在纯粹的形式中，身体只是一系列符号。符号有两种美味：4
#### **产生式（Production Rule）**

每个产生式（也叫规则）由两部分组成：
- **头（Head）**：唯一一个 **非终结符**（nonterminal）。
- **体（Body）**：一串 **符号序列**，可以包含终结符（terminal）和非终结符。
A → α
- A 是非终结符。
- α是由终结符和非终结符组成的序列
⚠ **关键定义**：  
上下文无关文法 **严格要求** 左边（head）只能是单个非终结符。  
更强的形式（如无限制文法、上下文相关文法）则允许左边出现符号序列。
#### **终结符（Terminal）**
- 语法字母表中的字母（不可再被替换）。
- 类比为 **字面量值**。
- 在编译器中，对应词法分析器（scanner）产出的 **token**，
- 
- 比如：
- `if`
- `1234`
- `+`
称为“terminal”，是因为它们是推导的终点，不再展开。
#### 非终结符（Nonterminal）
- 对语法中其他规则的 **命名引用**。
- 类比为 **占位符**，需要被替换展开。
- 编译器中对应的是语法分析器（parser）识别的高层结构：
	- `Expression`
	- `Statement`
	- `IfClause`
非终结符 **驱动了语法的递归构造能力**，实现复合表达式。
#### 多重定义（Multiple Rules per Nonterminal）

Expr → Expr "+" Term ;
Expr → Term ;
Term → "NUMBER" ;
当推导到 `Expr` 时，可以任选其一。
这种设计提供了语法的 **选择性（non-determinism）**。、

#### 表示方法：Backus-Naur Form (BNF)
BNF 表示法：

- 非终结符：小写单词（或用尖括号包裹，如 `<expr>`）。
- 终结符：带引号的字符串。
- 规则结构：`name → body ;`

```rust
expr → term "+" expr ;
expr → term ;
term → "NUMBER" ;
```

利用这一点，这里有一个早餐菜单的语法：
```rust
breakfast  → protein "with" breakfast "on the side" ;
breakfast  → protein ;
breakfast  → bread ;

protein    → crispiness "crispy" "bacon" ;
protein    → "sausage" ;
protein    → cooked "eggs" ;

crispiness → "really" ;
crispiness → "really" crispiness ;

cooked     → "scrambled" ;
cooked     → "poached" ;
cooked     → "fried" ;

bread      → "toast" ;
bread      → "biscuits" ;
bread      → "English muffin" ;
```
我们可以使用这个语法来生成随机早餐。我们来玩一局看看效果如何。按照古老的惯例，游戏从语法中的第一条规则开始，这里是`breakfast` 。有三个产品，我们随机选择第一个。我们得到的字符串看起来像：
`protein "with" breakfast "on the side"`
我们需要扩展第一个非末端`蛋白质` ，所以我们为它选择一个产物。让我们挑选：
`protein → cooked "eggs" ;`
接下来，我们需要一`个 cooked` 的生产，所以我们选择了 `“poached”`。这是一个终端，所以我们添加它。现在我们的字符串看起来像：
`"poached" "eggs" "with" breakfast "on the side"`
下一个非终点站又是`breakfast` 。我们选择的第一个`breakfast`生产递归地引用了`breakfast`规则。
- 语法中的递归是一个很好的迹象，表明所定义的语言是上下文无关的，而不是常规的。特别是，递归非终结符两边都有结果的递归意味着语言不是正则的。

这样，字符串中的每个非终结符都被扩展，直到它最终只包含终结符，我们剩下：![](Pasted%20image%2020250601210829.png)

任何时候我们遇到一个有多个产品的规则，我们只是任意选择一个。正是这种灵活性允许使用少量的语法规则来编码组合上更大的字符串集合。规则可以直接或间接地引用自身， 这一事实使它变得更加强大，让我们可以将无限数量的字符串打包到一个有限的语法中。
### ### [Enhancing our notation  增强我们的符号](https://craftinginterpreters.com/representing-code.html#enhancing-our-notation)

将一个无限的字符串集合填充到少数几个规则中是非常奇妙的，但让我们更进一步。我们的记谱法很好用，但很繁琐。因此，像任何优秀的语言设计者一样，我们将在上面撒上一点语法糖 - 一些额外的方便符号。除了终结符和非终结符之外，我们还允许在规则体中使用一些其他类型的表达式：

我们将允许一系列由管道分隔的产品，而不是每次为它添加另一个产品时重复规则名称（`|`).

```
bread → "toast" | "biscuits" | "English muffin" ;
```

此外，我们将允许括号进行分组，然后允许 `|` 在制作过程中从一系列选项中选择一个。
```
protein → ( "scrambled" | "poached" | "fried" ) "eggs" ;
```
使用递归来支持重复的符号序列具有一定的吸引力 ，但是每次我们想要循环时都要创建一个单独的命名子规则，这有点麻烦。因此，我们也使用后缀 `*` 来允许前面的符号或组重复零次或多次。
```
crispiness → "really" "really"* ;
```

后缀 `+` 类似，但要求前面的产品至少出现一次。
```
crispiness → "really"+ ;
```
后缀 `？` 是一个可选的生产。在它之前的事物可以出现零次或一次，但不能出现更多。
```
breakfast → protein ( "with" breakfast "on the side" )? ;
```

有了所有这些语法细节，我们的早餐语法可以浓缩为：
```
breakfast → protein ( "with" breakfast "on the side" )?
          | bread ;

protein   → "really"+ "crispy" "bacon"
          | "sausage"
          | ( "scrambled" | "poached" | "fried" ) "eggs" ;

bread     → "toast" | "biscuits" | "English muffin" ;
```

我们将在本书的其余部分使用这种符号来精确地描述 Lox 的语法。当您使用编程语言时，您会发现上下文无关语法（使用 this 或 [EBNF](https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form) 或其他一些表示法）可以帮助您明确非正式的语法设计思想。它们也是与其他语言黑客交流语法的方便媒介。

我们为 Lox 定义的规则和产生式也是我们将要实现的表示内存中代码的树数据结构的指南。在我们这样做之前，我们需要一个实际的 Lox 语法，或者至少足够让我们开始。
#### **递归 vs. 正则语言**

让我们定位到形式语言的四大类（Chomsky hierarchy）：

- **Type 3（正则语言）**：
    
    - 用有限状态自动机（FSM）表示。
        
    - 没有跨层匹配能力，不能处理递归。
        
    - 典型例子：正则表达式（regex）。
        
- **Type 2（上下文无关语言）**：
    
    - 用上下文无关文法（CFG）和下推自动机（PDA）表示。
        
    - **允许递归产生式**，可表达像括号匹配、嵌套结构等。
当一个语言需要**平衡的嵌套**或**成对结构**（如：  
anbna^n b^nanbn，即 nnn 个 a 后接 nnn 个 b），它就超出了正则语言的表达能力，必须依赖 CFG。

因此：

> **递归是上下文无关语言的核心标志之一**，特别是双边递归（左右都有非终结符）的存在，是非正则的直接证据。

### ### [Lox 表达式的语法](https://craftinginterpreters.com/representing-code.html#a-grammar-for-lox-expressions)

在前一章中，我们一下子完成了 Lox 的整个词汇语法。每一个关键词和标点符号都在那里。句法语法比较大，在我们真正启动并运行解释器之前，要把整个东西都磨一遍是一件真实的无聊的事。

相反，我们将在接下来的几章中粗略地介绍这种语言的一个子集。一旦我们表示、解析和解释了这种迷你语言，后面的章节将逐步为其添加新的功能，包括新的语法。现在，我们只关心几个表达式：
- **字面意思。** 数字、字符串、布尔值和 `nil`。
- **一元表达式。** 前缀 `！` 来执行逻辑“非”运算， `然后`对一个数字求反
- **二进制表达式。** 中缀算术运算符（`+`、`-`、`*`、`/`）和逻辑运算符（`==`、`！=` ，`<`，`<=`，`>`，`>=`）我们认识并喜爱。
- **括号。** 一对 `（` 和 `）` 包裹在一个表达式周围。
这为我们提供了足够的语法表达式，如：
```
1 - (2 * 3) < 4 == false
```
使用我们方便的新符号，这里有一个语法：
```rust
expression     → literal
               | unary
               | binary
               | grouping ;

literal        → NUMBER | STRING | "true" | "false" | "nil" ;
grouping       → "(" expression ")" ;
unary          → ( "-" | "!" ) expression ;
binary         → expression operator expression ;
operator       → "==" | "!=" | "<" | "<=" | ">" | ">="
               | "+"  | "-"  | "*" | "/" ;
```

这里有一点额外的金属税 。除了与词素完全匹配的终端的引号字符串之外，我们`还将 CAPITALIZE` 终端是其文本表示可以变化的单个词素。`NUMBER` 是任何数字字面量，而 `STRING` 是任何字符串字面量。稍后，我们将为 `IDENTIFIER` 做同样的事情。

> **expression → literal | unary | binary | grouping ;**

这是主规则：  
**表达式（expression）** 可以是：
- 字面量（literal）
- 一元表达式（unary）
- 二元表达式（binary）
- 分组（grouping）
用 BNF 定义就是用 `→` 表示“可以展开成”，`|` 表示“或者”。

---

> **literal → NUMBER | STRING | "true" | "false" | "nil" ;**

字面量可以是：
- 数字字面量（NUMBER）
- 字符串字面量（STRING）
- 布尔值 true、false
- 空值 nil

---

> **grouping → "(" expression ")" ;**

分组，就是用括号包起来的表达式。

---

> **unary → ( "-" | "!" ) expression ;**

一元表达式，就是前面加上负号 `-` 或逻辑非 `!` 的表达式。

---

> **binary → expression operator expression ;**

二元表达式，就是：
表达式 operator 表达式
比如 `1 + 2`、`a == b`。

除了用引号表示那些需要**精确匹配的终端符号**（比如 `"=="`、`"("` 这种固定字符），  
我们还用全大写单词（如 `NUMBER`、`STRING`）表示**单个词素但文本表示可能变化的终端符号**。

换句话说：

- `"=="` → 固定字面符号（固定拼写）
    
- `NUMBER` → 只代表一种词法类别（任意数字都可以，具体内容变化）

这个语法实际上是二义性的，我们在解析它的时候会看到这一点，但是现在已经足够好了。

`NUMBER`：任何数字字面量（比如 `42`、`3.14`）。  
`STRING`：任何字符串字面量（比如 `"hello"`、`"abc"`）。

它们不是写死的，而是抽象类别。

稍后我们也会对 **标识符（IDENTIFIER）** 做同样的处理：  
用大写 `IDENTIFIER` 表示“任何合法的标识符”而不是具体拼写。
### [Implementing Syntax Trees  实现树](https://craftinginterpreters.com/representing-code.html#implementing-syntax-trees)

最后，我们要写一些代码。这个小小的表达语法就是我们的骨架。由于语法是递归的， 请注意`分组` 、 `一元`和 `二进制`所有引用回`表达式` - 我们的数据结构将形成一棵树。因为这个结构代表了我们语言的语法，所以它被称为**语法树** 。
1️⃣ **什么是抽象语法树（AST）？**  
AST 是源代码的结构化表示，比直接的源代码更适合后续处理（比如语义分析、优化、代码生成等）。  
区别于：
- **Parse Tree（解析树）**：忠实还原语法产生式的每一个步骤，包括中间非终结符。
- **AST**：只保留对后续阶段有用的语法信息，省略无用节点。
例如：  
对于 `1 + 2 * 3`，
- 解析树保留了完整的层次（包括加法、乘法的每一层次细节）。
- AST 只需要表示 `+`，其左子树是 `1`，右子树是 `*`，其左右子树是 `2` 和 `3`。
- 
我们_可以_将它们混合到一个带有任意子级列表的 Expression 类中。有些编译器会。但是我喜欢从 Java 的类型系统中得到最大的好处。所以我们将为**表达式定义一个基类。然后，对于每一种表达式 （`expression` 下的每一个产生式 ）， 我们创建一个子类，该子类具有特定于该规则的非终结符字段。**
这样，如果我们试图访问一个一元表达式的第二个操作数，就会得到一个编译错误。
```java
package com.craftinginterpreters.lox;

abstract class Expr { 
  static class Binary extends Expr {
    Binary(Expr left, Token operator, Expr right) {
      this.left = left;
      this.operator = operator;
      this.right = right;
    }

    final Expr left;
    final Token operator;
    final Expr right;
  }

  // Other expressions...
}
```
Expr 是所有表达式类继承的基类。正如你在 `Binary` 中看到的，子类嵌套在它里面，这在技术上没有必要，但它允许我们把所有的类都塞进一个 Java 文件中。
这种设计最大化利用 Java 的类型系统：
1. **类型安全**  
    每个表达式的子类都显式声明其需要的字段和结构，比如 `BinaryExpression` 一定有 `left` 和 `right`，`LiteralExpression` 一定有 `value`。
2. **可维护性与可扩展性**  
    如果未来要扩展表达式（比如支持三目运算符、条件表达式等），只需增加一个新子类，不影响已有代码。
3. **避免通用容器的不必要复杂性**  
    如果你用一个统一的 `List<Expression>` 存储所有子节点，你失去了对不同类型结构的区分，检查时只能动态判断类型，增加出错率。
这种设计基于 **抽象语法树（AST）建模**：
- 每个语法规则的产生式对应一个类。
- 组合关系通过对象引用（子节点）表示。
- 运行时可以用 **多态**（`instanceof` 或 **访问者模式**）区分和处理不同子类。
    

它和 **组合模式（Composite Pattern）** 紧密相关。  
AST 本质上就是一个典型的组合结构
- 节点（表达式、语句等）
- 子节点（根据类型可以是零个、一对、或者多个）

### ### [5 . 2 . 1Disoriented objects  迷失方向的物体](https://craftinginterpreters.com/representing-code.html#disoriented-objects)

您会注意到，与 Token 类非常相似，这里没有任何方法。这是一个愚蠢的结构。打字很好，但只是一袋数据。这在像 Java 这样的面向对象语言中感觉很奇怪。班级不应该_做点什么_吗？

问题是这些树类不属于任何一个域。它们是否应该有解析的方法，因为树是在那里创建的？或者翻译，因为那是他们消费的地方？树木跨越这两个地区之间的边界，这意味着它们实际上_不属于任何_一方。

`Token` 类、`Expression` 类（AST 节点类）这些都是“**袋状数据**”（bag of data）：

- 它们内部几乎只有字段，没有方法。
- 它们的职责只是 **承载信息**，供其他系统（如解析器、解释器、代码生成器）消费。

实际上，这些类型的存在是为了使解析器和解释器能够 _communicate_.这使其适合于仅仅是没有关联行为的数据的类型。这种风格在 Lisp 和 ML 等函数式语言中非常自然， _所有_数据都与行为分离，但在 Java 中感觉很奇怪。

- 它们既不属于解析器（虽然解析器创建它们），也不属于解释器（虽然解释器消费它们）。
- 它们是两者之间的“中立区域”。

**职责单一化（Single Responsibility Principle, SRP）**  
如果你把解析、翻译、执行等行为塞进 AST 节点：

- 那就意味着 AST 节点必须知道它们是要被如何使用（比如解释还是编译？）。
- 会引入 **耦合**，使得 AST 无法被多个系统独立复用。

在编译器设计里，即使用 OOP 语言（Java、C++），中间结构通常用的是“纯数据”风格，因为：
- 这样可以让不同阶段（前端、优化、后端）用不同方式处理数据，而不把处理逻辑硬编码到数据对象本身。



函数式编程的爱好者们现在都跳起来惊呼：“看！面向对象语言不适合解释器！”我不会那么过分你会记得扫描仪本身非常适合面向对象。它有所有的可变状态来跟踪它在源代码中的位置，一组定义良好的公共方法，以及一些私有助手。

我的感觉是，解释器的每个阶段或部分在面向对象的风格下工作得很好。正是在它们之间流动的数据结构被剥夺了行为。

### ### [Metaprogramming the trees](https://craftinginterpreters.com/representing-code.html#metaprogramming-the-trees)
### [对树进行元编程](https://craftinginterpreters.com/representing-code.html#metaprogramming-the-trees)


