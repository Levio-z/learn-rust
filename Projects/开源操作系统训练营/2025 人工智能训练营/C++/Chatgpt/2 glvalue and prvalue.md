- https://en.cppreference.com/w/cpp/language/value_category.html
-  a [glvalue](https://en.cppreference.com/w/cpp/language/value_category.html#glvalue) (“generalized” lvalue) is an expression whose evaluation determines the identity of an object or function;  
    [gl 值](https://en.cppreference.com/w/cpp/language/value_category.html#glvalue) （“广义”左值）是一个表达式，其计算确定对象或函数的身份;
- a [prvalue](https://en.cppreference.com/w/cpp/language/value_category.html#prvalue) (“pure” rvalue) is an expression whose evaluation  
    [pr 值](https://en.cppreference.com/w/cpp/language/value_category.html#prvalue) （“纯”右值）是一个表达式，其计算值
	- computes the value of an operand of a built-in operator (such prvalue has no _result object_), or  
	    计值（此类 prv算内置运算符的操作数的alue 没有_结果对象_ ），或
	- initializes an object (such prvalue is said to have a _result object_).  
	    初始化一个对象（这样的 prvalue 被称为具有_结果对象_ ）。


**中文**：计算内置运算符的操作数的值（这种 prvalue 没有 *结果对象*）

#### ✅ 解释：

当你在执行 **内置运算符**（比如 `+`、`*`、`<` 等）时，操作数如果是一个临时值，那么它是一个 **prvalue**。但这个 prvalue **并没有绑定到任何变量或内存地址**，所以也就不存在“结果对象”（*result object*）。

🧠 关键点是：

-   它的值存在于计算过程中，但没有物理存储（如变量或对象）。
    
-   比如 `1 + 2` 中，`1` 和 `2` 都是 prvalue，`1 + 2` 的结果也是一个 prvalue。
    

---

> **initializes an object (such prvalue is said to have a result object).**

**中文**：用于初始化一个对象（这种 prvalue 被称为具有 *结果对象*）

#### ✅ 解释：

有时候 prvalue 会被用来初始化变量或临时对象，那么此时它的值就会 **物化（materialize）** 成一个实际存在的对象，也就称为“具有结果对象”。

🔍 示例：

```cpp
int x = 1 + 2;  // 1+2 是 prvalue，用来初始化 x，此时就有结果对象
```

或：

```cpp
std::string s = std::string("hello") + "world";
```

这里 `std::string("hello") + "world"` 是 prvalue，但它被用来初始化变量 `s`，于是会生成一个结果对象。

---

### ✅ 总结对比表：

| 用途 | 是否有结果对象 | 示例 | 解释 |
| --- | --- | --- | --- |
| 运算中使用 | ❌ 否 | `1 + 2` | 计算用的临时值，无对象被创建 |
| 初始化对象 | ✅ 是 | `int x = 1 + 2;` | 被用来初始化变量，产生结果对象 |

---