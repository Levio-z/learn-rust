### 🧱 1. 动态异常说明（C++98/C++03，**已废弃**）

```cpp
void f() throw(int, std::string);
```
-   表示 `f` **可能抛出** `int` 或 `std::string` 类型的异常。
-   如果抛出不在列表中的异常，则调用 `std::unexpected()`。
-   C++98 设计初衷是为了进行异常类型检查，但因运行时开销大且灵活性差，被废弃。

#### 🚫 C++11 起标记为 deprecated，C++17 正式移除。

---

### 🧊 2. `noexcept`（C++11 引入，C++17 起唯一合法异常说明）

```cpp
void f() noexcept;             // 声明不会抛出异常（编译器可优化）
void g() noexcept(false);      // 明确表示可能抛出异常（等同于无 noexcept）
```

-   `noexcept` 是 **编译期常量表达式**，可用作条件：

```cpp
template <typename T>
void call(T t) noexcept(noexcept(t()));
```
- 外层 `noexcept(...)` —— 异常说明
	- 这是**条件异常说明**：表示当 `表达式` 的值为 `true` 时，`call` 函数被标记为 `noexcept`（即不会抛异常）。
	- 如果值为 `false`，则 `call` 被认为 **可能抛异常**（即无 `noexcept` 修饰）。
- 内层 `noexcept(t())` —— 编译期表达式
	- 这是 `noexcept` 作为**编译期运算符**的用法，类似于 `sizeof()` 或 `decltype()`。
	- 它的作用是判断 **表达式 `t()` 是否会抛异常**：
- 总
	- 如果 `t()` **不会抛异常**（即 `noexcept(t()) == true`），  那么 `call(t)` 函数就标记为 `noexcept`。
	- `call` 是否加 `noexcept`，**取决于传进来的 `t` 这个可调用对象是否 noexcept**。
-   用于 **启用 move 构造函数 / move 赋值优化**：

```cpp
class A {
public:
    A(A&&) noexcept;  // 如果不加 noexcept，某些 STL 容器会退回使用拷贝构造
};
```
[STL：能安全地移动就移动，否则退回复制](../容器/容器的构造/STL：能安全地移动就移动，否则退回复制.md)


### 🧪 3. `noexcept` 与函数类型的区别（函数类型 vs 声明）

```cpp
using Fn = void() noexcept;     // 函数类型本身包含 noexcept
void f() noexcept;              // 函数声明
```

> 这意味着 `noexcept` 是函数类型的一部分，可以参与重载和模板推导。

```cpp
void foo() noexcept;
void foo(); // 与上不同，不能重定义为相同函数！
```

---

## 🔍 扩展：判断是否为 `noexcept` 函数

```cpp
#include <type_traits>

template<typename T>
void check() {
    std::cout << std::boolalpha << noexcept(T()) << '\n';
}
```

或者直接在编译时判断：

```cpp
static_assert(noexcept(Foo()), "Foo() must be noexcept!");
```

---

## 🧠 总结对比表

| 特性 | `throw()` / `throw(T)` | `noexcept` |
| --- | --- | --- |
| 引入版本 | C++98 | C++11 |
| 状态 | C++11 弃用，C++17 删除 | C++11 引入，C++17 起唯一合法 |
| 编译时检测 | ❌（主要是运行时检查） | ✅ 完全编译期分析 |
| 可用于条件推导 | ❌ | ✅ `noexcept(expr)` |
| 性能优化（如移动优化） | ❌ | ✅ STL 容器依赖此判断 |

---

### 🎯 使用场景推荐

| 场景 | 推荐方式 |
| --- | --- |
| STL 容器、move 构造函数优化 | `noexcept` 必须显式声明 |
| 模板中条件 noexcept 推导 | 使用 `noexcept(expr)` |
| 错误恢复控制 / 自定义策略 | 用 `try` + `catch` 控制异常行为 |
