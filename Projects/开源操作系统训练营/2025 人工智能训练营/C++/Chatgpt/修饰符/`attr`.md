`attr`（attributes，属性）是 **C++11 起引入的新语言机制**，用于**向编译器传达额外的元信息**，比如：

-   这个函数的返回值不能被忽略；
-   这个函数或变量可能废弃（deprecate）；
-   编译器应发出警告或做优化；
-   一些 ABI、对齐、内联控制指令等。

---

## ✅ 一、什么是 `[[attribute]]`（属性）

属性（`attribute`）是用来告诉编译器：

> “这段代码我不是写错了，我是有意为之”，或者 “这个函数/变量有特殊含义，你应该特别对待”。

其语法形式为：

```cpp
[[attribute_name]]
[[attribute_name(parameters)]]
```

例如：

```cpp
[[nodiscard]] int compute();            // 表示返回值不能被忽略
[[deprecated]] void old_api();          // 表示此函数已废弃
[[noreturn]] void panic();              // 表示此函数不会返回
```

---

## 🔎 二、常见标准属性列表（C++11 起）

| 属性 | 引入版本 | 用途说明 |
| --- | --- | --- |
| `[[nodiscard]]` | C++17 | 若调用者忽略返回值，编译器给出警告 |
| `[[noreturn]]` | C++11 | 表示函数不会返回（如 `exit`, `abort`） |
| `[[deprecated]]` | C++14 | 表示某函数/变量已废弃，建议不再使用 |
| `[[maybe_unused]]` | C++17 | 防止未使用的变量或函数产生警告 |
| `[[likely]]`, `[[unlikely]]` | C++20 | 用于 if 语句分支预测提示 |
| `[[gnu::always_inline]]` | GCC 扩展 | 强制内联函数（非标准） |

---

## 🎯 三、示例详解

### ✅ `[[nodiscard]]`

```cpp
[[nodiscard]] int result();

int main() {
    result(); // ⚠️ 警告：忽略了 [[nodiscard]] 函数的返回值
}
```

> 意图：返回值很重要，不应被悄悄忽略，比如错误码、计算结果。

### ✅ `[[noreturn]]`

```cpp
[[noreturn]] void fatal() {
    throw std::runtime_error("fatal error");
}
```

> 编译器知道这个函数**不会返回**，因此可优化调用后的代码路径。

### ✅ `[[deprecated("reason")]]`

```cpp
[[deprecated("use new_func() instead")]]
void old_func();
```