
## ✅ 一、ref-qualification 是什么？

ref-qualification 是 `&` 或 `&&` **引用限定符**，用于**非静态成员函数的声明或定义**，限定这些成员函数只能通过某类对象（左值或右值）调用。

```cpp
struct X {
    void f() &;   // 左值限定
    void f() &&;  // 右值限定
};
```

这两个版本是函数重载，编译器根据 `this` 的值类别（是左值还是右值）进行选择。

---

## 📌 二、基本规则

| ref-qualified 函数 | 限定调用对象的值类别 |
| --- | --- |
| `void f() &` | 只能被左值对象调用（如变量名） |
| `void f() &&` | 只能被右值对象调用（如 `std::move(x)`） |
| 无 ref-qualifier | 默认可用于左值和右值对象 |

---

## 🧠 三、为什么需要 ref-qualification？

主要用于**区分左值对象和右值对象的操作行为**：

-   左值对象可多次使用，适合返回引用、延迟操作；
    
-   右值对象是临时的、可被“吃掉”，适合转移资源或优化性能。
    

### 示例：只允许右值调用 `consume()` 函数

```cpp
class Buffer {
public:
    void consume() && {
        // 可以安全地"消耗"资源
        std::cout << "Buffer consumed.\n";
    }
};

Buffer get_buffer();

get_buffer().consume(); // ✅ 右值调用成功
Buffer b;
b.consume();            // ❌ 编译错误：左值不能调用 && 限定函数
```

---

## 🔍 四、编译器选择规则（简化版）

```cpp
struct S {
    void f() &;
    void f() &&;
};
```

调用语句：

| 表达式类型 | 选择版本 |
| --- | --- |
| `S s; s.f();` | 选择 `f() &` |
| `S{}.f();` | 选择 `f() &&` |
| `std::move(s).f();` | 选择 `f() &&` |

---

## 🎯 五、典型用途场景

### ✅ 1. 实现移动语义

```cpp
struct MyString {
    std::string data;

    std::string&& str() && { return std::move(data); } // 只在右值对象上允许 move 出资源
    const std::string& str() const& { return data; }   // 左值调用返回 const 引用
};
```

调用行为：

```cpp
MyString s;
auto a = s.str();             // 返回 const 引用（左值）
auto b = MyString{}.str();    // 返回右值引用，资源被 move
```

---

### ✅ 2. 延迟计算返回值，区分使用场景

```cpp
struct LazyValue {
    std::string compute() && {
        return "expensive calculation";  // 临时对象，move 返回
    }

    std::string compute() const& {
        return "cached value";           // 保守做法，不 move
    }
};
```

---

## ⚠️ 六、限制说明

-   ref-qualifier **只适用于非静态成员函数**
    
-   不能与 static 成员函数或自由函数联用
    
    ```cpp
    struct Bad {
        static void foo() &; // ❌ 非法，static 函数无 this 指针
    };
    ```
    

---

## 📚 七、与 `cv` 限定符组合使用

```cpp
struct Demo {
    void f() const&;   // const 左值调用
    void f() &&;       // 右值调用
};
```

这允许更细粒度地控制函数重载匹配。

---

## ✅ 总结

| 函数签名 | 适用对象值类别 | 典型用途 |
| --- | --- | --- |
| `void f() &` | 左值对象 | 安全操作、引用返回 |
| `void f() &&` | 右值对象 | 转移资源、临时优化 |
| `void f() const&` | const 左值 | 只读访问、缓存读取等 |
| `void f() const&&` | const 右值 | 少见，限制右值上的只读操作 |

---