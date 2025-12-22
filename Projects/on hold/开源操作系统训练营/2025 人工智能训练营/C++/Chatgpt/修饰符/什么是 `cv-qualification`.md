## 🧾 一、什么是 `cv-qualification`？

`cv` 是 **const/volatile 修饰符的统称**，表示对函数中 **隐含的 `this` 指针的修饰**。

它的全称是：

```cpp
cv-qualification = const / volatile / const volatile
```

适用于 **非静态成员函数的声明与定义**。

---

## 🧠 二、cv 修饰的真正含义：修饰 `this` 指针的类型

```cpp
struct X {
    int get() const;
};
```

背后发生了什么？

等价于：

```cpp
int get(X const* this);  // 编译器自动翻译视图
```

也就是说：

-   普通成员函数：`X* this`
    
-   `const` 成员函数：`X const* this`
    
-   `volatile` 成员函数：`X volatile* this`
    
-   `const volatile` 成员函数：`X const volatile* this`
    

---

## 📌 三、用途与规则

### ✅ 1. 限制修改成员变量

```cpp
struct Account {
    int balance;

    int get_balance() const {
        return balance;  // ✅ OK
        // balance++;    // ❌ 错误：尝试修改 const 对象
    }
};
```

-   该函数**承诺不修改对象状态**；
    
-   编译器将拒绝你在 `const` 函数中修改成员变量。
    

### ✅ 2. 区分重载版本（函数重载基于 cv）

```cpp
class File {
public:
    std::string name() const { return _name; }
    std::string& name()       { return _name; }

private:
    std::string _name;
};
```

-   当对象是 `const File`，只能调用 `const` 版本；
    
-   普通对象调用非 `const` 版本，允许修改返回值。
    

---

## ⚠️ 四、只能用于非静态成员函数

```cpp
class A {
    static void foo() const; // ❌ 错误，static 成员函数不能加 const
};
```

原因是：

-   静态成员函数**没有 this 指针**；
    
-   所以也无所谓对 `this` 限定是否是 const。
    

---

## 🧪 五、示例：看一下编译器的行为

```cpp
struct Demo {
    int x;

    void set(int val) { x = val; }
    int get() const { return x; }
};

void f(const Demo& d) {
    d.get();     // ✅ OK
    // d.set(10);  // ❌ 错误，不能调用非常量成员函数
}
```

说明：

-   `d` 是 const 引用；
    
-   只能调用 const 成员函数；
    
-   若你声明 `int get()` 而非 `int get() const`，这段代码会编译失败。
    

---

## 📦 总结

| 表达式 | 实际含义 |
| --- | --- |
| `int f() const;` | `this` 是 `const X*`，不能修改成员 |
| `int f() volatile;` | `this` 是 `volatile X*`，用于多线程/IO |
| `int f() const volatile;` | `this` 是 `const volatile X*` |
| `int f();` | `this` 是 `X*`，可以修改成员 |

---

## 🧩 延伸：配合 `mutable` 使用

即使函数是 `const`，你也可以通过 `mutable` 成员突破限制：

```cpp
class Logger {
    mutable int counter = 0;

    void log() const {
        ++counter; // ✅ OK：因为 counter 是 mutable
    }
};
```

---