typedef 声明是具有作为存储类的 typedef 的声明。 声明符将成为新类型。 可以使用 typedef 声明为已由 C 定义的类型或你已声明的类型**构造更短和更有意义的名称**。 利用 Typedef 名称，您可以**封装可能会发生更改的实现详细信息**。

typedef 声明的解释方式与变量或函数声明的解释方式相同，只不过标识符没有假定由声明指定的类型，而是成为了该类型的同义词。

1. **declaration（声明）**：
```plaintext
    declaration-specifiers opt init-declarator-list;
 ```
这表示一个声明由可选的声明说明符（declaration-specifiers）和初始化声明符列表（init-declarator-list）组成，最后以分号结尾。`opt`表示该部分是可选的。
1. **declaration-specifiers（声明说明符）**：  
    声明说明符可以是以下任意一种组合，按任意顺序出现：
    - 存储类说明符（storage-class-specifier）加上可选的声明说明符
    - 类型说明符（type-specifier）加上可选的声明说明符
    - 类型限定符（type-qualifier）加上可选的声明说明符
2. **storage-class-specifier（存储类说明符）**：  
    这里列出的是`typedef`，它用于为已有类型创建新的名称（类型别名）。  
    （注：完整的 C 语言标准中还有其他存储类说明符，如 auto、static、extern、register 等）
    
3. **type-specifier（类型说明符）**：  
    这些是 C 语言中的基本数据类型和类型相关说明符：
    - 基本类型：void, char, short, int, long, float, double
    - 符号修饰符：signed, unsigned（用于修饰整数类型）
    - 结构或联合说明符：struct-or-union-specifier
    - 枚举说明符：enum-specifier
    - 类型定义名：typedef-name（即通过 typedef 定义的类型别名）
4. **typedef-name（类型定义名）**：
    ```plaintext
    identifier
    ```  
    表示通过 typedef 定义的类型别名是一个标识符（identifier），也就是一个符合 C 语言命名规则的名字。

    

  

简单来说，这些规则描述了在 C 语言中如何声明变量或类型。例如：

  

- `int a;` 中，`int`是 type-specifier，`a`是 init-declarator-list 的一部分
- `typedef int Integer;` 中，`typedef`是 storage-class-specifier，`int`是 type-specifier，`Integer`是 typedef-name
- `const unsigned long b;` 中，`const`是 type-qualifier，`unsigned long`是 type-specifier，`b`是声明的变量名


typedef 声明不会创建新类型。 而是创建现有类型的同义词或可通过其他方式指定的类型的名称。 当使用 typedef 名称作为类型说明符时，可以将其与特定的类型说明符组合，但不可以将其与其他类型说明符组合。 可接受的修饰符包括 **`const`** 和 **`volatile`**。



