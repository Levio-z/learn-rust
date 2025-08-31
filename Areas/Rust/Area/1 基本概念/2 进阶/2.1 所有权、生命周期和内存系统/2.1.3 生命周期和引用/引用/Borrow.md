 - 官网： K: [Borrow](https://doc.rust-lang.org/std/borrow/trait.Borrow.html "trait std::borrow::Borrow")\<Q>？
```rust
pub trait Borrow<Borrowed>where
    Borrowed: ?Sized,{
    // Required method
    fn borrow(&self) -> &Borrowed;
}
```
- A trait for borrowing data.
	- **借用数据的特性**，它**允许不同类型以统一方式访问底层数据，而不必拥有数据的所有权**。
		- 不同类型
			- 字符串可变类型和不可变类型String和&str
		- 以统一方式访问底层数据
			- String和&str底层数据内容都可以通过&str来表示
		- 借用数据本身
			- 不必拥有所有权
		- 该trait里的borrow方法含义就是，我可以获取一个借用，这个借用本质上是该对象底层数据内容一个视图
	- 这可以用在什么地方呢？
		- 可以使用数据视图而不不必传入具体对象时，都可以这样使用
		- 为什么？
			- 传入具体的对象就会产生管理开销，分配和释放的开销，而引用的视图是零成本抽象，无分配，无多有权概念。

- 在 Rust 中，通常会为**不同的用例提供不同的类型表示**，也就提供多种类型表示来表示数据的不同存储和管理策略
	- 例如：
		- 值的存储位置和管理可以通过指针类型
			- （例如 [`Box <T >`](https://doc.rust-lang.org/std/boxed/struct.Box.html) 或 [`Rc <T >）`](https://doc.rust-lang.org/std/rc/struct.Rc.html) 针对特定用途进行适当的具体选择。
		- **提供了可选的方面，这些方面提供了潜在的昂贵功能。**
			- 这种类型的一个例子是 [`String`](https://doc.rust-lang.org/std/string/struct.String.html)，它将扩展字符串的能力添加到基本的 [`str`](https://doc.rust-lang.org/std/primitive.str.html "primitive str").这需要保留简单、不可变的字符串所不需要的附加信息。
				-  [2. String 如何“扩展” str？](../../../2.2%20类型系统、数据布局/2.2.1%20类型基础/String%20和%20&str.md#2.%20String%20如何“扩展”%20str？)
			- 这两者是紧密关联的，`String` 里的字节数据最终就是 `str` 的字节内容，只是 `String` 额外管理内存和可变性。
- 这些类型[通过对基础数据类型的引用提供对该数据的访问](#通过对基础数据类型的引用提供对该数据的访问)。
	- 他们被称为“借来的”那种类型。例如，[`Box<T>`](https://doc.rust-lang.org/std/boxed/struct.Box.html) 可以借用为 `T`，而 [`String`](https://doc.rust-lang.org/std/string/struct.String.html) 可以借用为 `str`。
- 类型表示它们可以通过实现 `Borrow<T>`借用作为某些类型T，提供一个对T的引用在borrow方法中
	- 一种类型可以自由地借用为几种不同的类型。如果它希望可变地借用为类型，允许修改基础数据，它可以另外实现 [`BorrowMut<T>`](https://doc.rust-lang.org/std/borrow/trait.BorrowMut.html "trait std::borrow::BorrowMut")。
- 此外，当为额外的 trait 提供实现时，需要考虑它们是否应该作为底层类型的表示而与底层类型的实现行为相同。当泛型代码依赖于这些额外 trait 实现的相同行为时，它通常使用 `Borrow<T>`。这些特征可能会作为额外的特征边界出现。
- 特别地 `，**Eq`、`Ord` 和 `Hash` 对于借用值和拥有值必须是等价**的：`x.borrow（）== y.borrow（）` 应该给出与 `x == y` 相同的结果。
	- 在 Rust 中，`Borrow<T>` 的设计目的是：
		- 让泛型代码可以**用“借用者”去匹配“拥有者”。**
		- 核心要求：等价,而语言中等价的表示就是Eq`、`Ord` 和 `Hash
		- **对于 `&str` 这样的裸借用类型（primitive reference type），调用 `borrow()` 就是返回自己**。
		- 例子
			- String底层的hash就是使用&str来（借用视图）完成的。
- 如果泛型代码只需要为所有可以提供对相关类型 `T` 的引用的类型工作， [`AsRef<T>`](https://doc.rust-lang.org/std/convert/trait.AsRef.html "trait std::convert::AsRef")，因为更多类型可以安全地实现它。
## Examples  示例

- 作为一个数据集合，[`HashMap<K，V>`](https://doc.rust-lang.org/std/collections/struct.HashMap.html) 拥有键和值。如果键的实际数据被包装在某种管理类型中，那么仍然可以使用对键数据的引用来搜索值。例如，如果键是一个字符串，那么它很可能与哈希映射一起存储为 [`String`](https://doc.rust-lang.org/std/string/struct.String.html)，而使用 [`&str`](https://doc.rust-lang.org/std/primitive.str.html "primitive str") 搜索应该是可能的。因此，`insert` 需要对 `String 进行`操作，而 `get` 需要能够使用 `&str`。
- 稍微简化一下，`HashMap<K，V>` 的相关部分看起来像这样：
```rust
use std::borrow::Borrow;
use std::hash::Hash;

pub struct HashMap<K, V> {
    // fields omitted
}

impl<K, V> HashMap<K, V> {
    pub fn insert(&self, key: K, value: V) -> Option<V>
    where K: Hash + Eq
    {
        // ...
    }

    pub fn get<Q>(&self, k: &Q) -> Option<&V>
    where
		// `K: Borrow<Q>` 表示**类型 `K` 可以借用为类型 `Q`**
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized
    {
        // ...
    }
}
```
- 整个散列映射在键类型 `K` 上是通用的。由于这些键与哈希映射一起存储，因此此类型必须拥有键的数据。当插入一个键值对时，映射被赋予这样一个 `K`，需要找到正确的哈希桶，并根据该 `K` 检查键是否已经存在。因此，它需要 `K：Hash + Eq`。
- 然而，当在 map 中搜索值时，必须提供对 `K` 的引用作为要搜索的键，这将需要始终创建这样一个拥有的值。对于字符串键，这意味着一个 `String` 值只需要创建用于搜索只有 `String` 可用。
- 相反，`get` 方法对底层键数据的类型是泛型的，在上面的方法签名中称为 `Q`。他说，`K` 通过要求 `K：Borrow<Q>` 作为 `Q` 借用。通过额外要求 `Q：Hash + Eq`，它表示 `K` 和 `Q` 具有产生相同结果的 `Hash` 和 `Eq`trait 的实现。
- `get` 的实现特别依赖于 `Hash` 的相同实现，通过调用 `Hash：：hash` 对 `Q` 值进行哈希，即使它根据从 `K` 值计算的哈希值插入键。
- 因此，如果 `K` 包裹 `Q` 值产生与 `Q` 不同的散列，则散列映射中断。例如，假设你有一个类型，它包装了一个字符串，但比较 ASCII 字母时忽略了它们的大小写：
```rust
pub struct CaseInsensitiveString(String);

impl PartialEq for CaseInsensitiveString {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq_ignore_ascii_case(&other.0)
    }
}

impl Eq for CaseInsensitiveString { }
```
- 因为两个相等的值需要产生相同的哈希值， `所以 Hash` 的实现也需要忽略 ASCII 大小写：
```rust
impl Hash for CaseInsensitiveString {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for c in self.0.as_bytes() {
            c.to_ascii_lowercase().hash(state)
        }
    }
}
```
- `CaseInsensiveString` 能否实现 `Borrow<str>`？它当然可以通过其包含的所有字符串提供对字符串切片的引用。但是因为它`的 Hash` 实现不同，它的行为与 `str` 不同，因此实际上不必实现 `Borrow<str>`。如果它想允许其他人访问底层 `str`，它可以通过 AsRef\<str\> 来实现，AsRef\<str\>没有任何额外的要求。
# 附录
## 通过对基础数据类型的引用提供对该数据的访问
## 核心解析：借用类型 vs 拥有类型

### 1. “这些类型通过对基础数据类型的引用提供访问”是什么意思？

- Rust 中很多复杂类型，底层都包含某种数据，通常被称为“基础数据类型”（`T`）。
    
- 这些复杂类型（比如 `Box<T>`、`String`）提供了对这些基础数据的访问方式，通常是通过对基础数据的 **引用**（`&T` 或 `&mut T`）。
    
- 这种引用允许用户访问或操作基础数据，但不会转移其所有权。
    

### 2. “他们被称为‘借来的’那种类型”是什么意思？

- 这里“借来”是借用（borrowing）的意思，即：
    
    - 拥有类型（如 `Box<T>`、`String`）拥有数据的所有权。
        
    - 它们可以 **借用**（通过实现 `Borrow` trait）数据的引用（`&T` 或 `&str`），供外部使用，但所有权依然保留在原类型中。
        
- 这保证了数据安全和所有权的明确管理，是 Rust 的核心安全机制。
    

### 3. 示例说明

- `Box<T>` 拥有 `T` 的所有权，但可以借用出 `&T` 给调用者使用。
    
- `String` 拥有 UTF-8 字符串数据的所有权，但可以借用出 `&str` （字符串切片）供读取。
    
- 这样，借用类型只是对底层数据的“视图”，不拥有数据本身。