- 官方文档：https://doc.rust-lang.org/core/ptr/trait.Pointee.html
```rust
pub trait Pointee {

    /// 指向`Self`的指针和引用中的元数据类型。

    #[lang = "metadata_type"]

    // 注意：保持`static_assert_expected_bounds_for_metadata`中的特征边界

    // 在`library/core/src/ptr/metadata.rs`中

    // 与此处同步：

    // 注意：`dyn Trait + 'a`的元数据是`DynMetadata<dyn Trait + 'a>`

    // 所以不能添加`'static`边界。

    type Metadata: fmt::Debug + Copy + Send + Sync + Ord + Hash + Unpin + Freeze;

}
```

Provides the pointer metadata type of any pointed-to type.  
提供任何指向类型的**指针元数据类型**。
- 指针的元数据是 **指针内部携带的额外信息**，用于描述指针所指向数据的动态特征。
	- 对于 **普通（固定大小）类型**，元数据是单位类型 `()`，表示**无附加元数据**。
	- 对于 **动态大小类型（DST）**，比如 `str`、切片 `[T]`、`dyn Trait`，元数据会有实际信息：
## 指针元数据
Rust 中的原始指针类型和引用类型可以被认为是由两部分组成的：一个包含值的内存地址的数据指针，以及一些元数据。
- 瘦指针：对于静态大小的类型（实现 `Sized`traits）以及`外部`类型，指针被称为“瘦”(thin)：元数据大小为零，其类型为 `（）`
```rust
#[unstable(feature = "ptr_metadata", issue = "81513")]

// 注意：在语言中的特征别名稳定之前不要稳定这个？

pub trait Thin = Pointee<Metadata = ()>;
```
- 宽或胖指针：指向[动态大小类型的](https://doc.rust-lang.org/nomicon/exotic-sizes.html#dynamically-sized-types-dsts)指针被称为“宽”或“胖”，它们具有非零大小的元数据：
	- 对于最后一个字段是 DST 的结构，元数据是最后一个字段的元数据
	- 对于 `str` 类型，metadata 是以字节为单位的长度，作为 `usize`
	- 对于像 `[T]这样的`切片类型，元数据是以项目为单位的长度，如 `usize`
	- 对于像 `dyn SomeTrait 这样的` trait 对象，元数据是 [`DynMetadata<Self>`](https://doc.rust-lang.org/core/ptr/struct.DynMetadata.html "struct core::ptr::DynMetadata") (e.g.`DynMetadata<dyn SomeTrait>`）

## `Pointee`特征


这个特征的重点是其`Metadata`关联类型，
如上所述，它是`()`或`usize`或`DynMetadata<_>`。
它自动为每个类型实现。
即使在泛型上下文中没有相应的约束，也可以假定它已实现。

## 用法
- 原始指针可以使用其[`to_raw_parts`]方法分解为数据指针和元数据组件。
- 或者，可以使用[`metadata`]函数单独提取元数据
- 可以将引用传递给[`metadata`]并隐式强制转换。
- 可以使用[`from_raw_parts`]或[`from_raw_parts_mut`]从其数据指针和元数据重新组合（可能是宽）原始指针。

## 源码

### 提取指针的元数据
```rust
/// 提取指针的元数据组件。

///

/// 类型为`*mut T`、`&T`或`&mut T`的值可以直接传递给此函数，

/// 因为它们隐式强制转换为`*const T`。

///

/// # 示例

///

/// ```

/// #![feature(ptr_metadata)]

///

/// assert_eq!(std::ptr::metadata("foo"), 3_usize);

/// ```

#[inline]

pub const fn metadata<T: ?Sized>(ptr: *const T) -> <T as Pointee>::Metadata {

    ptr_metadata(ptr)

}
```
### 从数据指针和元数据形成（可能是宽）原始指针
```rust
/// 从数据指针和元数据形成（可能是宽）原始指针。

///

/// 此函数是安全的，但返回的指针不一定可以安全地解引用。

/// 对于切片，请参阅[`slice::from_raw_parts`]的安全要求文档。

/// 对于trait对象，元数据必须来自指向相同底层擦除类型的指针。

///

/// [`slice::from_raw_parts`]: crate::slice::from_raw_parts

#[unstable(feature = "ptr_metadata", issue = "81513")]

#[inline]

pub const fn from_raw_parts<T: ?Sized>(

    data_pointer: *const impl Thin,

    metadata: <T as Pointee>::Metadata,

) -> *const T {

    aggregate_raw_ptr(data_pointer, metadata)

}
```
### 执行与[`from_raw_parts`]相同的功能，但返回原始`*mut`指针
```rust
/// 执行与[`from_raw_parts`]相同的功能，但返回原始`*mut`指针，

/// 而不是原始`*const`指针。

///

/// 有关更多详细信息，请参阅[`from_raw_parts`]的文档。

#[unstable(feature = "ptr_metadata", issue = "81513")]

#[inline]

pub const fn from_raw_parts_mut<T: ?Sized>(

    data_pointer: *mut impl Thin,

    metadata: <T as Pointee>::Metadata,

) -> *mut T {

    aggregate_raw_ptr(data_pointer, metadata)

}
```
### `Dyn = dyn SomeTrait` trait对象类型的元数据
```rust
/// `Dyn = dyn SomeTrait` trait对象类型的元数据。

///

/// 它是指向vtable（虚拟调用表）的指针，

/// 表示操作存储在trait对象内的具体类型所需的所有信息。

/// vtable特别包含：

///

/// * 类型大小

/// * 类型对齐

/// * 指向类型的`drop_in_place`实现的指针（对于普通旧数据可能是无操作）

/// * 指向类型实现trait的所有方法的指针

///

/// 注意，前三个是特殊的，因为它们对于分配、删除和释放任何trait对象都是必需的。

///

/// 可以用不是`dyn` trait对象的类型参数命名此结构

/// （例如`DynMetadata<u64>`），但无法获得该结构的有意义的值。

///

/// 注意，虽然此类型实现了`PartialEq`，但比较vtable指针是不可靠的：

/// 指向同一trait的同一类型的vtable的指针可能比较不相等（因为vtable在多个代码生成单元中重复），

/// 而指向*不同*类型/trait的vtable的指针可能比较相等（因为相同的vtable可以在代码生成单元内去重）。

#[lang = "dyn_metadata"]

pub struct DynMetadata<Dyn: ?Sized> {

    _vtable_ptr: NonNull<VTable>,

    _phantom: crate::marker::PhantomData<Dyn>,

}
```
DynMetadata是Dyn = dyn SomeTrait` trait对象类型的元数据
- 它是指向vtable（虚拟调用表）的指针
	- 操作存储在trait对象内的具体类型所需的所有信息
	- [vtable](#vtable) 是一个结构体（但在 Rust 中是 opaque，不公开暴露）
# 附录
### vtable 
```rust
    /// 用于访问vtable的不透明类型。

    ///

    /// `DynMetadata::size_of`等的私有实现细节。

    /// 从概念上讲，这个指针后面实际上没有任何抽象机器内存。

    type VTable;

}
```

在 Rust 里，vtable（虚函数表）是：  
**一个由编译器生成的只读内存块**，  
**存储 trait 对象的动态调度信息**，  
**运行时指向具体实现（即具体类型的函数实现）**。
这背后的经典 C++/Rust 编译器思路是：
```lua
dyn Trait:
+-----------------+        +------------------------+
| 数据指针 *const | ----> | 具体对象数据           |
+-----------------+        +------------------------+
| vtable 指针 *const| ---> | [vtable 结构体（只读）]|
+-----------------+        +------------------------+

```
Rust 里，vtable ≈ C 里的 struct，但它是编译器私有的、opaque（不可见、不可直接定义）。
根据 Rust 官方文档与 rustc 源码分析：
```rust
vtable layout:
| slot 0: fn drop_in_place::<T>(*mut T)
| slot 1: usize size_of::<T>
| slot 2: usize align_of::<T>
| slot 3...: trait method 1
| slot 4...: trait method 2
| ...
```
简单说：

- 前三项是内存管理（析构、大小、对齐）。
	- **分配、删除和释放任何trait对象都是必需的。**
- 后续项是 trait 的方法指针。

例如：
```rust
trait MyTrait {
    fn foo(&self);
    fn bar(&self);
}
```
对应的vtable可能是：
```rust
[
    drop_in_place::<ConcreteType>,
    size_of::<ConcreteType>,
    align_of::<ConcreteType>,
    ConcreteType::foo (as fn ptr),
    ConcreteType::bar (as fn ptr)
]
```