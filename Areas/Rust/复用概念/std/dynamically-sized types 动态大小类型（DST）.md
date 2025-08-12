
## [动态调整大小的类型（DST）](https://doc.rust-lang.org/nomicon/exotic-sizes.html#dynamically-sized-types-dsts)
- Rust 支持动态大小类型（DST）：没有静态已知大小或对齐的类型。从表面上看，这有点荒谬：Rust _必须_ 知道的大小和对齐的东西，以便正确地与它的工作！在 在这方面，DST 不是正常类型。因为它们缺少一个已知的静态 size，这些类型只能存在于指针后面。任何指向 因此，DST 变成了一个_宽_指针，由指针和“完成”它们的信息组成（下面有更多的介绍）。
- 该语言暴露了两个主要的 DST：
	- trait objects: `dyn MyTrait`  
	- slices：[`[T]`](https://doc.rust-lang.org/std/primitive.slice.html)、[`str`](https://doc.rust-lang.org/std/primitive.str.html) 和其他
- trait 对象表示实现它指定的 trait 的某个类型。为了支持运行时反射，使用包含使用该类型所需的所有信息的 vtable _擦除_确切的原始类型。完成 trait 对象指针的信息是 vtable 指针。指针对象的运行时大小可以从 vtable 动态请求。
- 切片只是一些连续存储的视图
	- 通常是一个数组或Vec完成一个切片指针的信息就是它所指向的元素的数量。指针对象的运行时大小就是元素的静态已知大小乘以元素的数量。
- 结构体实际上可以直接存储一个 DST 作为它们的最后一个字段，但这也使它们成为一个 DST：
	- 不幸的是，这样的类型在没有构造方法的情况下基本上是无用的。目前唯一正确支持的创建自定义 DST 的方法是通过使您的类型成为泛型并执行 _unsizing 强制_ ：


# 附录
- 官方：https://doc.rust-lang.org/nomicon/exotic-sizes.html#dynamically-sized-types-dsts