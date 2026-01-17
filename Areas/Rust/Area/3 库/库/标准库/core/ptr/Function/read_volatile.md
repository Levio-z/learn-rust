从 `src` 执行值的易失读取，而不移动它。

易失性作旨在作用于 I/O 内存。因此，它们被视为外部可观察事件（就像系统调用一样，但不透明程度较低），并且保证编译器不会在其他外部可观察事件中被省略或重新排序。考虑到这一点，需要区分两种使用情况：

当易失性作用于[分配](https://doc.rust-lang.org/nightly/core/ptr/index.html#allocated-object "mod core::ptr")中的内存时，它的行为与 [`read`](https://doc.rust-lang.org/nightly/core/ptr/fn.read.html "fn core::ptr::read")，但不会被省略或重新排序的额外保证除外（见上文）。这意味着该作实际上将访问内存，而不是例如降低为重用先前读取的数据。除此之外，所有常用的内存访问规则都适用（包括出处）。

特别是，就像在 C 中一样，作是否易失与涉及来自多个线程的并发访问的问题没有任何关系。在这方面，易失性访问的行为与非原子访问完全相同。
>**易失性访问并不解决线程间的并发访问问题**。换句话说，**易失性访问的行为和非原子访问是相同的**，并不会对多个线程同时访问该变量进行保护。
>**与原子操作的区别**：原子操作（如 `AtomicUsize` 或 `AtomicBool`）会提供原子性保证，确保在多线程并发情况下的安全性。而易失性访问只是保证了内存访问的顺序，不会进行编译器优化，但没有提供任何线程同步或原子操作保证。

然而，易失性作也可用于访问任何 Rust 分配_之外_的内存。在此用例中，指针_不必_对读取[有效](https://doc.rust-lang.org/nightly/core/ptr/index.html#safety "mod core::ptr") 。这通常用于必须通过 I/O 内存映射访问的 CPU 和外设寄存器，最常见的是硬件保留的固定地址。这些通常具有与其作相关的特殊语义，不能用作通用内存。在这里，任何地址值都是可能的，包括 0 和 [`usize：：MAX`](https://doc.rust-lang.org/nightly/core/primitive.usize.html#associatedconstant.MAX "associated constant usize::MAX")，只要目标硬件明确定义了此类读取的语义。指针的出处无关紧要，可以用 [`without_provenance`](https://doc.rust-lang.org/nightly/core/ptr/fn.without_provenance.html "fn core::ptr::without_provenance") 创建。访问不得捕获。它可能会导致副作用，但这些副作用不得以任何方式影响 Rust 分配的内存。此访问仍然不被视为[原子访问](https://doc.rust-lang.org/nightly/core/sync/atomic/index.html#memory-model-for-atomic-accesses "mod core::sync::atomic") ，因此它不能用于线程间同步。