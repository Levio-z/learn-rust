---
tags:
  - fleeting
---
## 1. 核心观点  
### Ⅰ. 概念层

在 Rust 中，`Pin` 是一个**[编译原理-指针封装类型](../../../../../../../basic/编译原理/指针封装类型/编译原理-指针封装类型.md)**，它的核心作用是保证所指向的数据在内存中**不会被移动**。

- [Pinning如何实现固定语义](#Pinning如何实现固定语义)
- [pin！](#pin！)
- [Rust-move-内存模型-三种语义-move](../../2.1%20所有权、生命周期和内存系统/2.1.1%20内存安全与防止数据竞争/Rust内存模型/move/Rust-move-内存模型-三种语义-move.md)
### Ⅱ. 应用层




### Ⅲ. 实现层

### **IV**.原理层

这个功能**并不是由语言本身或者编译器直接强制实现的**，而是通过类型系统和 API 约束实现的

## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### Pinning设计的动机

值得注意的是，pinning 是一个专为 Rust 异步实现而设计的底层构建模块。虽然它并非直接与 Rust 异步绑定，也可以用于其他用途，但它并非设计为通用机制，尤其不是处理自引用字段的开箱即用解决方案。将 pinning 用于异步代码以外的任何[情况](https://rust-lang.github.io/async-book/part-reference/pinning.html#fr-design-1) ，通常只有在将其封装在厚厚的抽象层中才能奏效，因为它需要编写大量繁琐且难以理解的不安全代码。

### 自引用结构体

如上所述，状态机转换会把每个暂停点的局部变量存储在结构体中。对于 `example` 函数这样的小例子，这很直接且不会导致任何问题。但当变量相互引用时，情况就变得复杂了。例如，考虑以下函数：

```rust
async fn pin_example() -> i32 {
    let array = [1, 2, 3];
    let element = &array[2];
    async_write_file("foo.txt", element.to_string()).await;
    *element
}
```

该函数创建了一个包含元素 `1`, `2`, 和 `3` 的小型 `array`。然后它创建对最后一个数组元素的引用，并将其存储在 `element` 变量中。接着，它异步地将数字转换为字符串并写入 `foo.txt` 文件。最后，它返回由 `element` 引用的数字。

由于该函数使用了单个 `await` 操作，生成的状态机包含三个状态：开始、结束和“等待写入“。该函数不接受参数，因此开始状态的结构体为空。如前所述，结束状态的结构体为空，因为函数在此处已执行完毕。“等待写入“状态的结构体则更为有趣：
```rust
struct WaitingOnWriteState {
    array: [1, 2, 3],
    element: 0x1001c, // 最后一个数组元素的地址
}
```

我们需要同时保存 `array` 数组和 `element` 变量，因为 `element` 对于返回值是必需的，而 `array` 被 `element` 引用。由于 `element` 是一个引用，它存储了一个 _指针_ （即内存地址）指向被引用的元素。这里我们以 `0x1001c` 为例。实际上，它就是 `array` 字段最后一个元素的地址，因此这取决于结构体在内存中的位置。具有这种内部指针的结构体被称为 _自引用结构体_ （_self-referential_ ），因为它们通过其中某个字段引用了自身。
#### 自引用结构体的问题

我们自引用结构体的内部指针引出了一个根本性问题，当我们查看其内存布局时，这一点变得显而易见：
![](asserts/Pasted%20image%2020251216162809.png)

`array` 字段起始于地址 0x10014，`element` 元素字段位于地址 0x10020。它指向地址 0x1001c，因为最后一个数组元素位于此地址。此时一切正常。然而，当我们把这个结构体移动到不同的内存地址时就会出现问题：
![](asserts/Pasted%20image%2020251216162827.png)


我们将结构体稍微移动了一下，现在它从地址 `0x10024` 开始。这种情况可能发生在，例如当我们把**结构体作为函数参数传递或将其赋值给不同的栈变量时**。问题在于，即使最后一个 `array` 元素已经移动，`element` 字段仍然指向地址 `0x1001c` ，然而实际上该元素现在位于地址 `0x1002c`。因此这个指针变成悬垂指针，导致在下一次 `poll` 调用时出现未定义行为。

#### 可能的解决方案

解决悬垂指针问题有三种基本方法

- **移动时更新指针：**其理念是每次结构体在内存中移动时都更新内部指针，从而保持有效。遗憾的是，这种方法需要对 Rust 进行大量修改，这可能导致巨大的性能损失。原因是需要某种运行时机制来跟踪所有结构体的字段类型并在每次移动操作时检查是否需要更新指针。
    
- **存储偏移量而非自引用：**为避免更新指针，编译器可以尝试将自引用存储为相对于结构体起始位置的偏移量。例如，上述 `WaitingOnWriteState` 结构体中的 `element` 字段可以存储为一个值为 8 的 `element_offset` 字段，因为引用点指向的数组元素在结构体起始位置之后 8 字节处。由于偏移量结构体被移动时保持不变，没有字段需要更新。这种方法的问题在于需要编译器检测所有自引用。这在编译时无法实现，因为引用的值可能取决于用户输入，因此就又需要一个运行时系统来分析引用并正确创建状态结构体。这不仅会导致运行时开销，还会影响某些编译器优化，从而再次造成较大的性能损失。
    
- **禁止移动结构体：**如上所述，只有在内存中移动结构体时才会出现悬垂指针。通过完全禁止对自引用结构体的移动操作就可以避免这个问题。这种方法的最大优势在于它能够在类型系统层面实现，无需额外的运行时开销。缺点是它将处理可能移动的自引用结构体的责任交给了程序员。

Rust 选择了第三种解决方案，这源于其提供 _零成本抽象_ 的原则，即抽象不应带来额外的运行时开销。_pinning_ API 正是为此目的而在 [RFC 2349](https://github.com/rust-lang/rfcs/blob/master/text/2349-pin.md) 中提出的。接下来，我们将简要概述这个API，并解释它如何与 async/await 和 futures 协同工作。
### 堆上的值

第一个观察结果是，[堆分配的](https://os.phil-opp.com/heap-allocation/) 值在大多数情况下已经拥有固定的内存地址。它们通过调用 `allocate` 来创建，由一个指针类型比如 `Box<T>` 来引用。虽然可以移动指针类型，但指针所指向的堆值在内存中的地址保持不变，除非调用 `deallocate` 将其释放。

使用堆分配，我们可以尝试创建一个自引用结构体：

```rust
fn main() {
    let mut heap_value = Box::new(SelfReferential {
        self_ptr: 0 as *const _,
    });
    let ptr = &*heap_value as *const SelfReferential;
    heap_value.self_ptr = ptr;
    println!("heap value at: {:p}", heap_value);
    println!("internal reference: {:p}", heap_value.self_ptr);
}

struct SelfReferential {
    self_ptr: *const Self,
}
```
我们创建了一个名为 `SelfReferential` 的简单结构体，它包含一个单独的指针字段。首先，我们使用空指针初始化此结构体，然后通过 `Box::new` 在堆上分配内存存储它。接下来尝试确定堆分配结构体的内存地址并将其存储在 `ptr` 变量中。最后，通过将 `ptr` 变量赋值给 `self_ptr` 字段使结构体形成自引用。

当我们在 playground 上执行这段代码时，可以看到堆值的地址与其内部指针是相等的，这意味着 `self_ptr` 字段是一个有效的自引用。由于 `heap_value` 变量仅是一个指针，移动它（例如传递给函数）并不会改变结构体自身的地址，因此即使指针被移动，`self_ptr` 仍保持有效。

然而，仍有一种方式可以破坏这个示例：我们可以从 `Box<T>` 将结构体移出或替换其内容:

```rust
let stack_value = mem::replace(&mut *heap_value, SelfReferential {
    self_ptr: 0 as *const _,
});
println!("value at: {:p}", &stack_value);
println!("internal reference: {:p}", stack_value.self_ptr);
```
([Try it on the playground](https://play.rust-lang.org/?version=stable&mode=debug&edition=2024&gist=e160ee8a64cba4cebc1c0473dcecb7c8))
这里我们使用 [`mem::replace`](https://doc.rust-lang.org/nightly/core/mem/fn.replace.html) 函数将堆分配的值替换为一个新的结构体实例。 **这样我们就可以将原始的 `heap_value` 移动到栈上，而结构体的 `self_ptr` 字段此时变成了一个悬垂指针**，仍然指向旧的堆地址。当您尝试在 playground 上运行示例时，会看到打印的 _“value at:”_ and _“internal reference:”_ 行确实显示了不同的指针。因此仅对值进行堆分配并不足以确保自引用安全。

导致上述破坏的根本问题是 `Box<T>` 允许我们获取堆分配值的 `&mut T` 引用。这个 `&mut T` 引用导致可以使用诸如 [`mem::replace`](https://doc.rust-lang.org/nightly/core/mem/fn.replace.html) 或者 [`mem::swap`](https://doc.rust-lang.org/nightly/core/mem/fn.swap.html) 这样的方法使堆分配的值失效。**为解决此问题，我们必须防止创建指向自引用结构体的 `&mut` 引用。**


```
heap value at: 0x55e3b875dd00
internal reference: 0x55e3b875dd00
value at: 0x7ffd7fccfd30
internal reference: 0x55e3b875dd00
```
替换之后，整个结构体被移动到栈上
### 不使用unpin
举个例子，让我们更新上面的 `SelfReferential` 类型来让其不实现 `Unpin`：
```rust
use core::marker::PhantomPinned;

struct SelfReferential {
    self_ptr: *const Self,
    _pin: PhantomPinned,
}
```

我们通过添加第二个类型为 [`PhantomPinned`](https://doc.rust-lang.org/nightly/core/marker/struct.PhantomPinned.html) 的 `_pin` 字段来选择退出。该类型是零大小的标记类型，仅用于不实现 `Unpin` trait。根据 [_auto trait_](https://doc.rust-lang.org/reference/special-types-and-traits.html#auto-traits) 的工作原理，当某个字段不是 `Unpin` 时，就足以使整个结构体不实现 `Unpin` trait。

第二步是将示例中的 `Box<SelfReferential>` 类型更改为 `Pin<Box<SelfReferential>>` 类型。最简单的方法是使用 [`Box::pin`](https://doc.rust-lang.org/nightly/alloc/boxed/struct.Box.html#method.pin) 函数而非 [`Box::new`](https://doc.rust-lang.org/nightly/alloc/boxed/struct.Box.html#method.new) 来创建堆分配的值：

```rust
let mut heap_value = Box::pin(SelfReferential {
    self_ptr: 0 as *const _,
    _pin: PhantomPinned,
});
```

除了将 `Box::new` 改为 `Box::pin` 外，我们还需要在结构体初始化器中添加新的 `_pin` 字段。由于 `PhantomPinned` 是零大小类型，我们只要有其类型名称即可完成初始化。

当我们现在[尝试运行调整后的示例](https://play.rust-lang.org/?version=stable&mode=debug&edition=2024&gist=961b0db194bbe851ff4d0ed08d3bd98a)时，会发现它会报错：

这两个错误的发生是因为 `Pin<Box<SelfReferential>>` 类型不再实现 `DerefMut` trait。这正是我们想要的，因为 `DerefMut` trait 会返回一个 `&mut` 引用，而这正是我们想要避免的。这种情况之所以发生，仅仅是因为我们同时选择了不实现 `Unpin` 并将 `Box::new` 改为 `Box::pin`。

现在的问题是，编译器不仅阻止了第16行中的类型移动，还禁止在第10行初始化 `self_ptr` 字段。这是因为编译器无法区分 `&mut` 引用的有效和无效使用。要使初始化正常工作，我们必须使用不安全的 [`get_unchecked_mut`](https://doc.rust-lang.org/nightly/core/pin/struct.Pin.html#method.get_unchecked_mut) 方法：

```rust
// 安全，因为修改一个字段不会移动整个结构体
unsafe {
    let mut_ref = Pin::as_mut(&mut heap_value);
    Pin::get_unchecked_mut(mut_ref).self_ptr = ptr;
}
```

`get_unchecked_mut` 函数工作于 `Pin<&mut T>` 之上，而非 `Pin<Box<T>>` ，因此我们必须使用 [`Pin::as_mut`](https://doc.rust-lang.org/nightly/core/pin/struct.Pin.html#method.as_mut) 转换值。然后我们可以通过 `get_unchecked_mut` 返回的 `&mut` 引用来设置 `self_ptr` 字段。

现在剩下的唯一错误就是 `mem::replace` 上的预期错误了。记住，这个操作试图将堆分配的值移动到栈上，这会破坏存储在 `self_ptr` 字段的自引用。通过选择不实现 `Unpin` 并采用 `Pin<Box<T>>` ，我们可以在编译器阻止此类操作并安全地处理自引用结构体。正如我们所看到的，编译器（目前）还无法证明创建自引用是安全的，因此我们需要使用 unsafe 代码块自行验证其正确性。

### 栈上的Pinning与 `Pin<&mut T>`

在上一节中，我们学习了如何使用 `Pin<Box<T>>` 安全地创建堆分配的自引用值。虽然这种方法效果良好且相对安全（除了不安全的构造过程外），但所需的堆分配会带来性能开销。由于 Rust 致力于尽可能实现零成本抽象，pinning API 也允许创建指向栈上值的 `Pin<&mut T>` 实例。

与拥有被包装值的所有权的 `Pin<Box<T>>` 实例不同， `Pin<&mut T>` 实例仅临时借用所包装的值。这使得情况更加复杂，因为它要求程序员自行提供额外的保证。最重要的是，一个 `Pin<&mut T>` 必须在被引用的 `T` 的整个生命周期内保持固定，这一点对于基于栈的变量来说难以验证。为此，存在像 [`pin-utils`](https://docs.rs/pin-utils/0.1.0-alpha.4/pin_utils/) 这样的 crate，但我仍然不建议固定到栈上，除非你非常清楚自己在做什么。

如需进一步阅读，请查阅 [`pin` 模块](https://doc.rust-lang.org/nightly/core/pin/index.html) 的文档以及 [`Pin::new_unchecked`](https://doc.rust-lang.org/nightly/core/pin/struct.Pin.html#method.new_unchecked) 方法。
### Pinning如何实现固定语义

`Pin<P>` 是对一个指针 `P`（如 `Box<T>` 或 `&mut T`）的封装，它本身提供以下机制：
1. **类型约束**
    - **`Pin<P>` 会阻止通过可变引用 (`&mut`) 修改内部对象的地址。**
    - 只能通过安全或不安全 API 获取内部数据，但**不能直接移动**。
2. **不安全承诺**
    - 编译器无法验证 Pin 保证的安全性，只能通过类型系统进行静态约束。
    - **开发者需要自己保证，Pin 内的数据在生命周期内不会被移动。**
3. **典型用法**
    - 结合异步 Future（`async` / `await`）使用，保证状态机在堆上固定，以便安全生成 self-referential 的类型。
    - 结合 `!Unpin` 标记的类型，实现“堆上固定、不允许移动”的数据结构。

### 固定（pinning）

- 使得结构体中不同字段之间的引用（有时称为自引用）是安全的
- **这对于异步函数的实现至关重要**（异步函数以数据结构的形式实现，其中变量存储在字段中。由于变量之间可以相互引用，因此**实现异步函数的 Future 对象中的字段必须能够相互引用**）。通常情况下，程序员无需关注这一细节，但当直接处理 Future 对象时，则需要注意，因为 `Future::poll` 的签名要求 `self` 被固定。

### pin！

- 如果你通过引用使用 Future，可能需要使用 `pin!(...)` 来固定引用，以确保该引用仍然实现了` Future` 特性（这通常会在使用 `select` 宏时出现）。
- 同样地，如果你想手动调用 `poll` 来访问一个 Future（通常是因为你正在实现另一个 Future），你需要一个固定的引用（使用 `pin!` 或确保参数的类型已固定）。
- 如果你正在实现一个 Future，或者由于其他原因你拥有一个固定的引用，并且想要可变地访问对象的内部结构，你需要了解下面关于固定字段的部分，才能知道如何操作以及何时可以安全地进行操作。

### 固定的概念
如果一个对象不会被移动或以其他方式失效，则该对象会被固定。
- **是否可以移动是编译器根据上下文推断的结果，而不是类型本身的属性**。ust 的类型系统并没有明确定义一个对象是否可以移动，但编译器知道（这就是为什么你会收到“无法移出”的错误信息）。
- 固定的生命周期：与借用（以及借用导致的移动限制）不同，固定是永久性的。一个对象可以从非固定状态变为固定状态，但一旦被固定，它就必须保持固定状态，直到被释放[为止](https://rust-lang.github.io/async-book/part-reference/pinning.html#footnote-inherent) 。

- 在 Rust 中，指针类型本身并不承载“固定性”（pinned）的语义，指针可以自由复制、移动或存储。固定性描述的是**被指向的数据在内存中的位置是否可以移动**，而不是指针自身的移动能力。

- 粗略地说， `Pin<Box<T>>` 是指向拥有的、已固定对象的指针，而 `Pin<&mut T>` 是指向唯一借用的、可变的、已固定对象的指针（参见 `&mut T` ，它是指向唯一借用的、可变的对象的指针，该对象可能已固定，也可能未固定）。

- Rust 直到 1.0 版本之后才引入了 pinning 的概念。出于向后兼容性的考虑，Rust 无法显式地表达一个_对象_是否被 pinning。我们只能表达一个**引用指向的是一个 pinning 对象还是一个非 pinning 对象**。

### 正交性
|可变性|固定性|Rust 类型示例|含义|
|---|---|---|---|
|可变|固定|`Pin<&mut T>`|可以修改对象，但不能移动它|
|可变|非固定|`&mut T`|可以修改对象，也可以移动它|
|不可变|固定|`Pin<&T>`|不可修改对象，且不可移动|
|不可变|非固定|`&T`|不可修改对象，但可以移动它（在未借用时）|
**键点**：

1. `&T` 是不可变的，但它本身没有固定性限制：借用期间它阻止对象被移出，但这只是暂时的，借用结束后对象仍然可以移动。
2. `Pin<&mut T>` 提供了**永久性的移动限制**，而不仅仅是借用期间的临时限制。
3. 正交意味着**你可以同时讨论“是否可修改”和“是否可移动”**，两者互不干扰，但结合起来才决定对象的完整行为。

### Unpin
- 定义:
    - 标记trait，表示类型不需要固定内存位置的保证
    - 大多数类型自动实现Unpin(编译器保守策略)
- 与Pin的关系:
	- 如果T: Unpin，Pin`<Ptr>`对T没有额外限制
    - 非Unpin类型必须通过Pin保证内存稳定性
- 特殊处理:
    - 自引用结构需要显式标记为!Unpin(通过PhantomPinned)
    - Future通常需要Pin保护，因为可能包含自引用状态
        - 实现future的数据结构都需要使用

虽然“移动”和“不移动”是我们引入“固定”这个概念的方式，而且从名称上也略有暗示， `Pin` 实际上并不能告诉你被固定者是否真的会移动。

固定（Pinning）实际上是一种关于有效性的契约，而非关于移动的契约。它保证_如果一个对象是地址敏感的，那么_它的地址就不会改变（因此，从它派生的地址，例如其字段的地址，也不会改变）。Rust 中的大多数数据都不是地址敏感的。它们可以被移动，一切都不会有问题。 **`Pin` 保证了指向对象相对于其地址的有效性。如果指向对象是地址敏感的，那么它就不能被移动；如果它不是地址敏感的，那么移动与否都无关紧要。**

`Unpin` 是一个特性，用于表示对象是否对地址敏感。**如果一个对象实现了 `Unpin` ，则它对地址_不_敏感**。如果一个对象 `!Unpin` ，则它对地址敏感。或者，如果我们把 pinning 理解为将对象固定在某个位置，那么 `Unpin` 意味着可以安全地撤销该操作，并允许移动对象。或者，如果我们把 pinning 理解为将对象固定在某个位置，那么 `Unpin` 意味着可以安全地撤销该操作，并允许移动对象。

`Unpin` 是一个自动特性，大多数类型都是 `Unpin` 。只有具有 `!Unpin` 字段或明确选择退出的类型才不是 `Unpin` 。您可以通过拥有 [`PhantomPinned`](https://doc.rust-lang.org/std/marker/struct.PhantomPinned.html) 字段来选择退出，或者（如果您使用的是 nightly 版本）使用 `impl !Unpin for ... {}` 。

对于实现了 `Unpin` 类型， `Pin` 方法实际上什么也不做。Pin `Pin<Box<T>>` 和 `Pin<&mut T>` 的使用方式与 `Box<T>` 和 `&mut T` 完全相同。事实上，对于 `Unpin` 类型，可以使用 `Pin::new` 和 `Pin::into_inner` 自由地在 `Pin` 指针和普通指针之间进行转换。**值得重申的是： `Pin<...>` 并不保证被指向的指针不会移动，它只保证当被指向的指针是 `!Unpin` 类型时不会移动。**

上述内容的实际意义在于，使用 `Unpin` 类型和绑定比使用`!Unpin` 类型要容易得多。事实上， `Pin` 标记对 `Unpin` 类型和指向 `Unpin` 类型的指针基本上没有影响，您基本上可以忽略所有绑定保证和要求。

**`Unpin` 不应被理解为对象自身的属性； `Unpin` 唯一改变的是对象与 `Pin` 的交互方式**。在 pinning 上下文之外使用 `Unpin` 绑定不会影响编译器的行为或对对象的操作。使用 `Unpin` 的唯一理由是与 pinning 结合使用，或者将绑定传播到与 pinning 一起使用的位置。

### [`Pin`](https://rust-lang.github.io/async-book/part-reference/pinning.html#pin)

[`Pin`](https://doc.rust-lang.org/std/pin/struct.Pin.html) **是一种标记类型，它对类型检查很重要，但会被编译掉，运行时不存在（ `Pin<Ptr>` 保证与 `Ptr` 具有相同的内存布局和 ABI）**。它是指针（例如 `Box` ）的包装器，因此它的行为类似于指针类型，但它不会增加间接寻址。在程序运行时， `Box<Foo>` 和 `Pin<Box<Foo>>` 是相同的。**最好将 `Pin` 视为指针的修饰符，而不是指针本身。**

`Pin<Ptr>` 表示 `Ptr` 的指向对象（而非 `Ptr` 本身）被固定。也就是说， `Pin` 保证指向对象（而非指针本身）在其地址上保持有效，直到被释放为止。如果指向对象是地址敏感的（即，` !Unpin` ），则指向对象不会被移动。
### [Pinning values  固定值](https://rust-lang.github.io/async-book/part-reference/pinning.html#pinning-values)

对象并非创建时就已固定。对象初始状态为未固定（可以自由移动），只有当创建指向该对象的固定指针时，它才会被固定。如果对象是 `Unpin` ，则可以使用 `Pin::new` 轻松完成此操作；但是，如果对象不是 `Unpin` ，则固定它必须确保它无法被移动或通过别名失效。

要将对象固定在堆上，可以使用 [`Box::pin`](https://doc.rust-lang.org/std/boxed/struct.Box.html#method.pin) 创建一个新的固定 `Box` ，或者使用 [`Box::into_pin`](https://doc.rust-lang.org/std/boxed/struct.Box.html#method.into_pin) 将现有的 `Box` 转换为固定 `Box` 。无论哪种方式，最终都会得到 `Pin<Box<T>>` 。一些其他指针（例如 `Arc` 和 `Rc` ）也有类似的机制。对于没有类似机制的指针，或者对于自定义的指针类型，需要使用 [`Pin::new_unchecked`](https://doc.rust-lang.org/std/pin/struct.Pin.html#method.new_unchecked) 创建一个[固定](https://rust-lang.github.io/async-book/part-reference/pinning.html#footnote-box-pin)指针。这是一个不安全函数，因此程序员必须确保 ` Pin` 的不变量得到维护。也就是说，被指向的对象在任何情况下都保持有效，直到其析构函数被调用。要确保这一点，有一些细节需要注意，请参阅函数[文档](https://doc.rust-lang.org/std/pin/struct.Pin.html#method.new_unchecked)或下文“ [固定的工作原理”](https://rust-lang.github.io/async-book/part-reference/pinning.html#how-pinning-works) 部分了解更多信息。

`Box::pin` 将对象固定到堆中的某个位置。要将对象固定到栈上，可以使用 [`pin`](https://doc.rust-lang.org/std/pin/macro.pin.html) 宏来创建并固定一个可变引用（ `Pin<&mut T>` ） [5](https://rust-lang.github.io/async-book/part-reference/pinning.html#footnote-not-stack) 。

Tokio 也提供了一个 [`pin`](https://docs.rs/tokio/latest/tokio/macro.pin.html) 宏，其功能与标准宏相同，并且支持在宏内部将值赋给变量。futures-rs 和 pin-utils crate 中包含一个 `pin_mut` 宏，该宏曾经很常用，但现在已被弃用，建议使用标准宏。

您还可以使用 `Pin::static_ref` 和 `Pin::static_mut` 来固定静态引用。

### [Using pinned types  使用固定类型](https://rust-lang.github.io/async-book/part-reference/pinning.html#using-pinned-types)

理论上，使用固定指针与使用其他指针类型并无二致。然而，由于它并非最直观的抽象概念，且缺乏语言支持，因此使用固定指针往往不太方便。最常见的固定情况是处理 future 和 stream 时，我们将在下文中详细介绍这些细节。

**由于 `Pin` 的 `Deref` 实现，将固定指针用作不可变借用的引用非常简单**。通常情况下，你可以直接将 `Poll<Ptr<T>>` 视为 `&T` ，必要时使用显式的 `deref()` 函数。同样，使用 `as_ref()` 获取 `Pin<&T>` 也非常容易。

处理固定类型的最常用方法是使用 `Pin<&mut T>` （例如，在 [`Future::poll`](https://doc.rust-lang.org/std/future/trait.Future.html#tymethod.poll) 中），但是，生成固定对象的最简单方法是使用 `Box::pin` ，它会返回一个 `Pin<Box<T>>` 。你可以使用 [`Pin::as_mut`](https://doc.rust-lang.org/std/pin/struct.Pin.html#method.as_mut) 将后者转换为前者。然而，由于语言不支持重用引用（隐式重借），你必须不断调用 `as_mut` 而不是重用其结果。例如（摘自 `as_mut` 文档）：

```
impl Type {
    fn method(self: Pin<&mut Self>) {
        // do something
    }

    fn call_method_twice(mut self: Pin<&mut Self>) {
        // `method` consumes `self`, so reborrow the `Pin<&mut Self>` via `as_mut`.
        self.as_mut().method();
        self.as_mut().method();
    }
}

```
方法签名 `fn method(self: Pin<&mut Self>)` 表明：
- 方法会 **消耗** `self`（按值接收 Pin<&mut Self>）。
- 因此你不能直接使用原来的 `self` 再调用一次 `method`，否则会违反所有权规则。
但是 `Pin<&mut T>` **不允许这种隐式重借**：
- 原因：Pin 的安全性依赖于其生命周期内对象不会移动。
- 如果允许隐式重借，可能导致多个 `Pin<&mut T>` 同时存在，而这破坏了 Pin 的移动保证。
- 所以你必须 **显式调用 `as_mut()`** 来创建新的可变固定引用，用于调用消耗 `self` 的方法。

如果您需要以其他方式访问已固定的指针，可以使用 [`Pin::into_inner_unchecked`](https://doc.rust-lang.org/std/pin/struct.Pin.html#method.into_inner_unchecked) 。但是，这样做并不安全，您必须_非常_小心地确保遵守 `Pin` 的安全要求。

### [How pinning works  固定是如何工作的](https://rust-lang.github.io/async-book/part-reference/pinning.html#how-pinning-works)

`Pin` 是一个简单的指针包装结构体（也称为 newtype）。它通过要求其泛型参数上的 `Deref` 绑定才能执行任何有意义的操作，从而强制其只能处理指针。然而，这仅仅是为了表达意图，而非为了保证安全性。与大多数 newtype 包装器一样， `Pin` 存在是为了在编译时表达一个不变式，而不是为了任何运行时效果。实际上，在大多数情况下， `Pin` 及其绑定机制会在编译期间完全消失。

准确地说， `Pin` 所表达的不变量是关于有效性的，而不仅仅是可移动性。它也是一种有效性不变量，仅在指针被固定后才生效——在此之前， `Pin` 不起作用，并且对固定之前发生的事情没有任何要求。**一旦指针被固定， `Pin` 要求（并在安全代码中保证）被固定的对象在内存中的同一地址保持有效，直到该对象的析构函数被调用**。

对于不可变指针（例如，借用的引用）， `Pin` 没有效果——因为被指向的对象不能被改变或替换，所以不存在失效的风险。

对于允许修改的指针（例如 `Box` 或 `&mut` ），直接访问该指针或访问指向它的可变引用（ `&mut` ）可能会导致被指向指针的修改或移动。Pin `Pin` 本身并不提供任何（非 `unsafe` ）方法来直接访问指针或可变引用。指针通常通过实现 [`DerefMut`](https://doc.rust-lang.org/std/ops/trait.DerefMut.html) 接口来提供指向其指向对象的可变引用， `Pin` 仅在指向对象被 `Unpin` 时才实现 `DerefMut` 。

这个实现极其简单！简而言之： `Pin` 是一个封装指针的结构体，它只提供对被指向对象的不可变访问（如果被指向对象是 `Unpin` ，则提供可变访问）。其他一切都是细节（以及用于不安全代码的微妙不变式）。为了方便起见， `Pin` 提供了一种在 `Pin` 类型之间转换的方法（始终安全，因为指针无法从 `Pin` 中逃逸），等等。

`Pin` 也提供了用于创建固定指针和访问底层数据的不安全函数。与所有 `unsafe` 函数一样，维护安全不变性是程序员而非编译器的责任。遗憾的是，固定指针的安全不变性比较分散，因为它们在不同的地方强制执行，难以用全局统一的方式描述。我不会在这里详细描述它们，请参考相关文档，但我会尝试进行总结（详细概述请参见[模块文档](https://doc.rust-lang.org/std/pin/index.html) ）：

创建一个新的固定指针  [`new_unchecked`](https://doc.rust-lang.org/std/pin/struct.Pin.html#method.new_unchecked) 。程序员必须确保被指向的指针是固定的（即，符合固定不变式）。此要求可能仅由指针类型满足（例如， `Box` ），也可能需要被指向指针类型的参与（例如， `&mut` ）。这包括（但不限于）：

- 在 `Deref` 和 `DerefMut` 中不移出 `self` 。
- 正确实现 `Drop` ，请参阅 [Drop 保证](https://doc.rust-lang.org/std/pin/index.html#subtle-details-and-the-drop-guarantee) 。
- 如果您需要锁定保证，可以选择退出 `Unpin` （通过使用 [`PhantomPinned`](https://doc.rust-lang.org/std/marker/struct.PhantomPinned.html) ）。
- 被指向的对象可能不是 `#[repr(packed)]` 。

访问已固定的值 [`into_inner_unchecked`](https://doc.rust-lang.org/std/pin/struct.Pin.html#method.into_inner_unchecked) 、 [`get_unchecked_mut`](https://doc.rust-lang.org/std/pin/struct.Pin.html#method.get_unchecked_mut) 、 [`map_unchecked`](https://doc.rust-lang.org/std/pin/struct.Pin.html#method.map_unchecked) 和 [`map_unchecked_mut`](https://doc.rust-lang.org/std/pin/struct.Pin.html#method.map_unchecked_mut) 。从访问数据的那一刻起，直到析构函数运行，程序员就有责任强制执行固定保证（包括不移动数据）（请注意，此责任范围不仅限于不安全调用，而且适用于底层数据发生的任何事情）。


没有提供其他方法将数据移出固定类型（这需要不安全的实现）。
#### [Pinning pointer types  指针类型](https://rust-lang.github.io/async-book/part-reference/pinning.html#pinning-pointer-types)
我们之前提到过， `Pin` 封装了指针类型。常见的有 `Pin<Box<T>>` 、 `Pin<&T>` 和 `Pin<&mut T>` 。从技术上讲，绑定指针类型的唯一要求是它实现了 `Deref` 。然而，除了使用不安全代码（通过 `new_unchecked` ）之外，没有其他方法可以为任何其他指针类型创建 `Pin<Ptr>` 。这样做对指针类型有要求，以确保满足绑定约定：
- 指针的 `Deref` 和 `DerefMut` 实现不能移出其指向的对象。
- 在 `Pin` 创建之后，即使在 `Pin` 被放置之后，也绝对不能获取指向对象的 `&mut` 引用（这就是为什么不能安全地从 `&mut T` 构造 `Pin<&mut T>` 原因）。这一规则必须通过多步操作或引用保持成立（这会阻止使用 `Rc` 或 `Arc` ）。
	- **原因**：如果能获取 &mut，程序就可以移动数据，从而破坏 Pin 的保证。
	- **额外说明**：这也禁止用 `Rc` 或 `Arc` 创建安全的 `Pin`，因为它们可以在某些条件下生成可变引用，从而破坏固定性。
- *指针的 `Drop` 实现不能移动（或以其他方式使其指向的对象失效）。*


### [固定和 `Drop`](https://rust-lang.github.io/async-book/part-reference/pinning.html#pinning-and-drop)

#### 绑定机制有效期

在 Rust 中，`Pin` 封装的指针保证所指向对象在其生命周期内不会被移动。绑定机制一直有效，直到**对象真正被释放为止**。严格来说，“释放”是指对象的 `drop` 方法**返回完成时**，而不是 `drop` 被调用的那一刻。这意味着：

- 对象的析构过程中（`drop` 正在执行），它依然处于“绑定”状态；
    
- 对象的内存被回收之前，绑定保证对象地址稳定，便于自引用结构安全使用。

如果您要实现一个地址敏感类型（即 ` !Unpin` 类型），则必须格外注意 `Drop` 实现。即使 `drop` 中的 `self` 类型是 `&mut Self` ，您也必须将其视为 `Pin<&mut Self>` 。换句话说，您必须确保对象在 `drop` 函数返回之前保持有效。在源代码中明确这一点的一种方法是遵循以下惯用法：
```
impl Drop for Type {
    fn drop(&mut self) {
        // `new_unchecked` is okay because we know this value is never used
        // again after being dropped.
        inner_drop(unsafe { Pin::new_unchecked(self)});

        fn inner_drop(this: Pin<&mut Self>) {
            // Actual drop code goes here.
        }
    }
}

```

请注意，有效性要求将取决于所实现的类型。建议精确定义这些要求，尤其是在对象销毁方面，如果可能涉及多个对象（例如，侵入式链表）。确保此处的正确性可能很有意思！

### [Pinned self in methods](https://rust-lang.github.io/async-book/part-reference/pinning.html#pinned-self-in-methods)

对固定类型调用方法时，需要考虑这些方法中的自身类型。如果方法不需要修改 `self` ，那么仍然可以使用 `&self` ，因为 `Pin<...>` 可以解引用借用的引用。但是，如果需要修改 `self` （并且你的类型不是 `Unpin` ），则需要在 `&mut self` 和 `self: Pin<&mut Self>` 之间进行选择（尽管固定指针不能隐式强制转换为后一种类型，但可以使用 `Pin::as_mut` 轻松转换）。

使用 `&mut self` 可以简化实现，但意味着该方法不能在已固定的对象上调用。使用 `self: Pin<&mut Self>` 则意味着要考虑固定投影（参见下一节），并且只能在已固定的对象上调用。虽然这听起来有点令人困惑，但当你记住固定是一个分阶段的概念时，这一切就很容易理解了——对象最初是未固定的，然后在某个阶段经历阶段变化成为已固定的。` &mut self` 方法可以在第一阶段（未固定）调用，而 `self: Pin<&mut Self>` 方法可以在第二阶段（已固定）调用。

请注意， `drop` 需要使用 `&mut self` （即使它可以在两个阶段中调用）。这是由于语言的限制以及为了向后兼容而做出的。它需要在编译器中进行特殊处理，并且具有一定的安全要求。

### [Pinned fields, structural pinning, and pin projection](https://rust-lang.github.io/async-book/part-reference/pinning.html#pinned-fields-structural-pinning-and-pin-projection)

如果一个对象被固定，这告诉我们关于其字段的“固定性”是什么？答案取决于数据类型实现者的选择，没有通用的答案（实际上，同一个对象的不同字段的固定性可能不同）。

如果对象的结构固定性传播到字段，我们称该场表现出“结构固定”，或者说该固定性随场一起投影。在这种情况下，应该存在一个投影方法 `fn get_field(self: Pin<&mut Self>) -> Pin<&mut Field>` 。如果场不是结构固定的，则投影方法的签名应为 `fn get_field(self: Pin<&mut Self>) -> &mut Field` 。实现这两种方法（或实现类似代码）都需要编写 `unsafe` 代码，并且两种选择都会对安全性产生影响。固定性传播必须保持一致，场要么始终是结构固定的，要么始终不是结构固定的；场在某些时候是结构固定的而在其他时候不是结构固定的，几乎总是不安全的。





## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
