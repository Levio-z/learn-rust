---
tags:
  - fleeting
---
## 1. 核心观点  
### Ⅰ. 概念层



### Ⅱ. 应用层




### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  

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


## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [ ] 深入阅读 xxx  
- [ ] 验证这个观点的边界条件  
