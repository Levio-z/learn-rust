---
tags:
  - note
---

## 1. 核心观点  



## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  

```rust
async fn example(min_len: usize) -> String {
    let content = async_read_file("foo.txt").await;
    if content.len() < min_len {
        content + &async_read_file("bar.txt").await
    } else {
        content
    }
}
```

### 状态机
在底层，编译器将 `async` 函数体转换为一个 [状态机](https://en.wikipedia.org/wiki/Finite-state_machine) ，每次调用 `.await` 都代表一个不同的状态。对于上述 `example` 函数，编译器会创建一个包含以下四种状态的状态机：

![](asserts/Pasted%20image%2020251211162821.png)

每个状态代表函数执行过程中的不同暂停点。_“Start”_ 和 _“End”_ 状态分别表示函数执行的开端和终止。_“Waiting on foo.txt”_ 状态表示该函数当前正在等待第一个 `async_read_file` 的结果。类似的，_“Waiting on bar.txt”_ 状态表示该函数正在等待第二个 `async_read_file` 的结果。

状态机通过将每次 `poll` 调用作为可能的状态转换来实现 `Future` trait：
![](asserts/Pasted%20image%2020251211162835.png)

图表使用箭头表示状态转换，菱形表示条件分支。例如，如果 `foo.txt` 文件尚未就绪，则会选择标记为 _“no”_ 的分支，到达 _“Waiting on foo.txt”_ 状态。否则，将执行 _“yes”_ 分支。那个小的无标注的红色菱形代表 `example` 函数中 `if content.len() < 100` 分支。


我们看到第一次 `poll` 调用启动了该函数并让其运行，直到遇到一个尚未可用的 future。如果路径上所有 future 都已就绪，函数可以一直运行到 _“End”_ 状态，此时它会返回包裹在 `Poll::Ready` 中的结果。否则，状态机将进入等待状态并且返回 `Poll::Pending`。在下一次 `poll` 调用时，状态机将从上次的等待状态中恢复并尝试之前的操作。
### 保存状态

**`async fn` 会被编译成一个可暂停/可恢复的状态机，任何在未来恢复执行时仍需要使用的变量，都必须被存入状态机的结构体中。**

换句话说：

> **只要一个变量在未来某个 `await` 之后仍要使用，它就必须被保存。**



为了能够从上一个等待状态继续执行，状态机必须在内部跟踪当前状态。此外，它还必须保存所有在下一次 `poll` 调用时继续执行所需的变量。这正是编译器可以大显身手的地方：因为它知道哪些变量在何时被使用，当需要时，它能自动生成包含这些确切变量的结构体。

例如，编译器会为上述 `example` 函数生成类似以下的结构体：


```rust
// 这里再写一遍 `example` 函数，方便阅读
async fn example(min_len: usize) -> String {
    let content = async_read_file("foo.txt").await;
    if content.len() < min_len {
        content + &async_read_file("bar.txt").await
    } else {
        content
    }
}

// 由编译器生成的状态结构体：

struct StartState {
    min_len: usize,
}

struct WaitingOnFooTxtState {
    min_len: usize,
    foo_txt_future: impl Future<Output = String>,
}

struct WaitingOnBarTxtState {
    content: String,
    bar_txt_future: impl Future<Output = String>,
}

struct EndState {}
```

在 “start” 和 _“Waiting on foo.txt”_ 状态下，需要存储 `min_len` 参数以供稍后与 `content.len()` 进行比较。 _“Waiting on foo.txt”_ 状态额外存储了一个 `foo_txt_future` ，它代表 `async_read_file` 调用返回的 future。这个 future 在状态机继续运行时需要再次被轮询，因此需要保存它。

_“Waiting on bar.txt”_ 状态包含用于后续 `bar.txt` 准备就绪时字符串拼接的 `content` 变量。它还存储了一个 `bar_txt_future` ，用于表示正在加载中的 `bar.txt` 。

该结构体不再包含 `min_len` 变量，因为在 `content.len()` 比较之后就不再需要它。在 _“end”_ 状态，不会存储任何变量，因为函数已经运行完毕。

请注意，这只是编译器可能生成的代码示例。结构体名称和字段布局的实现细节可能会有所不同。
### 完整状态机状态

虽然编译器生成的具体代码属于实现细节，但想象一下为 `example` 函数生成的状态机 _可能_ 是什么样子有助于理解。我们已经定义了表示不同状态并包含所需变量的结构体。为了基于它们创建一个状态机，我们可以将它们组合成一个 [`enum`](https://doc.rust-lang.org/book/ch06-01-defining-an-enum.html)：

```rust
enum ExampleStateMachine {
    Start(StartState),
    WaitingOnFooTxt(WaitingOnFooTxtState),
    WaitingOnBarTxt(WaitingOnBarTxtState),
    End(EndState),
```

我们为每个状态定义独立的枚举变体，并为每个变体添加对应的状态结构体作为字段。为实现状态转换，编译器会生成一个基于 `example` 函数的 `Future` trait：

```rust
impl Future for ExampleStateMachine {
    type Output = String; // `example` 的返回类型

    fn poll(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Self::Output> {
        loop {
            match self { // TODO: 处理 pinning
                ExampleStateMachine::Start(state) => {…}
                ExampleStateMachine::WaitingOnFooTxt(state) => {…}
                ExampleStateMachine::WaitingOnBarTxt(state) => {…}
                ExampleStateMachine::End(state) => {…}
            }
        }
    }
}
```
该 future 的 `Output` 类型是 `String` ，因为它是 `example` 函数的返回类型。为了实现 `poll` 函数，我们在 `loop` 内对当前状态使用 `match` 语句。其核心思想是只要可能就切换到下一个状态，并在无法继续时使用显式的 `return Poll::Pending` 。

为简单起见，我们仅展示简化代码，不处理 [_pinning_](https://doc.rust-lang.org/stable/core/pin/index.html)、所有权、生命周期等问题。因此，当前及后续代码应视为伪代码，不可直接使用。当然，编译器实际生成的代码会正确处理所有情况，尽管实现方式可能有所不同。
为保持代码片段简洁，我们将分别展示每个 `match` 分支的代码。让我们从 `Start` 状态开始：
```rust
ExampleStateMachine::Start(state) => {
    // 来自 `example` 函数体
    let foo_txt_future = async_read_file("foo.txt");
    // `.await` 运算符
    let state = WaitingOnFooTxtState {
        min_len: state.min_len,
        foo_txt_future,
    };
    *self = ExampleStateMachine::WaitingOnFooTxt(state);
}
```
状态机在函数刚开始时为 `Start` 状态。在这种情况下，我们会执行 `example` 函数体内的所有代码，直到遇到第一个 `.await` 为止。为了处理 `.await` 运算符，我们会将 `self` 状态机的状态设置为 `WaitingOnFooTxt`，其中包括了 `WaitingOnFooTxtState` 结构体的构建。

由于 `match self {…}` 语句是在循环中执行的，执行流程会跳转到之后的 `WaitingOnFooTxt` 分支：
```rust
ExampleStateMachine::WaitingOnFooTxt(state) => {
    match state.foo_txt_future.poll(cx) {
        Poll::Pending => return Poll::Pending,
        Poll::Ready(content) => {
            // 来自 `example` 函数体
            if content.len() < state.min_len {
                let bar_txt_future = async_read_file("bar.txt");
                // `.await` 运算符
                let state = WaitingOnBarTxtState {
                    content,
                    bar_txt_future,
                };
                *self = ExampleStateMachine::WaitingOnBarTxt(state);
            } else {
                *self = ExampleStateMachine::End(EndState);
                return Poll::Ready(content);
            }
        }
    }
}
```
在这个 `match` 分支中，我们首先调用 `foo_txt_future` 的 `poll` 函数。如果它尚未就绪，我们直接退出循环并返回 `Poll::Pending` 。由于此时 `self` 仍处于 `WaitingOnFooTxt` 状态，状态机的下一次 `poll` 调用将进入相同的 `match` 分支，并重新尝试轮询 `foo_txt_future`。

当 `foo_txt_future` 就绪时，我们将结果赋值给 `content` 变量并继续执行 `example` 函数的代码：如果 `content.len()` 小于状态结构体中保存的 `min_len` 则异步读取 `bar.txt` 文件。我们再次将 `.await` 操作转换为状态变更，这次变更为 `WaitingOnBarTxt` 状态。由于我们是在循环中执行 `match` 操作，执行流程会直接跳转到新状态对应的 `match` 分支继续处理。其中会对 `bar_txt_future` 进行轮询。
`WaitingOnBarTxt` 状态的代码如下所示：

```rust
ExampleStateMachine::WaitingOnBarTxt(state) => {
    match state.bar_txt_future.poll(cx) {
        Poll::Pending => return Poll::Pending,
        Poll::Ready(bar_txt) => {
            *self = ExampleStateMachine::End(EndState);
            // 来自 `example` 函数体
            return Poll::Ready(state.content + &bar_txt);
        }
    }
}
```
与 `WaitingOnFooTxt` 状态类似，我们首先轮询 `bar_txt_future` 。如果它仍然如果处于 pending 状态，我们退出循环并返回 `Poll::Pending` 。否则，我们可以执行 `example` 函数最后的操作：拼接 `content` 以及 future 的返回值。我们将状态机更新为 `End` 状态，然后返回包装在 `Poll::Ready` 中的结果。
最终，`End` 状态的代码如下所示：
```rust
ExampleStateMachine::End(_) => {
    panic!("poll called after Poll::Ready was returned");
}
```
Futures 在返回 `Poll::Ready` 后不应再次轮询，所以在处于 `End` 状态时发生 `poll` 调用，则直接 panic。

我们现在已经了解了编译器生成的状态机及其对 Future 的实现 _可能_ 的样子。实际上，编译器是以另一种方式生成代码的。（如果你感兴趣的话：这个实现当前基于 [协程](https://doc.rust-lang.org/stable/unstable-book/language-features/coroutines.html)，但这仅仅是一种实现细节。）

拼图的最后一块是为 `example` 函数本身生成的代码。记住，函数签名是这样定义的：

```rust
async fn example(min_len: usize) -> String
```
由于完整函数体现在已由状态机实现，唯一需要该函数完成的是初始化状态机并返回它。生成的代码可能如下所示：

```rust
fn example(min_len: usize) -> ExampleStateMachine {
    ExampleStateMachine::Start(StartState {
        min_len,
    })
}
```


该函数不再具有 `async` 修饰符，因为它现在显式返回一个实现了 `Future` trait 的 `ExampleStateMachine` 类型。正如预期的那样，这个状态机构建出来处于 `Start` 状态，并使用 `min_len` 参数初始化对应的状态结构体。

请注意，此函数不会启动状态机的执行。这是 Rust 中 future 的一个基本设计决策：在首次被轮询之前，它们不会执行任何操作。



## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [ ] 深入阅读更多详情请参阅 [_Rust中的零成本 futures_](https://aturon.github.io/blog/2016/08/11/futures/) 文章，它宣布了 futures 被加入 Rust 生态系统的消息。
- [ ] 验证这个观点的边界条件  
