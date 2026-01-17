>https://doc.rust-lang.org/std/cmp/trait.Eq.html?utm_source=chatgpt.com

与 [`PartialEq`](https://doc.rust-lang.org/std/cmp/trait.PartialEq.html "trait std::cmp::PartialEq") 的主要区别在于对反身性的额外要求。实现 [`PartialEq`](https://doc.rust-lang.org/std/cmp/trait.PartialEq.html "trait std::cmp::PartialEq") 的类型保证对于所有 `a`、`b` 和 `c`：

- symmetric: `a == b` implies `b == a` and `a != b` implies `!(a == b)`  
    对称：`a == b` 表示 `b == a` 和 `a ！= b` 表示 `！（a == b）`
- transitive: `a == b` and `b == c` implies `a == c`  
    及物动词：`a == b` 和 `b == c` 表示 `a == c`

`Eq`, which builds on top of [`PartialEq`](https://doc.rust-lang.org/std/cmp/trait.PartialEq.html "trait std::cmp::PartialEq") also implies:  
`Eq` 建立在 [`PartialEq`](https://doc.rust-lang.org/std/cmp/trait.PartialEq.html "trait std::cmp::PartialEq") 之上，它还意味着：

- reflexive: `a == a`  
    反身：`a == a`

违反此属性是逻辑错误。未指定由逻辑错误导致的行为，但特征的用户必须确保此类逻辑错误_不_会导致未定义的行为。这意味着`不安全的`代码**不得**依赖于这些方法的正确性。

浮点类型（如 [`f32`](https://doc.rust-lang.org/std/primitive.f32.html "primitive f32") 和 [`f64`](https://doc.rust-lang.org/std/primitive.f64.html "primitive f64")）仅实现 [`PartialEq`](https://doc.rust-lang.org/std/cmp/trait.PartialEq.html "trait std::cmp::PartialEq") _而不是_ `Eq` 因为 `NaN` ！= `NaN`。

这句话意味着，**如果你定义了类型的 `PartialEq` 或 `Eq` 实现，但它违反了这些 trait 所要求的行为**（例如对称性、传递性或自反性），那么这就是一个逻辑错误。在正常情况下，`PartialEq` 和 `Eq` 应该满足对称性、传递性和自反性。但如果实现不符合这些原则，比如错误地实现了某些比较规则，那么就会发生逻辑错误

这意味着，Rust 标准库本身**并没有明确规定当违反这些规则时会出现什么行为**。违反规则后，行为将依赖于程序的运行时环境，但具体的错误行为是未定义的。在 Rust 的设计哲学中，标准库并不强制对每一个逻辑错误提供明确的行为描述，而是允许开发者自己决定如何处理。

**特征的用户必须确保此类逻辑错误不会导致未定义的行为**
- 这里强调的是，作为开发者，如果你自定义了类型的 `PartialEq` 或 `Eq` 实现，必须确保它们遵循正确的逻辑规则，否则可能导致程序在运行时出现意外或不可预测的行为。
    
- 这也提示，**不安全的代码**（如使用 `unsafe` 关键字的代码）**不得依赖于这些方法的正确性。也就是说，如果你使用了 `unsafe` 代码，并且依赖于相等性比较的方法，那么你要非常小心，因为这些方法可能会被错误实现，从而导致未定义行为**。


