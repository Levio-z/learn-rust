为了确保即使打印很多行且有些行超出屏幕的情况下也没有panic发生，我们可以创建另一个测试：
```rust
#[test_case]
fn test_println_many() {
    for _ in 0..200 {
        println!("test_println_many output");
    }
}
```

我们还可以创建另一个测试函数，来验证打印的几行字符是否真的出现在了屏幕上:

```rust
#[test_case]
fn test_println_output() {
    let s = "Some test string that fits on a single line";
    println!("{}", s);
    for (i, c) in s.chars().enumerate() {
        let screen_char = WRITER.lock().buffer.chars[BUFFER_HEIGHT - 2][i].read();
        assert_eq!(char::from(screen_char.ascii_character), c);
    }
}
```

该函数定义了一个测试字符串，并通过 `println`将其输出，然后遍历静态 `WRITER` 也就是vga字符缓冲区的屏幕字符。由于 `println` 在将字符串打印到屏幕上最后一行后会立刻附加一个新行(即输出完后有一个换行符)，所以这个字符串应该会出现在第 `BUFFER_HEIGHT - 2`行。

通过使用[`enumerate`](https://doc.rust-lang.org/core/iter/trait.Iterator.html#method.enumerate) ，我们统计了变量 `i` 的迭代次数，然后用它来加载对应于`c`的屏幕字符。 通过比较屏幕字符的 `ascii_character` 和 `c` ，我们可以确保字符串的每个字符确实出现在vga文本缓冲区中。

如你所想，我们可以创建更多的测试函数：例如一个用来测试当打印一个很长的且包装正确的行时是否会发生panic的函数，或是一个用于测试换行符、不可打印字符、非unicode字符是否能被正确处理的函数。

在这篇文章的剩余部分，我们还会解释如何创建一个 _集成测试_ 以测试不同组件之间的交互。