## Rust 的 `continue 'label;` 语法

Rust 允许你给循环加标签（label），然后用 `continue 'label;` 或 `break 'label;` 跳出到指定的循环层级。

- `'next` 就是一个循环标签。
- `continue 'next;` 表示**直接开始下一次外层 `'next` 循环**。

### 你代码的结构（简化版）

假设你的代码结构如下：

Rust  锈

```
'next: for (i, state) in connections.iter_mut().enumerate() {
    if let ConnectionState::Write { response, written } = state {
        loop {
            match connection.write(&response[*written..]) {
                Ok(0) => { 
                    // ... 
                    continue 'next; // 直接跳到外层 for 的下一次循环
                }
                Ok(n) => { *written += n; }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                    continue 'next; // 直接跳到外层 for 的下一次循环
                }
                Err(e) => panic!("{e}"),
            }
            if *written == response.len() { break; }
        }
        *state = ConnectionState::Flush;
    }
    // ... 可能还有别的逻辑
}
```

### `continue 'next;` 作用

- 当你在**内层的 loop 或 match 分支**里写 `continue 'next;`，会**跳出所有内层 loop，直接开始外层 for 的下一个元素**。
- 如果没有标签，`continue` 只会影响最近的循环（也就是 loop），**不会跳到外层 for**。

---

## 对应你的代码场景

- 当遇到 `Ok(0)`（对端断开）或者 `WouldBlock`（还不能写入非阻塞），都直接 `continue 'next;`，不再处理当前连接，**直接处理下一个连接**。
- 这样可以避免复杂的嵌套和冗余的 flag 变量。

---

## 总结

- `continue 'next;` 是**带标签的 continue**，直接进行外层 `'next` 循环的下一轮。
- 常用于多层循环时，想要优雅地跳过当前外层循环的剩余部分。

---

### 拓展阅读

如果还不明白，可以问我“标签循环怎么写”，我可以给你一个完整的小例子