`map(fn) -> Option<U>`
```rust
let x = Some(2);
let y = x.map(|v| v * 3); // y = Some(6)
```

`map_or(default, fn) -> U`
```rust
let x = None;
let y = x.map_or(10, |v| v * 2); // y = 10
```

`and_then(fn) -> Option<U>`
```rust
fn double(x: i32) -> Option<i32> {
    if x % 2 == 0 { Some(x * 2) } else { None }
}
let x = Some(4).and_then(double); // x = Some(8)
```
- 类似于 `flatMap`，闭包返回的是 `Option` 而不是原始值。