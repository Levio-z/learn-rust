
### 复刻报错
```rust
fn main() {

    let mut x = 2;

    let mut y = &mut 4;

    // x的引用生命周期开始

    let ref_x = &mut x;

  

    if 1 > 0 {

        // x再次被分配

        x = 2;

    } else {

        // y绑定ref_x,x引用生命周期延长到函数结尾

        y = ref_x;

    }

    print!("{}", y);

    // 两个充分条件才会发生：1、x的引用生命周期存在 2、引用存在期间发生了x=2的变量再次分配

    // 但是这里有一个问题，这两个条件是if else的两个分支，不可能同时发生

    // Rust的编译器现在是有问题的

}
}
```
结果：
```rust
error[E0506]: cannot assign to `x` because it is borrowed
  --> src\bin\reference.rs:9:9
   |
5  |     let ref_x = &mut x;
   |                 ------ `x` is borrowed here
...
9  |         x = 2;
   |         ^^^^^ `x` is assigned to here but it was already borrowed
...
14 |     print!("{}", y);
   |                  - borrow later used here
```