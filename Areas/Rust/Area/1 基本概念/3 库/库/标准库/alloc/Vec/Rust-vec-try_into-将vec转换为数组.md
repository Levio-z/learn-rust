---
tags:
  - fleeting
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层

`impl<T, A: Allocator, const N: usize> TryFrom<Vec<T, A>> for [T; N]`
- 数组可以使用TryFrom从

```rust
use std::convert::TryFrom;

fn main() {
    let v = vec![1, 2, 3];

    // 直接使用 TryFrom 调用
    let arr: [i32; 3] = <[i32; 3]>::try_from(v).unwrap();

    assert_eq!(arr, [1, 2, 3]);
    println!("{:?}", arr);
}
```


一般直接使用try_into()

```
let arr: [i32; 3] = v.try_into().unwrap();
```


### Ⅱ. 实现层
```rust
impl<T, A: Allocator, const N: usize> TryFrom<Vec<T, A>> for [T; N] {

    type Error = Vec<T, A>;

  

    /// Gets the entire contents of the `Vec<T>` as an array,

    /// if its size exactly matches that of the requested array.

    ///

    /// # Examples

    ///

    /// ```

    /// assert_eq!(vec![1, 2, 3].try_into(), Ok([1, 2, 3]));

    /// assert_eq!(<Vec<i32>>::new().try_into(), Ok([]));

    /// ```

    ///

    /// If the length doesn't match, the input comes back in `Err`:

    /// ```

    /// let r: Result<[i32; 4], _> = (0..10).collect::<Vec<_>>().try_into();

    /// assert_eq!(r, Err(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]));

    /// ```

    ///

    /// If you're fine with just getting a prefix of the `Vec<T>`,

    /// you can call [`.truncate(N)`](Vec::truncate) first.

    /// ```

    /// let mut v = String::from("hello world").into_bytes();

    /// v.sort();

    /// v.truncate(2);

    /// let [a, b]: [_; 2] = v.try_into().unwrap();

    /// assert_eq!(a, b' ');

    /// assert_eq!(b, b'd');

    /// ```

    fn try_from(mut vec: Vec<T, A>) -> Result<[T; N], Vec<T, A>> {

        if vec.len() != N {

            return Err(vec);

        }

  

        // SAFETY: `.set_len(0)` is always sound.

        unsafe { vec.set_len(0) };

  

        // SAFETY: A `Vec`'s pointer is always aligned properly, and

        // the alignment the array needs is the same as the items.

        // We checked earlier that we have sufficient items.

        // The items will not double-drop as the `set_len`

        // tells the `Vec` not to also drop them.

        let array = unsafe { ptr::read(vec.as_ptr() as *const [T; N]) };

        Ok(array)

    }

}

```


### Ⅲ. 原理层

## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
详细阐述这个观点，包括逻辑、例子、类比。  
- 要点1  
- 要点2  

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
