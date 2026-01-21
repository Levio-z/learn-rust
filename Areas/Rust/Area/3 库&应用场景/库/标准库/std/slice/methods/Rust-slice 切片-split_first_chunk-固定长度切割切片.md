---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层

Returns an array reference to the first `N` items in the slice and the remaining slice.

### Ⅱ. 实现层



### Ⅲ. 原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
```rust
/// Returns an array reference to the first `N` items in the slice and the remaining slice.

    ///

    /// If the slice is not at least `N` in length, this will return `None`.

    ///

    /// # Examples

    ///

    /// ```

    /// let x = &[0, 1, 2];

    ///

    /// if let Some((first, elements)) = x.split_first_chunk::<2>() {

    ///     assert_eq!(first, &[0, 1]);

    ///     assert_eq!(elements, &[2]);

    /// }

    ///

    /// assert_eq!(None, x.split_first_chunk::<4>());

    /// ```

    #[inline]

    #[stable(feature = "slice_first_last_chunk", since = "1.77.0")]

    #[rustc_const_stable(feature = "slice_first_last_chunk", since = "1.77.0")]

    pub const fn split_first_chunk<const N: usize>(&self) -> Option<(&[T; N], &[T])> {

        let Some((first, tail)) = self.split_at_checked(N) else { return None };

  

        // SAFETY: We explicitly check for the correct number of elements,

        //   and do not let the references outlive the slice.

        Some((unsafe { &*(first.as_ptr().cast_array()) }, tail))

    }

```

## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
	-    
- 因此数组的 `get` 方法与切片一致：
    

`let arr: [i32; 3] = [1, 2, 3]; let x: Option<&i32> = arr.get(1);`

- 数组也有 `get_mut` 可获得可变引用
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
