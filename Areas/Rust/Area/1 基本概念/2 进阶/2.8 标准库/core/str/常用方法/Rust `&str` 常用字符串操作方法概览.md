### Rust `&str` 常用字符串操作方法概览

#### 1. `split_once`

`let s = "blob 123"; 
if let Some((kind, size)) = s.split_once(' ') 
{     println!("kind: {}, size: {}", kind, size); 
}`

- **作用**：返回第一个匹配分隔符前后的两部分
- **返回值**：`Option<(&str, &str)>`
- **特点**：
    - 只分割一次，第一个匹配点即可
    - 文件名或其他内容中包含空格也不会影响
    - 如果没有匹配分隔符，返回 `None`
---

#### 2. `split`
`let parts: Vec<&str> = s.split(' ').collect();`
- **作用**：按所有匹配的分隔符切割
- **返回值**：迭代器 `Split<&str>`
- **特点**：
    - 可多次匹配，生成多片段
    - 遇到连续分隔符，会生成空片段（可以用 `split_whitespace` 避免）
---

#### 3. `splitn` / `rsplitn`
`// 最多分割两次 let parts: Vec<&str> = s.splitn(2, ' ').collect();`
- **作用**：限制分割次数
- `rsplitn`：从右侧开始分割
- **用途**：类似 `split_once`，但可以分多次
---

#### 4. `split_whitespace`

`let words: Vec<&str> = s.split_whitespace().collect();`

- **作用**：按 Unicode 空白字符分割，并自动忽略连续空格
    
- **常用场景**：处理命令行输入、文本解析
    

---

#### 5. `trim` / `trim_start` / `trim_end`

`let s = "  hello  "; let t = s.trim(); // "hello"`

- 去掉前后空白字符
    
- 不会修改原字符串，返回新的 `&str`
    

---

#### 6. `starts_with` / `ends_with`

`s.starts_with("he"); // true s.ends_with("lo");   // true`

- 判断字符串前后缀
    
- 常用于匹配特定模式
    

---

#### 7. `contains`

`s.contains("ell"); // true`

- 检查子串是否存在
    
- 可用于简单匹配
    

---

#### 8. `find` / `rfind`

`if let Some(idx) = s.find('l') {     println!("first 'l' at index {}", idx); }`

- 返回第一个/最后一个匹配字符或子串的位置
    
- 返回 `Option<usize>`
    

---

#### 9. `replace` / `replacen`

`let t = s.replace("l", "L"); // "heLLo"`

- `replace` 替换所有匹配
    
- `replacen` 限制替换次数
    

---

#### 10. `to_lowercase` / `to_uppercase`

`let t = s.to_uppercase(); // "HELLO"`

- 返回新的 `String`，非原地修改
    

---

#### 11. `chars()` / `bytes()` / `lines()`

- `chars()`：按 Unicode 字符迭代
- `bytes()`：按字节迭代

- `lines()`：按换行符迭代
    

---

### 总结

- **分割相关**：
    
    - `split_once` → 第一次匹配，安全用于含空格内容
        
    - `split` → 全部匹配
        
    - `splitn` / `rsplitn` → 限次数分割
        
    - `split_whitespace` → 按空白字符分割
        
- **匹配/查找**：
    
    - `starts_with` / `ends_with` / `contains` / `find` / `rfind`
        
- **清理/转换**：
    
    - `trim` / `trim_start` / `trim_end`
        
    - `to_lowercase` / `to_uppercase`
        
- **迭代**：
    
    - `chars()` / `bytes()` / `lines()`
        

---

### 练习题

1. 用 `split_once` 解析 `"blob 123\0"`，取出 type 和 size
    
2. 用 `splitn(3, ',')` 解析 `"a,b,c,d"`，看看结果
    
3. 用 `split_whitespace` 解析 `" foo bar "`，验证连续空格被忽略
    
4. 用 `find` 找出字符串中第一个 `:` 的位置
    
5. 用 `lines()` 遍历多行文本，打印每行长度