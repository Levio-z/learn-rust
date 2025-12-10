- `other-tip` 表示普通引用（ref → object）。
- `other-peeled` 用 `"^{}"` 表示 tag 剥离后的真实对象。

###  other-ref

`other-ref` 是 Git **pkt-line 协议**中用于描述“其他引用（refs）信息”的一行，它本质上是一个 **pkt-line 封装的引用描述记录**，格式允许两种：

- `other-tip`
    
- `other-peeled`
    

也就是说，一个 `other-ref` = 一行 pkt-line，其中内容要么是某个 ref 指向的对象，要么是其剥离（peeled）后的对象。

---

###  other-tip —— 普通引用（ref → object）

格式：

```
obj-id SP refname
```

包含两个字段：

|字段|含义|
|---|---|
|`obj-id`|40字节 SHA-1 或 64字节 SHA-256 的对象 ID（commit / tag / tree / blob）|
|`refname`|引用名称，如 `refs/heads/main`、`refs/tags/v1.0`|
|`SP`|单个空格分隔|

**含义：**  
该引用直接指向 `obj-id` 这个对象。例如：

```
ab12cd34... refs/heads/main
```

表示 `main` 分支指向一个提交对象。

---

###  other-peeled —— 剥离后的 tag（annotated tag → commit）

格式：

```
obj-id SP refname "^{}"
```

区别是最后有 `"^{}"` 后缀。

Git 中 **annotated tag（带附注 tag）指向一个 tag object**，而不是 commit 本体。  
“剥离（peeled）”后的值表示 **tag 最终指向的底层对象**（通常是 commit）。

例如：

```
1234abcd... refs/tags/v1.0^{}
```

意思是：

- `refs/tags/v1.0` 原本指向一个 tag 对象（tag-object）
    
- 该 tag 指向的实际 commit 是 `1234abcd...`
    
- 这个剥离后的对象用 `"^{}"` 表示
    

**用途**：让客户端无需再次解析 tag object，就能知道 tag 真正指向哪个 commit，提升 clone/fetch 效率。

---

### ###### 总结 + 学习方法 + 习题

**总结：**

- `other-ref` 是 pkt-line 形式的 Git 引用描述。
    
- `other-tip` 表示普通引用（ref → object）。
- `other-peeled` 用 `"^{}"` 表示 tag 剥离后的真实对象。
    
- 剥离信息让客户端无需解析 tag 对象即可获取底层 commit，提高协议效率。
    

**学习方法论：**

1. 先理解 Git 逻辑对象：ref、tag-object、commit 的关系。
    
2. 再对照 Git 协议文档验证每种 pkt-line 格式。
    
3. 最后用抓包（`GIT_TRACE_PACKET=1`）观察真实 clone/fetch 流量，加深理解。
    

**自测习题：**

1. 写出一个表示 `refs/heads/dev` 指向 commit 的 `other-tip` 示例 pkt-line 内容。
    
2. 针对 `refs/tags/v2.0`（annotated tag），分别写出 `tip` 和 `peeled` 行的内容。
    
3. 解释为什么 Git 协议需要提供 `peeled` 信息，而不仅仅是提供 tag object 的 ID。
    

如需，我可以生成真实 Git 抓包例子并逐行解释。