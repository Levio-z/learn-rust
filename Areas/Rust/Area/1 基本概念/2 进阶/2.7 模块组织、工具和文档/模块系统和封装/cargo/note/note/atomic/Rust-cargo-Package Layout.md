---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层
```
.
├── Cargo.lock
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── main.rs
│   └── bin/
│       ├── named-executable.rs
│       ├── another-executable.rs
│       └── multi-file-executable/
│           ├── main.rs
│           └── some_module.rs
├── benches/
│   ├── large-input.rs
│   └── multi-file-bench/
│       ├── main.rs
│       └── bench_module.rs
├── examples/
│   ├── simple.rs
│   └── multi-file-example/
│       ├── main.rs
│       └── ex_module.rs
└── tests/
    ├── some-integration-tests.rs
    └── multi-file-test/
        ├── main.rs
        └── test_module.rs

```
- `Cargo.toml` 和 `Cargo.lock` 存储在您的包的根目录（ _包根目录_ ）中。
- 源代码放在 `src` 目录下。
- 默认库文件是 `src/lib.rs` 。
- 默认可执行文件为 `src/main.rs` 。
- 其他可执行文件可以放在 `src/bin/` 目录中。
- 基准测试结果应放在 `benches` 目录中。
- 示例文件放在 `examples` 目录中。
- 集成测试用例放在 `tests` 目录中。
### 当某个可执行文件非常复杂，需要多个模块时，可以将其放在子目录下

如果二进制文件、示例、基准测试或集成测试包含多个源文件，请将 `main.rs` 文件与额外的[_模块_](https://doc.rust-lang.org/cargo/appendix/glossary.html#module "\"module\" (glossary entry)")放在一起。 在 `src/bin` 、 `examples` 、 `benches` 或 `tests` 的子目录中 目录。可执行文件的名称将与目录名称相同。

- 编译时，生成的可执行文件名仍然是 `server`（目录名）。
- 这种方式可以把一个可执行文件的所有源文件聚集在一起，便于维护。

案例
```
examples/
  basic.rs
  advanced/
    main.rs
    helper.rs
```
- `cargo run --example basic` 会运行 `examples/basic.rs`。
- 对于 `advanced` 子目录，编译器会将目录名 `advanced` 作为可执行文件名，并将 `main.rs` 作为入口，其他模块如 `helper.rs` 可被 `main.rs` 引入。


### 惯例

 按照惯例，二进制文件、示例、基准测试和集成测试都遵循 `kebab-case` 命名风格，除非出于兼容性原因需要另行处理（例如，与已有的二进制文件名称兼容）。这些目标中的模块则遵循 [Rust 标准的](https://rust-lang.github.io/rfcs/0430-finalizing-naming-conventions.html) `snake_case` 命名风格。
 

### Ⅱ. 应用层




### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
	- https://doc.rust-lang.org/cargo/guide/project-layout.html
	- 有关手动配置目标的更多详细信息，请参阅 [“配置目标”](https://doc.rust-lang.org/cargo/reference/cargo-targets.html#configuring-a-target) 。有关控制 Cargo 如何自动推断目标名称的更多信息，请参阅 [“目标自动发现”](https://doc.rust-lang.org/cargo/reference/cargo-targets.html#target-auto-discovery) 。 
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  




## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [x] 深入阅读 xxx
- [x] 验证这个观点的边界条件
