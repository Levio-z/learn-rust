`rust-toolchain` 是 Rust 项目中的一个特殊文件，用于指定项目应使用的 Rust 工具链版本。它的主要作用是**确保在不同环境（如开发者本地、CI/CD）下，所有人都用相同的 Rust 编译器版本和通道**（比如 stable、beta、nightly 或具体版本号），避免因版本差异导致的构建、测试或运行问题。
### 应用场景
#### 1. 在 jobs.ci.matrix.rust-toolchain

YAML

```
matrix:
  rust-toolchain: [nightly]
```

这里的 `rust-toolchain` 被设置为 `nightly`，意思是在 CI 运行时，所有的步骤都将使用 Rust 的 nightly 版本（最新的开发版，有很多新特性但相对不稳定）。
#### 2. 在 dtolnay/rust-toolchain@nightly Action
```
- uses: dtolnay/rust-toolchain@nightly
  with:
    toolchain: ${{ matrix.rust-toolchain }}
    components: rust-src, clippy, rustfmt
    targets: ${{ matrix.targets }}
```
- 这里调用了 [dtolnay/rust-toolchain](https://github.com/dtolnay/rust-toolchain) 这个 Action，用来安装并激活你指定的 Rust 工具链。
- `toolchain` 字段就是用来指定你要用的工具链版本，这里实际值等同于 `nightly`（因为你用的是 `${{ matrix.rust-toolchain }}`）。
- `components` 字段指定了要安装的附加组件，比如 `clippy`（代码静态检查）、`rustfmt`（代码格式化工具）、`rust-src`（源码支持，适配 embedded/交叉编译）。
- `targets` 用于指定交叉编译的目标平台。