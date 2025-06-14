```yaml
name: CI

on: [push, pull_request]
```
- **name: CI**：这个 workflow 的名字，显示在 GitHub Actions UI。
- **on: [push, pull_request]**：当代码被 push 到仓库或有 pull request 时触发本 workflow。  
    **on： [push， pull_request]**：当代码被 push 到仓库或有 pull request 时触发本 workflow。
# jobs: CI 主流程
```yaml
jobs:
  ci:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        rust-toolchain: [nightly]
        targets: [x86_64-unknown-linux-gnu, x86_64-unknown-none, riscv64gc-unknown-none-elf, aarch64-unknown-none-softfloat]
```
- `jobs:` 下可以有多个**不同的 job（任务/作业）**，每个 job 可以并行、独立地运行
-  `ci:` 是**其中一个 job 的名字**，你可以随便取，这里通常代表“持续集成”（Continuous Integration）的意思。
- strategy:
	- 是用来**配置 job 的并行执行策略和参数矩阵**的一个字段，常用于自动化测试、构建等场景。
- **runs-on: ubuntu-latest**：每个 matrix 任务运行在 Ubuntu 的最新 runner 上。
	-  **`runs-on:`** 指定了 job（任务）运行的操作系统环境，也叫 runner（运行器）。
	- **`ubuntu-latest`** 是 GitHub 官方提供的最新版本的 Ubuntu Linux 云服务器环境。
- **strategy.fail-fast: false**：如果某个 matrix 任务失败，不会立刻终止其它任务，便于多平台并行观察多任务结果。
- **matrix**：定义了多个参数组合（rust-toolchain、targets），每种组合都会并行执行一次 job。这里会生成 4 个 target × 1 个 toolchain = 4 个任务。
- 
### CI steps 详解
**checkout 代码**
```yaml
- uses: actions/checkout@v4
```
拉取当前仓库代码，供后续步骤使用。
**设置 Rust 工具链及组件**
```yaml
- uses: dtolnay/rust-toolchain@nightly
  with:
    toolchain: ${{ matrix.rust-toolchain }}
    components: rust-src, clippy, rustfmt
    targets: ${{ matrix.targets }}
```
- 使用 `dtolnay/rust-toolchain` action 安装指定 Rust 工具链（nightly）、组件和 cross target 支持。
- `components`: rust-src（交叉编译必需）、clippy（代码静态检查）、rustfmt（格式化工具）。
- `targets`: 当前 matrix 的目标平台。
### **检查 Rust 版本**
```
- name: Check rust version
  run: rustc --version --verbose
```
- 便于日志排查，输出当前 Rust 工具链详细版本信息。
### **代码格式检查**
```rust
- name: Check code format
  run: cargo fmt --all -- --check
```
- 检查所有代码格式是否符合 rustfmt 标准（不自动修复，仅检查）。
### **Clippy 静态检查**
```rust
- name: Clippy
  run: cargo clippy --target ${{ matrix.targets }} --all-features -- -A clippy::new_without_default
```
- 用 Clippy 进行静态代码分析。
- 针对所有目标和功能。
- `-A clippy::new_without_default`：对 new_without_default lint 警告不报错（部分项目习惯）。
### **构建**
```
- name: Build
  run: cargo build --target ${{ matrix.targets }} --all-features
```
- 构建所有功能、指定目标架构。
### **单元测试**
```
- name: Unit test
  if: ${{ matrix.targets == 'x86_64-unknown-linux-gnu' }}
  run: cargo test --target ${{ matrix.targets }} -- --nocapture
```
- 仅在 host 平台（x86_64-unknown-linux-gnu）上执行测试，其它交叉平台通常无法直接运行测试。
- `--nocapture`：让测试输出直接显示在日志中。
### # jobs: doc 文档构建和部署
```yaml
  doc:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
    permissions:
      contents: write
    env:
      default-branch: ${{ format('refs/heads/{0}', github.event.repository.default_branch) }}
      RUSTDOCFLAGS: -D rustdoc::broken_intra_doc_links -D missing-docs

```
- **runs-on**: 依然是 Ubuntu。
- **strategy.fail-fast: false**：文档构建失败不影响其它 job。
- **permissions.contents: write**：允许推送内容（部署到 gh-pages）。
- **env**:
    - `default-branch`: 取当前仓库默认分支名，便于后续条件判断。
    - `RUSTDOCFLAGS`: 强制文档链接和注释完整性。
#### ## doc steps 详解
**checkout**
```
- uses: actions/checkout@v4
```
**rust-toolchain**
```
- uses: dtolnay/rust-toolchain@nightly
```
**构建文档**
```
- name: Build docs
  continue-on-error: ${{ github.ref != env.default-branch && github.event_name != 'pull_request' }}
  run: |
    cargo doc --no-deps --all-features
    printf '<meta http-equiv="refresh" content="0;url=%s/index.html">' $(cargo tree | head -1 | cut -d' ' -f1) > target/doc/index.html
```
- `continue-on-error`: 如果不是默认分支或是 PR 时，构建文档失败不导致整个 job 失败（通常只在主分支上强制文档构建通过）。
- `cargo doc --no-deps --all-features`：编译生成所有功能的文档，不包含依赖。
- `printf ... > target/doc/index.html`：生成一个自动跳转的 index.html，默认跳转到主 crate 文档页面。
**部署到 GitHub Pages**
```
- name: Deploy to Github Pages
  if: ${{ github.ref == env.default-branch }}
  uses: JamesIves/github-pages-deploy-action@v4
  with:
    single-commit: true
    branch: gh-pages
    folder: target/doc
```
- 只有在默认分支上才执行部署（防止 PR 或临时分支污染文档）。
- 使用 JamesIves 的 gh-pages 部署 action，将文档发布到 gh-pages 分支，内容为 `target/doc` 文件夹。
# 总结执行流程

1. **每次 push 或 PR 时自动触发**。
2. **ci job**：分别在所有平台和工具链组合上，完成编译、clippy、格式检查和（主机平台）测试。
3. **doc job**：构建文档，并在默认分支自动发布到 GitHub Pages。
4. **多平台并发**，失败互不影响，便于维护高质量、兼容性好的 Rust 项目。