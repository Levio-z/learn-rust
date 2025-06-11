### 第一行把你的代码下载到 runner。
```
steps:
  - uses: actions/checkout@v4
```
**作用**：  
拉取（checkout）你的仓库代码到 GitHub Actions 的 runner 上。

- runner 是 GitHub 提供的虚拟机或容器环境，后续所有步骤都在这个环境里执行。
- 这一步确保你的源代码可用于后续的编译、测试等操作。
- `@v4` 表示使用 actions/checkout 的第4个大版本（最新版，推荐）。
### 安装你希望用的 Rust 版本和工具
1. 第二步安装你希望用的 Rust 版本和工具，并配置好交叉编译环境，方便后面可以直接 cargo build、cargo test、cargo clippy 等。
```
  - uses: dtolnay/rust-toolchain@nightly
    with:
      toolchain: ${{ matrix.rust-toolchain }}
      components: rust-src, clippy, rustfmt
      targets: ${{ matrix.targets }}
```
**作用**：  

安装你需要的 Rust 工具链及相关组件/目标架构。
- `dtolnay/rust-toolchain@nightly` 是社区常用的 Rust 工具链安装 action。
- 这个 action 可能支持一些自定义的参数。  这些参数就通过 `with:` 语法块来传递。
	- `toolchain: ${{ matrix.rust-toolchain }}`
	    - 使用 matrix 里指定的 Rust 版本（如 nightly）。
	- `components: rust-src, clippy, rustfmt`
	    - 安装 rust-src（交叉编译用），clippy（代码静态检查器），rustfmt（代码格式化工具）。
	- `targets: ${{ matrix.targets }}`
	    - 安装 matrix 里指定的目标平台（如 x86_64-unknown-linux-gnu、riscv64gc-unknown-none-elf 等），用于交叉编译。

name:
-  **作用**：给这一步（step）起一个**可读的名字**。
- **显示位置**：这个名字会显示在 GitHub Actions 的执行日志界面里，让你一眼就能看到每一步在做什么。
```
- name: Check rust version
  run: rustc --version --verbose
```
这一步的名字叫“Check rust version”，你在操作界面左侧可以清楚看到这个标题。
run:
- **作用**：告诉 GitHub Actions **要在这一步运行什么命令**（shell 命令）。
- **用法**：写你想执行的 Linux 或 Windows 命令，和在本地终端输命令一样。
```
- name: Check rust version
  run: rustc --version --verbose
```
这行会在 runner 上执行 `rustc --version --verbose`，输出 Rust 编译器的详细版本信息。
```
- name: Check code format

      run: cargo fmt --all -- --check

    - name: Clippy

      run: cargo clippy --target ${{ matrix.targets }} --all-features -- -A clippy::new_without_default

    - name: Build

      run: cargo build --target ${{ matrix.targets }} --all-features

    - name: Unit test

      if: ${{ matrix.targets == 'x86_64-unknown-linux-gnu' }}

      run: cargo test --target ${{ matrix.targets }} -- --nocapture
```
-  cargo fmt --all -- --check
	- - `cargo fmt`  
	    使用 [rustfmt](https://github.com/rust-lang/rustfmt) 工具对 Rust 代码进行自动格式化。
	- `--all`  
	    对整个工作区（workspace）里的所有 crate 进行格式化检查，而不仅仅是当前 crate。
	- `--`  
	    这表示后面的参数会直接传递给 rustfmt（而不是 cargo）。
	- `--check`  
	    只检查代码格式是否符合 rustfmt 的规范，并**不会自动修改文件**。如果有不合规范的地方，命令会返回非零状态（CI 失败）。
- **本地开发**：可以用来检查你的代码格式是否合规。
- **CI 流程中**：
	- 常用于自动化检查，保证所有提交的代码都符合统一风格。  如果格式不规范，CI 会失败，开发者必须先格式化好代码再提交。**检查 Rust 项目所有代码是否格式规范，不规范就报错但不自动修复。**  常用于 CI 自动化流程，保证代码风格统一。
	
- run: cargo clippy --target ${{ matrix.targets }} --all-features -- -A clippy::new_without_default
	- `cargo clippy`
	    - 运行 [Clippy](https://github.com/rust-lang/rust-clippy)，这是 Rust 官方的代码静态检查器，专门用来发现常见的代码问题和改进建议。
	- `--target ${{ matrix.targets }}`
	    - 指定要检查的编译目标平台（比如 x86_64-unknown-linux-gnu、riscv64gc-unknown-none-elf 等）。
	    - `${{ matrix.targets }}` 是 GitHub Actions matrix 里的变量，代表当前这一步要检查的目标架构。
	- `--all-features`
	    - 在所有 feature（功能开关）都启用的情况下检查代码，保证所有功能组合下都不会出错。
	- `--`
	    - 这表示后面的参数会直接传递给 Clippy（而不是 cargo）。
	- `-A clippy::new_without_default`
	    - `-A` 是 allow 的意思，即“允许/不报错”。
	    - `clippy::new_without_default` 是 Clippy 的一条 lint，意思是如果你实现了 `fn new()` 但没实现 `Default` trait，Clippy 会警告。这里用 `-A` 关闭这个警告。
	
- run: cargo build --target ${{ matrix.targets }} --all-features
	- `cargo build`  
	    Rust 的官方构建命令，用于编译项目生成可执行文件或库。
	- `--target ${{ matrix.targets }}`  
	    指定编译的目标平台（架构），比如 `x86_64-unknown-linux-gnu`、`riscv64gc-unknown-none-elf` 等。  
	    `${{ matrix.targets }}` 是 GitHub Actions matrix 设置的一个变量，代表当前这一步要构建的目标平台。
	- `--all-features`  
	    构建时启用项目中定义的**所有 Cargo feature**。这样可以保证所有可选功能都能正常编译，不会有遗漏。
- 在 CI（持续集成）流程中，这样写可以确保你的项目能在**所有你声明支持的平台**、**所有 feature 组合下**都能正常通过编译，不会因为某个平台或某个 feature 被遗漏而出现编译错误。
- 在 CI 中，这条命令的作用就是**验证你的 Rust 代码在所有目标平台和所有功能下都能编译通过**，保证项目的跨平台和可选功能兼容性。

```
 if: ${{ matrix.targets == 'x86_64-unknown-linux-gnu' }}

      run: cargo test --target ${{ matrix.targets }} -- --nocapture
```
- **作用**：这一步（step）只会在 `${{ matrix.targets }}` 的值等于 `x86_64-unknown-linux-gnu` 时才会执行。
- `${{ matrix.targets }}` 是 matrix 策略里定义的变量，代表当前 job 的目标平台。
- **通常原因**：CI 环境只在 Linux 主机上直接运行测试，其他如嵌入式/交叉编译平台（如 riscv、arm 等）无法直接跑测试，所以要加条件。

- **作用**：运行 Rust 的测试命令，测试目标为 `${{ matrix.targets }}`（这里是 `x86_64-unknown-linux-gnu`）。
	- `cargo test` 会编译并运行所有测试用例。
	- `--target ...` 指定测试的目标平台。
	- `--` 表示后面参数传给测试运行器（test runner）。
	- `--nocapture` 让测试期间所有 `println!` 等输出都直接显示在 CI 日志里，方便调试。
- 这两行的意思是：**只有在 Linux x86_64 主机平台时，才会运行所有测试，并把测试输出完整打印出来。**  
	- 这样做可以避免在交叉编译平台（比如嵌入式架构）上误跑测试导致失败。

### 文档
```yaml
permissions:
  contents: write
```
 permissions:
 - 通常出现在 **GitHub Actions workflow** 的 YAML 文件里，用来设置**这个 workflow 运行时的 GitHub Token（GITHUB_TOKEN）具有什么权限**。
 - 这是 GitHub Actions workflow 的一个顶级字段，用来**声明 GITHUB_TOKEN 的权限范围**。
- 通过这个字段可以**最小化权限原则**，提高安全性，只给 workflow 需要的权限。
- contents: write
	- **授予“写入内容”的权限**。
	- 意味着允许这个 workflow **对仓库内容有写权限**，比如可以推送代码、创建/修改文件等。
	- 典型场景
		- 自动发布 Release
		- 自动推送更改（比如自动格式化、生成文档后推送回仓库）
```
	env: default-branch: ${{ format('refs/heads/{0}', github.event.repository.default_branch) }} RUSTDOCFLAGS: -D rustdoc::broken_intra_doc_links -D missing-docs
```
- - **作用**：设置了一个名为 `default-branch` 的环境变量。
- **值**：
    - `${{ ... }}` 是 GitHub Actions 的表达式语法。
    - `github.event.repository.default_branch` 代表当前仓库的默认分支名（通常是 `main` 或 `master`）。
    - `format('refs/heads/{0}', ...)` 把分支名拼接成完整的引用路径，比如 `refs/heads/main`。
- USTDOCFLAGS: -D rustdoc::broken_intra_doc_links -D missing-docs
	-  **作用**：设置 Rust 文档生成工具 rustdoc 的环境变量。
	- `RUSTDOCFLAGS` 会被 `cargo doc` 等命令读取，用来传递额外参数给 rustdoc。
	- **参数解释**：
	    - `-D rustdoc::broken_intra_doc_links`：遇到文档中损坏的“内部文档链接”时，直接报错（而不是警告）。
	    - `-D missing-docs`：遇到未写文档注释的项时，直接报错（强制所有公开项都要有文档）。
	- **用途**：
	    - 在 CI 中，确保你的文档链接都是有效的，且所有公开项都有文档注释，否则 CI 会失败，强制规范文档质量。
```
  steps:

    - uses: actions/checkout@v4

    - uses: dtolnay/rust-toolchain@nightly

    - name: Build docs

      continue-on-error: ${{ github.ref != env.default-branch && github.event_name != 'pull_request' }}

      run: |

        cargo doc --no-deps --all-features

        printf '<meta http-equiv="refresh" content="0;url=%s/index.html">' $(cargo tree | head -1 | cut -d' ' -f1) > target/doc/index.html

    - name: Deploy to Github Pages

      if: ${{ github.ref == env.default-branch }}

      uses: JamesIves/github-pages-deploy-action@v4

      with:

        single-commit: true

        branch: gh-pages

        folder: target/doc
```
-  `continue-on-error: ${{ github.ref != env.default-branch && github.event_name != 'pull_request' }}`
	- 这是 GitHub Actions 的一个特殊字段，**控制本步骤失败时 workflow 是否继续执行**。
	- 表达式含义：
	    - 只有当**当前分支不是默认分支**（如 main/master），**且事件不是 pull_request** 时，遇到本步骤失败才会继续执行。
	    - 反过来说，如果是在默认分支、或者是 PR 事件，本步骤失败会导致整个 workflow 失败（严格要求）。
	- 常见场景：对于开发分支或其他分支，文档生成失败不阻塞 CI；但在主分支或 PR 时，文档必须能正确生成，否则 CI 失败。
-  `run: |`
	- 这里的 `|` 代表多行 shell 脚本。
	-  ① `cargo doc --no-deps --all-features`
		- 用 Rust 官方工具生成文档。
		- `--no-deps` 只为你的 crate 生成文档，不生成依赖库的文档。
		- `--all-features` 打开所有 feature，保证所有功能的代码都有文档。
	- ②`printf '<meta http-equiv="refresh" content="0;url=%s/index.html">' $(cargo tree | head -1 | cut -d' ' -f1) > target/doc/index.html`
		-  作用：在 `target/doc/index.html` 生成一个 HTML 文件，自动跳转到你的主 crate 的文档首页。
		- printf 
			- `printf` 会按照你给的“格式字符串”格式输出后面的参数。
		- `cargo tree | head -1 | cut -d' ' -f1` 获取你的主 crate 名字（比如 my_crate）。
			-  `cargo tree`
				- 输出 Rust 项目的依赖树，第一行通常是你的主 crate 名字
					- my_crate v0.1.0 (/path/to/my_crate)
					- ├── dep1 v0.2.0
			-  `head -1`
				- - 作用：只取输出的**第一行**，即主 crate 这一行。
			-  `cut -d' ' -f1`
				- 用空格 `' '` 作为分隔符，取**第一个字段**，也就是主包名（如 `my_crate`）。
			-  `$( ... )`
				- 作用：把上面整个命令的输出结果（主包名）作为一个字符串，传递给外层命令。
- name: Deploy to Github Pages
	- `if: ${{ github.ref == env.default-branch }}`
		- - 只有当当前 workflow 运行在**默认分支**（通常是 `main` 或 `master`）时，这一步才会执行。
		- 这样可以避免开发分支或 PR 自动部署覆盖线上文档。
	- `uses: JamesIves/github-pages-deploy-action@v4`
		- -使用社区维护的 [github-pages-deploy-action](https://github.com/JamesIves/github-pages-deploy-action) v4 版本。
		- 这是 GitHub Pages 自动部署的主流 action，支持多种高级用法。
		-  `with:` 部分参数
			- `single-commit: true`
			    - 部署时只保留最新一次 commit，简化 gh-pages 历史，减少空间占用。
			- `branch: gh-pages`
			    - 指定部署到的分支为 `gh-pages`。
			    - 这是 GitHub Pages 的默认托管分支。
			- `folder: target/doc`
			    - 要部署的内容目录，这里是 Rust 文档的默认输出目录（`cargo doc` 的结果）。
