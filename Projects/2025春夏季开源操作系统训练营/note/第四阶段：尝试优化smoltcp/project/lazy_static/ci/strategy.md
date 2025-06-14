```
jobs:
  ci:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        rust: [stable, nightly]
        os: [ubuntu-latest, windows-latest]
```

- **matrix:**  
    定义一组参数，每个参数所有组合都会单独并行启动一个 job。
    
- **fail-fast:**  
    如果为 `true`（默认），只要有一个 matrix 任务失败，其它还没开始的就会被中止。  
    如果为 `false`，即使有任务失败，其它 matrix 任务仍会全部跑完。

```
matrix:
  rust: [stable, nightly]
  os: [ubuntu-latest, windows-latest]
```
- **rust: [stable, nightly]**
    
    - 表示这个 job 会分别在 `stable` 和 `nightly` 两个 Rust 工具链版本上运行。
    - `stable`：Rust 的稳定版，推荐用于生产。
    - `nightly`：Rust 的每晚构建版，包含最新特性，但可能不稳定。
- **os: [ubuntu-latest, windows-latest]**
    
    - 表示这个 job 会分别在两个操作系统环境运行：
        - `ubuntu-latest`：Ubuntu Linux 最新云主机环境
        - `windows-latest`：Windows 最新云主机环境
- matrix 
	- 会自动**组合所有参数**，即一共会生成 2（rust）× 2（os）= 4 种组合，分别执行 4 次 job：
		1. rust=stable, os=ubuntu-latest
		2. rust=nightly, os=ubuntu-latest
		3. rust=stable, os=windows-latest
		4. rust=nightly, os=windows-latest
每个组合**都是独立并行执行**的，所有步骤会在对应环境下跑一遍。