- 编译：`rustc main.rs`
- 版本切换：
	- `rustup default nightly #stable`
	- 临时：`rustup override set nightly`
- 更新： `rustup update`
	-  `rustup update nightly`(安装最新的 nightly 工具链)
- 本地文档：`rustup doc`
- 检查版本：
	- 查看当前最新 nightly 的版本信息
		- `rustc +nightly --version --verbose`
	- 检查版本：`rustc --version`
	    - 最新稳定版本的版本号、提交哈希值和提交日期
- 卸载：`rustup self uninstall`


