### [在 Linux 或 macOS 上安装 `rustup`](https://rust-book.cs.brown.edu/ch01-01-installation.html#installing-rustup-on-linux-or-macos)
```
$ curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh
```
该命令下载脚本并开始安装 `rustup` 工具，它安装了最新的稳定版本的 Rust。系统可能会提示您 输入您的密码。如果安装成功，将出现以下行：
```
Rust is installed now. Great!
```
您还需要一个链接器_，这是 Rust 用来将其编译后的输出连接到一个文件中的程序。您可能已经拥有一个。如果遇到链接器错误，则应安装 C 编译器，其中通常包含链接器。C 编译器也很有用，因为一些常见的 Rust 包依赖于 C 代码，并且需要 C 编译器。
在 macOS 上，您可以通过运行以下命令来获取 C 编译器：
```
$ xcode-select --install
```
根据其发行版的文档，Linux 用户通常应该安装 GCC 或 Clang。例如，如果您使用 Ubuntu，则可以安装 `build-essential` 包。