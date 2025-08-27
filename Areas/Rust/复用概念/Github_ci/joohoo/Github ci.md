
### 介绍
在本视频中，我们从一个没有 CI 且测试很少的 Rust crate 开始 — https://github.com/jonhoo/cargo-index...— 然后我们将其扩展为同时具有两者，从我自己的 CI 配置集合开始：https://github.com/jonhoo/rust-ci-conf。然后，我们继续使用 https://altsysrq.github.io/proptest-b...向 crate 添加基于属性的测试，这显着扩展了相关 crate 的测试

### 添加
在此文件夹中，有 codecoverage、dependabot 和 ci 工作流的配置，这些工作流比默认配置更深入地检查库。

可以使用 [https://github.com/jonhoo/rust-ci-conf/](https://github.com/jonhoo/rust-ci-conf/) 的 --allow-unrelated-histories 合并策略合并此文件夹，该策略为编写您自己的 ci 提供了一个合理合理的基础。通过使用此策略，CI 存储库的历史记录将包含在存储库中，并且以后可以合并 CI 的未来更新。

要执行此合并运行，请执行以下作：

```
git remote add ci https://github.com/jonhoo/rust-ci-conf.git
git fetch ci
git merge --allow-unrelated-histories ci/main
```

该项目中文件的概述可在以下位置获得： [https://www.youtube.com/watch?v=xUH-4y92jPg&t=491s](https://www.youtube.com/watch?v=xUH-4y92jPg&t=491s)，其中包含一些决策的基本原理，并通过解决最小版本和 OpenSSL 问题的示例运行。

- `--allow-unrelated-histories` 允许 Git 建立一个新的 merge commit，把两个历史连接起来。

**使用场景：**
- 将两个独立的仓库合并为一个。
- 迁移项目、引入外部模板或配置仓库。
-  当前分支会生成一个 **merge commit**，内容包含当前分支和 `ci/main` 的所有文件。
- 两个仓库的提交历史都保留。