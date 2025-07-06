如果不想每次都输入完整的 Git 命令，可以通过 `git config` 文件来轻松地为每一个命令设置一个别名。

```console
$ git config --global alias.co checkout
$ git config --global alias.br branch
$ git config --global alias.ci commit
$ git config --global alias.st status
```

这意味着，当要输入 `git commit` 时，只需要输入 `git ci`。 随着你继续不断地使用 Git，可能也会经常使用其他命令，所以创建别名时不要犹豫。

在创建你认为应该存在的命令时这个技术会很有用。 例如，为了解决取消暂存文件的易用性问题，可以向 Git 中添加你自己的取消暂存别名：
```console
$ git config --global alias.unstage 'reset HEAD --'
```
这会使下面的两个命令等价：
```console
$ git unstage fileA
$ git reset HEAD -- fileA
```

这样看起来更清楚一些。 通常也会添加一个 `last` 命令，像这样：
```console
$ git config --global alias.last 'log -1 HEAD'
```

可以看出，Git 只是简单地将别名替换为对应的命令。 然而，你可能想要执行外部命令，而不是一个 Git 子命令。 如果是那样的话，可以在命令前面加入 `!` 符号。 如果你自己要写一些与 Git 仓库协作的工具的话，那会很有用。 我们现在演示将 `git visual` 定义为 `gitk` 的别名：
```console
$ git config --global alias.visual '!gitk'
```