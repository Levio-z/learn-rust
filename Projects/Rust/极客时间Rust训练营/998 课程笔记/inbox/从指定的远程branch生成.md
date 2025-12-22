在使用 `cargo generate` 从远程模板生成项目时，如果想指定分支，可以使用 `--branch` 参数。命令示例如下：

```bash
cargo generate --git https://github.com/Levio-z/template.git --branch dev
```

解释：

- `--git <url>`：指定远程 Git 仓库 URL。
    
- `--branch <branch>`：指定使用的分支，例如 `dev`。
    
- 也可以加上 `--name <project_name>` 指定生成的项目目录名，例如：
    

```bash
cargo generate --git https://github.com/Levio-z/template.git --branch dev --name my_project
```

其他可选参数：

- `--force`：如果目标目录存在，强制覆盖。
    
- `--silent`：安静模式，不显示模板选择等交互信息。
    
- `--vcs none`：忽略 Git 仓库初始化。
    

✅ 总结：`cargo generate` 默认使用默认分支（通常是 `main` 或 `master`），通过 `--branch <branch>` 可以指定使用其它远程分支。

如果你需要，我可以给你写一个完整的**带模板变量替换、指定分支、生成到指定目录的命令示例**。你希望我写吗？