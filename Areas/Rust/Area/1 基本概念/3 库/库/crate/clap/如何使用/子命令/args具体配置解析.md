### 1. 参数名与选项

| 配置项                       | 作用                            | 示例                                          |
| ------------------------- | ----------------------------- | ------------------------------------------- |
| `short`                   | 设置短选项，单字符                     | `#[arg(short = 'p')] port: u16` → `-p 8080` |
| `long`                    | 设置长选项，完整名称                    | `#[arg(long)] port: u16` → `--port 8080`    |
| `name`                    | **进阶**：显式指定参数名称（主要用于 help 输出） | `#[arg(name = "PORT")]`                     |
| `visible_alias` / `alias` | 参数别名                          | `#[arg(long, alias = "port-number")]`       |

---

### 2. 参数必填/可选

| 配置项                                 | 作用                 | 示例                                               |
| ----------------------------------- | ------------------ | ------------------------------------------------ |
| `default_value` / `default_value_t` | **进阶**：提供默认值，用户可不传 | `#[arg(long, default_value_t = 8080)] port: u16` |
| `required`                          | **进阶**：强制用户必须提供    | `#[arg(long, required = true)] path: String`     |
| `env`                               | 从环境变量读取            | `#[arg(long, env = "PORT")] port: u16`           |

---

### 3. 布尔参数（flag）

| 配置项                                 | 作用                       | 示例                                                                 |
| ----------------------------------- | ------------------------ | ------------------------------------------------------------------ |
| `action = clap::ArgAction::SetTrue` | 出现该 flag → true，否则 false | `#[arg(long, action = clap::ArgAction::SetTrue)] write: bool`      |
| `action = clap::ArgAction::Count`   | 计数 flag 出现次数             | `#[arg(short, long, action = clap::ArgAction::Count)] verbose: u8` |

- `bool` 类型默认也可以直接作为 flag，不用显式设置 `ArgAction::SetTrue`。

---

### 4. 帮助信息

|配置项|作用|示例|
|---|---|---|
|`help`|简短说明，显示在 `--help`|`#[arg(long, help = "Path to file")] path: String`|
|`long_help`|长说明，显示在详细帮助中|`#[arg(long, long_help = "This option specifies ...")]`|

---

### 5. 值限制

|配置项|作用|示例|
|---|---|---|
|`value_parser`|自定义解析器 / 限制类型|`#[arg(long, value_parser = clap::value_parser!(u32))] port: u32`|
|`value_enum`|枚举类型，自动限制选项||

`#[derive(clap::ValueEnum, Clone)] enum Mode { Fast, Slow } #[arg(long, value_enum)] mode: Mode`

|

---

### 6. vec，多值与可重复

| 配置项                                | 作用          | 示例                                                                            |
| ---------------------------------- | ----------- | ----------------------------------------------------------------------------- |
| 作为位置参数默认是可选vec,加required=true不可选   | 位置参数不填也可以   | `#[arg(num_args = 1.., conflicts_with = "stdin")]`<br> `files: Vec<PathBuf>`, |
| `num_args = n`                     | 固定数量值       | `#[arg(long, num_args = 2)] coords: Vec<i32>`                                 |
| `num_args = 1..`                   | 可变数量值       | `#[arg(long, num_args = 1..)] files: Vec<String>`                             |
| `action = clap::ArgAction::Append` | 每次出现追加到 Vec | `#[arg(long, action = clap::ArgAction::Append)] files: Vec<String>`           |


---

### 7. 子命令相关

| 配置项                      | 作用            | 示例                                                 |
| ------------------------ | ------------- | -------------------------------------------------- |
| `subcommand`             | 标记字段为子命令      | `#[command(subcommand)] command: Option<Commands>` |
| `subcommand_required`    | 强制必须提供子命令     | `#[command(subcommand_required = true)]`           |
| `arg_required_else_help` | 没有子命令或参数时显示帮助 | `#[command(arg_required_else_help = true)]`        |

---

### 8. 参数互斥

- `ArgGroup` 更适合**一组参数互斥或至少提供一个**
- `conflicts_with` 更适合**两个或几个参数直接互斥**
```rust
    HashObject {

        #[arg(short = 'w', long = "write")]

        write: bool,



        #[arg(num_args = 1.., conflicts_with = "stdin")]

        files: Vec<PathBuf>,

  

        /// 从标准输入读取内容（stdin 模式）

        #[arg(long = "stdin", conflicts_with = "files")]

        stdin: bool,

    },
```
- error: the argument '[FILES]...' cannot be used with '--stdin'
- `conflicts_with_all=&["pretty", "type", "size"])`
等价于，在子命令上面加：
```rust
    #[command(group(

        ArgGroup::new("input")

            .required(false)              // 必须提供一个输入方式

            .args(&["files", "stdin"])  // 文件模式或 stdin 模式

    )  // 没提供参数时显示帮助

    )]
```

### 参考
- https://docs.rs/clap/4.5.47/clap/_derive/_tutorial/index.html#subcommands