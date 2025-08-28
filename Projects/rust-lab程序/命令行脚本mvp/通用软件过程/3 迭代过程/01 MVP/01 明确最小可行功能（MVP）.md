### 1. 明确最小可行功能（MVP）
- **目标**：先实现一个功能，即“自动在项目中添加 Git 开源声明”。
- **范围控制**：
    - 只实现对 README 或 LICENSE 文件的插入或更新。
    - 不考虑复杂的模板管理、UI、配置多种开源协议等未来扩展功能。
- 结果：一个**可用、可交付、可测试的最小版本**。

### 2. 分解开发任务

敏捷强调小步迭代，可以把任务拆成最小工作单元：

| 任务     | 描述                  | 交付标准         |
| ------ | ------------------- | ------------ |
| 文件识别   | 检测项目中 README 文件是否存在 | 返回文件路径或创建文件  |
| 声明内容生成 | 根据开源协议生成标准声明文本      | 输出字符串或文件内容   |
| 自动插入   | 将声明写入文件顶部或指定位置      | 文件更新成功       |
| 验证     | 确认文件中声明格式正确         | 可通过单元测试或脚本验证 |

### 3、task
命令设置
- [x] mvp name n
- name 名称
- n 可选参数 n表示没有版本控制文件
- 创建新程序，可选有git或者无git
- 自动添加vscode 配置内容
- 自动添加rustfmt.toml
- 自动添加md
- [x] mvp add md
- 自动添加md
- [x] mvp add lic
- 自动添加开源协议
- [x] mvp add vscode
- 自动添加vscode配置内容

### 复盘
策略模式实现：可以新增一个prelude.rs来放trait里面需要引入的结构
实现trait的策略只需要引入use super::prelude::*;
- 新增策略：复制一个已实现的粘贴
- 策略文件夹下导入，导出实现的策略类
```
mod vscode_strategy;
pub use vscode_strategy::VscodeStrategy;
```

策略工厂需要导入，策略模块下所有内容，trait以及实现的数据结构
- 使用lazylock管理策略
```
    pub fn get_add_strategy_factory() -> &'static AddStrategyFactory {

        static FACTORY: LazyLock<AddStrategyFactory> = LazyLock::new(|| {

            let mut factory = AddStrategyFactory::new();

            // 注册代码

            factory.register(Box::new(InitStrategy));

            factory.register(Box::new(MdStrategy));

            factory.register(Box::new(LicStrategy));

            factory.register(Box::new(VscodeStrategy));

            factory.register(Box::new(FmtStrategy));

            factory

        });

        &FACTORY

    }
```

策略的组合，使用组合模式完成
```
use super::prelude::*;

use crate::add::strategy::{

    fmt_strategy::FmtStrategy, md_strategy::MdStrategy, vscode_strategy::VscodeStrategy,

};

  

pub struct Composite {

    strategies: Vec<Box<dyn AddStrategy>>,

}

  

impl Default for Composite {

    fn default() -> Self {

        Self {

            strategies: vec![

                Box::new(VscodeStrategy),

                Box::new(FmtStrategy),

                Box::new(MdStrategy),

            ],

        }

    }

}

  

impl Composite {

    pub fn handle(&self, tera: &Tera, context: &mut Context) -> Result<(), MvpError> {

        for strat in &self.strategies {

            println!("Running strategy: {}", strat.name());

            strat.handle(tera, context)?;

        }

        Ok(())

    }

}
```
工厂类要导出工厂
整个模块顶层要导出工厂和所有策略实现的内容
```
mod factory;

mod strategy;

  

pub use factory::*;

pub use strategy::*;
```

该模块可以被lib导入
```
pub mod add;

pub mod error;
```