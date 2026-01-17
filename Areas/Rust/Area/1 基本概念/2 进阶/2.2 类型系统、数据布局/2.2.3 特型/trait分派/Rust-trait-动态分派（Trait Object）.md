---
tags:
  - permanent
---
## 1. 核心观点  
### Ⅰ. 概念层

- **核心方式**：使用 `&dyn Trait` 或 `Box<dyn Trait>`（Trait Object）实现动态分派
- **作用**：运行时确定具体调用的 Trait 方法（多态）

```rust
pub trait Formatter {
    fn format(&self, input: &mut String) -> bool;
}
// 接收任意实现了 Formatter 的类型的集合
pub fn format(input: &mut String, formatters: Vec<&dyn Formatter>) {
    for formatter in formatters {
        formatter.format(input);
    }
}
```

### Ⅱ. 应用层




### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 一个简单的案例
```rust
use std::fs::File;
use std::io::Write;

// ====================== Trait 定义 ======================
/// 日志抽象：允许日志器在 log 时修改自身状态
trait Logger {
    fn log(&mut self, message: &str);
}

// ====================== 1. 控制台日志器 ======================
struct ConsoleLogger;

impl Logger for ConsoleLogger {
    fn log(&mut self, message: &str) {
        println!("[Console] {}", message);
    }
}

// ====================== 2. 文件日志器 ======================
struct FileLogger {
    file: File,
}

impl FileLogger {
    fn new(path: &str) -> std::io::Result<Self> {
        Ok(Self {
            file: File::create(path)?,
        })
    }
}

impl Logger for FileLogger {
    fn log(&mut self, message: &str) {
        let _ = writeln!(self.file, "[File] {}", message);
         println!("[file -> {:?}] {}", self.file, message);
    }
}

// ====================== 3. 网络日志器（模拟） ======================
struct NetworkLogger {
    server_addr: String,
}

impl Logger for NetworkLogger {
    fn log(&mut self, message: &str) {
        println!("[Network -> {}] {}", self.server_addr, message);
    }
}

// ====================== 场景 1：&mut dyn Logger ======================
/// 临时组合多个日志器（异构、无所有权）
fn batch_log(loggers: &mut [&mut dyn Logger], message: &str) {
    for logger in loggers {
        logger.log(message);
    }
}

// ====================== 场景 2：Box<dyn Logger> ======================
struct LoggerManager {
    current_logger: Box<dyn Logger>,
}

impl LoggerManager {
    fn new(logger: Box<dyn Logger>) -> Self {
        Self {
            current_logger: logger,
        }
    }

    fn switch_logger(&mut self, new_logger: Box<dyn Logger>) {
        self.current_logger = new_logger;
    }

    fn log(&mut self, message: &str) {
        self.current_logger.log(message);
    }
}

// ====================== main ======================
fn main() -> std::io::Result<()> {
    // ---------- 场景 1 ----------
     println!("场景一");
    let mut console_logger = ConsoleLogger;
    let mut file_logger = FileLogger::new("app.log")?;
    let mut network_logger = NetworkLogger {
        server_addr: "127.0.0.1:8080".into(),
    };

    let mut loggers: Vec<&mut dyn Logger> = vec![
        &mut console_logger,
        &mut file_logger,
        &mut network_logger,
    ];

    batch_log(&mut loggers, "应用启动成功");

    // ---------- 场景 2 ----------
     println!("场景二");
    let mut manager = LoggerManager::new(Box::new(ConsoleLogger));
    manager.log("初始化完成");

    manager.switch_logger(Box::new(FileLogger::new("app.log")?));
    manager.log("处理用户请求");

    manager.switch_logger(Box::new(NetworkLogger {
        server_addr: "127.0.0.1:8080".into(),
    }));
    manager.log("请求处理完成");

    Ok(())
}

```
## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
