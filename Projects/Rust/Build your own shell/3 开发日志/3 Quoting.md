### 1参考
#### 1.1 挑战内容
#### 1.2 资料
##### web
- https://app.codecrafters.io/courses/shell/stages/ei0
#### 1.3 知识点
#### 1.4 需求分析

### 2 方案设计
#### 2.1 核心功能
一开始逻辑之间耦合严重需要维护不同的bool变量来确定在哪个状态，后续重构为状态机实现

### 3 开发
#### 重构
##### 1、Path改为单例，使用lazylock
**运行中的 Shell** 不会自动监控配置文件变化并更新 PATH；要更新，需要重新 source 配置文件：
所将shell改为单例模式，懒加载，静态变量
```
static GLOBAL_VEC: LazyLock<Vec<PathBuf>> = LazyLock::new(|| {

    let path = std::env::var("PATH").unwrap_or("".to_string());

    std::env::split_paths(&std::ffi::OsStr::new(&path)).collect::<Vec<_>>()

});
```
- 使用&GLOBAL_VEC
##### 2、匹配改为match
#### 代码演进
[z-Build yout own shell 单引号（非状态机，按照字符串分割）](../4%20note/note/reference/z-Build%20yout%20own%20shell%20单引号（非状态机，按照字符串分割）.md)
[z-Build yout own shell 双引号（状态机）](../4%20note/note/reference/z-Build%20yout%20own%20shell%20双引号（状态机）.md)
[z-Build yout own shell 双引号支持特殊字符](../4%20note/note/reference/z-Build%20yout%20own%20shell%20双引号支持特殊字符.md)

### 4 测试

### 5 总结

