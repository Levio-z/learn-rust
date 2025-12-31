### 1参考
#### 1.1 挑战内容
- https://app.codecrafters.io/courses/shell/stages/qp2
- https://app.codecrafters.io/courses/shell/stages/gm9
- https://app.codecrafters.io/courses/shell/stages/qm8
- https://app.codecrafters.io/courses/shell/stages/gy5
- https://app.codecrafters.io/courses/shell/stages/wh6
- https://app.codecrafters.io/courses/shell/stages/wt6
#### 1.2 资料
参考：
- https://github.com/cc-code-examples/adventurous-fish-316967/blob/main/src/builtin.rs
- https://github.com/wangzhen0518/codecrafters-shell-rust/commits/main/
#### 1.3 知识点
#### 1.4 需求分析

### 2 方案设计
#### 2.1 核心功能
一开始逻辑之间耦合严重需要维护不同的bool变量来确定在哪个状态，后续重构为状态机实现

### 3 开发
### 任务
- 看视频
- 了解rust如何fork
	- 使用fork写一个案例
- fork的几种形式
- 执行新进程的几种形式
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
- [z-Build yout own shell 重定向标准输出](../4%20note/note/reference/z-Build%20yout%20own%20shell%20重定向标准输出.md)

### 4 测试

### 5 总结

