### 1参考
#### 1.1 挑战内容
#### 1.2 资料
##### web
- https://app.codecrafters.io/courses/shell/stages/ni6
#### 1.3 知识点
#### 1.4 需求分析

### 2 方案设计
#### 2.1 核心功能

### 3 开发
#### 重构
##### 1、Path改为单例，使用lazylock
**运行中的 Shell** 不会自动监控配置文件变化并更新 PATH；要更新，需要重新 source 配置文件：
将shell改为单例模式
```
static GLOBAL_VEC: LazyLock<Vec<PathBuf>> = LazyLock::new(|| {

    let path = std::env::var("PATH").unwrap_or("".to_string());

    std::env::split_paths(&std::ffi::OsStr::new(&path)).collect::<Vec<_>>()

});
```
- 使用&GLOBAL_VEC
##### 2、匹配改为match



### 4 测试

### 5 总结

