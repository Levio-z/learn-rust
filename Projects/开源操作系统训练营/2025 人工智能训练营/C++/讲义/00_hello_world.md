```
std::cout << "Hello, InfiniTensor!" << std::endl;
```
相比printf，流式语法是类型安全的

类型标错的printf：
```
printf("%d\n", 3.14); // %d 表示整数，却对应一个浮点数
```
- 虽然现代编译器能提出警告，但这不是语法错误，真的会把这个地址里的东西当作整数打印出来。

可读性差
```
printf("%d + %d = %d\n", 23, 19, 23 + 19);
```
这一看就是一段完整的话，很舒服。但是流式语法就得写成：
```
std::cout << 23 << " + " << 19 << " = " << 23 + 19 << std::endl;
```

流式格式化显示 bool 变量的格式：
```
std::cout << true << ' '<< false << std::endl;
```
可以看到它居然是当作整数显示的，相当奇怪。必须给它挂上流修饰符 `std::boolalpha` 才行：
```
std::cout << std::boolalpha << true << ' ' << false << std::endl;
```
但是这个流修饰符的行为又很奇怪，它的影响竟然是全局的！比如我写成：
```
{
    std::cout << std::boolalpha << true << ' ' << false << std::endl;
}
{
    std::cout << true << ' ' << false << std::endl;
}
```
可以观察到修饰一次之后，再输出给 std::cout 的所有 bool 全部被转化了。这种一不留神改了全局状态的设计实在是太逆天了。关于标准库中其他的流修饰符大家可以自行查看 [cppreference](https://zh.cppreference.com/w/cpp/io/manip)。虽然我的评价是能不用就算了。

**所以在实践中，我们尽量使用 [{fmt}](https://fmt.dev/) 库完成格式化输出。这个库有可能是 C++ 世界里使用最广泛的库之一了，功能相当强大。我们 RefactorGraph 项目里也是用的这个库。具体这个库的用法大家可以自己看文档学习。**

由于这个库实在是太好用了，以至于它在 C++20 版本进入了标准。由于主流编译器实现这个库普遍比较晚，为了避免版本问题，我们就没有在习题里使用，这里可以展示一下它是怎么用的：
```
#include "../exercise.h"
#include <format>

int main(int argc, char **argv) {
    std::cout << std::format("Hello, {}!", "InfiniTensor") << std::endl
              << std::format("{} + {} = {}", 23, 19, 23 + 19) << std::endl;
    return 0;
}
```
推荐所有有条件选择项目运行环境的同学用这套输出。详细信息见 [cppreference](https://zh.cppreference.com/w/cpp/utility/format/format)。

最后要讲的是一些跟操作系统进程相关的东西，不是 C++ 本身的知识了。现代操作系统的进程设计中，为每个进程提供了 3 个管道，分别是标准输入、标准输出和标准错误。这个设定被映射到 C++ 标准库中，当然 C/Rust 或者其他语言，只要是能生成应用程序的基本都有这个设定。例如：
```
std::cout << "Hello, ";
std::cerr << "InfiniTensor!" << std::endl;
```
就是向输出流发送 `Hello,`，向错误流发送 `InfiniTensor!`。运行这个示例，看起来它们很正常地输出了。但是我们还是可以使用控制台的管道操作符来把标准输出和标准错误分别重定向到文件：
```
xmake run learn 0 > out.txt      # 输出流到文件，错误流到控制台
xmake run learn 0 > out.txt 2>&1 # 输出流和错误流都到文件
xmake run learn 0 2>err.txt      # 输出流到控制台，错误流到文件
```

好的，跟 C++ 输出相关的内容就讲这么多。标准输入流 `std::cin` 在我们的项目里完全用不到，这里就不讲了，需要的同学可以到 [cppreference](https://zh.cppreference.com/w/cpp/io/cin) 自学。