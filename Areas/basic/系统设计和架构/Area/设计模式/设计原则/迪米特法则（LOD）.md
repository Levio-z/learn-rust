
 >Law of Demeter
 
 >Each unit should have only limited knowledge about other units: only units“closely” related to the current unit. Or: Each unit should only talk to its  friends; Don’t talk to strangers

每个模块（unit）只应该了解那些与它关系密切的模块（units: only units “closely”related to the current unit）的有限知识（knowledge）。或者说，每个模块只和自己
的朋友“说话”（talk），不和陌生人“说话”（talk）。

>不该有直接依赖关系的类之间，不要有依赖；有依赖关系的类之间，尽量只依赖必要的接口（也就是定义中的“有限知识”）。

### 理论解读与代码实战一

“不该有直接依赖关系的类之间，不要有依赖”。
```java
1 public class NetworkTransporter {
2 // 省略属性和其他方法...
3 public Byte[] send(HtmlRequest htmlRequest) {
4 //...
5 }
6 }
7
8 public class HtmlDownloader {
9 private NetworkTransporter transporter;// 通过构造函数或 IOC 注入
10
11 public Html downloadHtml(String url) {
12 Byte[] rawHtml = transporter.send(new HtmlRequest(url));
13 return new Html(rawHtml);
14 }
15 }
16
17 public class Document {
18 private Html html;
19 private String url;
20
21 public Document(String url) {
22 this.url = url;
23 HtmlDownloader downloader = new HtmlDownloader();
24 this.html = downloader.downloadHtml(url);
25 }
26 //...
27 }

```
- 我们来看 NetworkTransporter 类
	- 作为一个底层网络通信类，我们希望它的功能尽可能通用，而不只是服务于下载 HTML
		- 我们应该把 address 和content 交给 NetworkTransporter，而非是直接把 HtmlRequest 交给NetworkTransporter。
		- public Byte[] send(String address, Byte[] data)
- Document
	- 第一，构造函数中的 **downloader.downloadHtml() 逻辑复杂，耗时长，不应该放到构造函数中**，会影响代码的可测试性。
	- 第二，**HtmlDownloader 对象在构造函数中通过 new 来创建，违反了基于接口而非实现编程的设计思想，也会影响到代码的可测试性**。
	- Document 网页文档没必要依赖 HtmlDownloader 类，违背了迪米特法则。
		- 使用依赖注入来解耦，使用构造工厂解耦文档和HtmlDownloader 类
```java
1 public class Document {
2 private Html html;
3 private String url;
4
5 public Document(String url, Html html) {
6 this.html = html;
7 this.url = url;
8 }
9 //...
10 }
11
12 // 通过一个工厂方法来创建 Document
13 public class DocumentFactory {
14 private HtmlDownloader downloader;
15
16 public DocumentFactory(HtmlDownloader downloader) {
17 this.downloader = downloader;
18 }
19
20 public Document createDocument(String url) {
21 Html html = downloader.downloadHtml(url);
22 return new Document(url, html);
23 }
24 }

```

### 理论解读与代码实战二
假设在我们的项目中，有些类只用到了序列化操作，而另一些类只用到反序列化操作。那基于迪米特法则后半部分“有依赖关系的类之间，尽量只依赖必要的接口”，只用到序列化操作的那部分类不应该依赖反序列化接口。同理，只用到反序列化操作的那部分类不应该依赖序列化接口。
尽管拆分之后的代码更能满足迪米特法则，但却违背了高内聚的设计思想。
```java
public interface Serializable {
    String serialize(Object object);
}
public interface Deserializable {
    Object deserialize(String text);
}
public class Serialization implements Serializable, Deserializable {
    @Override
    public String serialize(Object object) {
        String serializedResult = ...;
        return serializedResult;
    }

    @Override
    public Object deserialize(String str) {
        Object deserializedResult = ...;
        return deserializedResult;
    }
}
public class DemoClass_1 {
    private Serializable serializer;

    public DemoClass_1(Serializable serializer) {
        this.serializer = serializer;
    }
    //...
}
public class DemoClass_2 {
    private Deserializable deserializer;

    public DemoClass_2(Deserializable deserializer) {
        this.deserializer = deserializer;
    }
    //...
}


```

✅ 一个 `Serializable` 接口 → 负责序列化（对象 → 字符串）  
✅ 一个 `Deserializable` 接口 → 负责反序列化（字符串 → 对象）  
✅ 一个 `Serialization` 实现类 → 同时实现两者  
✅ 两个 Demo 类：

- `DemoClass_1` 依赖 `Serializable` 接口
    
- `DemoClass_2` 依赖 `Deserializable` 接口

**“基于最小接口而非最大实现编程”。**
- 设计特点：
    
    - 依赖注入（DI）风格，将实现通过构造函数传入。
        
    - 分离了序列化和反序列化关注点。
        
- 原理：
    
    - **接口隔离原则（ISP, Interface Segregation Principle）**：使用最小必要接口，避免依赖更多功能。
        
    - **依赖倒置原则（DIP, Dependency Inversion Principle）**：依赖接口而非具体实现，方便替换和扩展。

### 辩证思考与灵活应用

整个类只包含序列化和反序列化两个操作，只用到序列化操作的使用者，即便能够感知到仅有的一个反序列化函数，问题也不大。

对 Serialization 类添加更多的功能，实现更多更好用的序列化、反序列化函数
```java

public class Serializer {

    // --- 通用序列化方法 ---
    public String serialize(Object object) {
        // 根据对象类型判断调用
        if (object instanceof Map) {
            return serializeMap((Map<?, ?>) object);
        } else if (object instanceof List) {
            return serializeList((List<?>) object);
        } else {
            return serializeObject(object);
        }
    }

    private String serializeObject(Object object) {
        // 序列化普通对象
        // 例如：转成 JSON 字符串
        return "...";
    }

    private String serializeMap(Map<?, ?> map) {
        // 序列化 Map
        return "...";
    }

    private String serializeList(List<?> list) {
        // 序列化 List
        return "...";
    }

    // --- 通用反序列化方法 ---
    public Object deserialize(String string) {
        // 根据字符串特征判断类型（或需要外部指定类型）
        return deserializeObject(string);
    }

    private Map<?, ?> deserializeMap(String mapString) {
        // 反序列化为 Map
        return new HashMap<>();
    }

    private List<?> deserializeList(String listString) {
        // 反序列化为 List
        return new ArrayList<>();
    }

    private Object deserializeObject(String objectString) {
        // 反序列化为普通对象
        return new Object();
    }
}

```

在这种场景下，第二种设计思路要更好些。因为基于之前的应用场景来说，大部分代码只需要用到序列化的功能。对于这部分使用者，没必要了解反序列化的“知识”，而修改之后的Serialization 类，反序列化的“知识”，从一个函数变成了三个。一旦任一反序列化操作
有代码改动，我们都需要检查、测试所有依赖 Serialization 类的代码是否还能正常工作。
为了减少耦合和测试工作量，我们应该按照迪米特法则，将反序列化和序列化的功能隔离开来。