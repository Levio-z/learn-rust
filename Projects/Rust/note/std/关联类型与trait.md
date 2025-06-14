## 原因：**关联类型与 Trait Object**

### 1. 关联类型（associated types）

你的 `GuiFactory` trait 这样定义：

Rust

```
pub trait GuiFactory {
    type B: Button;
    type C: Checkbox;
    fn create_button(&self) -> Self::B;
    fn create_checkbox(&self) -> Self::C;
}
```

> **带有关联类型的 trait，不能作为 trait object (`dyn GuiFactory`) 使用！**

- 原因是：trait object（如 `&dyn GuiFactory`）在运行时并不知道 `Self::B`、`Self::C` 到底是什么类型。
- Rust 需要在编译期知道所有类型信息，而 trait object 的本质是“运行时多态”，而关联类型是“编译期多态”。
- 所以，**任何带未指定关联类型的 trait，都不能作为 trait object 用**，否则你会看到类似：
    
    > the value of the associated types `B` and `C` must be specified
    

### 2. 不带关联类型（只有方法返回 trait object）

`GuiFactoryDynamic` 的方法返回 trait object，不需要关联类型：

Rust

```
pub trait GuiFactoryDynamic {
    fn create_button(&self) -> Box<dyn Button>;
    fn create_checkbox(&self) -> Box<dyn Checkbox>;
}
```

这样，`dyn GuiFactoryDynamic` 就可以作为 trait object 使用，因为所有的方法签名在编译期都是确定的，不需要编译器知道具体的关联类型。

---

## 总结

- **带关联类型的 trait 不能作为 trait object 用**，只能作为泛型参数用（比如 `F: GuiFactory<B=..., C=...>`）。
- **不带关联类型且方法返回 trait object 的 trait，可以作为 trait object 用**（如 `&dyn GuiFactoryDynamic`）。

---

### 图解

|trait|能否用 `dyn Trait`|原因|
|---|---|---|
|有关联类型|❌|关联类型必须编译期确定|
|无关联类型/返回 Box<dyn Trait>|✔️|方法签名已知，运行时可多态|

---

**一句话记忆：**  
带关联类型的 trait 不能直接做 trait object，因为类型没法动态确定！

如需更多代码例子可随时提问！