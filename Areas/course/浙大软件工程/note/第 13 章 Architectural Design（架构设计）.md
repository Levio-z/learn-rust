# Ch.13 Architectural Design（架构设计）

## 1. Why Architecture?（为什么需要架构？）

架构并非可运行的软件，而是一种表述方式，能帮助软件工程师实现以下目标：

  

1. 分析设计在满足既定需求方面的有效性；
2. 在设计变更仍相对容易的阶段，考量架构的替代方案；
3. 降低与软件构建相关的风险。

## 2. Why is Architecture Important?（架构为何重要？）

- 软件架构的表述有助于所有与基于计算机的系统开发相关的各方（利益相关者）之间的沟通。
- 架构凸显了早期的设计决策，这些决策将对后续所有软件工程工作产生深远影响，同样重要的是，也会影响系统作为可运行实体的最终成功。
- 正如 [BAS03] 所述，架构 “构成了一种相对简洁、易于理解的模式，用于展示系统的结构以及各组件如何协同工作”。

## 3. Architectural Descriptions（架构描述）

电气和电子工程师协会（IEEE）计算机学会提出了**IEEE-Std-1471-2000 标准**（《软件密集型系统架构描述推荐实践》[IEE00]），该标准的目的包括：

  

- 为软件架构设计过程建立概念框架和术语体系；
- 为架构描述的呈现提供详细指南；
- 推广合理的架构设计实践。

  

根据该 IEEE 标准，**架构描述（AD）** 被定义为 “用于记录架构的一组产物”。其特点如下：

  

- 架构描述本身通过多个 “视图” 来呈现，每个视图是 “从一组相关（利益相关者）关注点的角度对整个系统的一种表述”。

## 4. Architectural Genres（架构类型）

“类型（Genre）” 指的是整个软件领域中的特定类别，每个类别下又包含多个子类别。  
例如，在 “建筑物” 这一类型中，会包含住宅、公寓楼、办公楼、工业建筑、仓库等通用风格；而在每个通用风格下，还可能存在更具体的风格，且每种风格都有可通过一组可预测模式描述的结构。

## 5. Architectural Styles（架构风格）

每种架构风格都描述了一个系统类别，包含以下 4 个核心要素：

  

1. 一组组件（如数据库、计算模块），用于执行系统所需的功能；
2. 一组连接器，用于实现组件之间的 “通信、协调与协作”；
3. 约束条件，定义组件如何集成以构成系统；
4. 语义模型，帮助设计人员通过分析组件的已知属性来理解系统的整体属性。

  

常见的架构风格包括：

  

- 数据中心架构（Data-centered architectures）
- 数据流架构（Data flow architectures）
- 调用返回架构（Call and return architectures）
- 面向对象架构（Object-oriented architectures）
- 分层架构（Layered architectures）

### 5.1 Data-Centered Architecture（数据中心架构）

![](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%27400%27%20height=%27256%27/%3e)![数据中心架构图](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjI1NiIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg==)![](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%27400%27%20height=%27256%27/%3e)![image](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjI1NiIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg==)

  
（图中包含数据存储库 / 黑板（Data store: repository or black board）及多个客户端软件（client software），客户端软件围绕数据存储进行交互）

### 5.2 Data Flow Architecture（数据流架构）

![](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%27400%27%20height=%27256%27/%3e)![数据流架构图](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjI1NiIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg==)![](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%27400%27%20height=%27256%27/%3e)![image](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjI1NiIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg==)

  
包含两种常见形式：

  

- （a）管道 - 过滤器（pipes and filters）：通过 “管道” 连接多个 “过滤器”，数据在过滤器间通过管道传输并处理；
- （b）批处理序列（batch sequential）：数据以批处理的方式按顺序在不同组件间传递处理。

### 5.3 Call and Return Architecture（调用返回架构）

![](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%27400%27%20height=%27256%27/%3e)![调用返回架构图](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjI1NiIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg==)![](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%27400%27%20height=%27256%27/%3e)![image](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjI1NiIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg==)

  
（图中展示了调用的 “扇出（fan-out）”“扇入（fan-in）”、调用深度（depth）和宽度（width）等特性，体现了组件间的调用关系）

### 5.4 Layered Architecture（分层架构）

![](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%27400%27%20height=%27256%27/%3e)![分层架构图](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjI1NiIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg==)![](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%27400%27%20height=%27256%27/%3e)![image](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjI1NiIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg==)

  
（图中展示了不同层级的组件（components），各层级按特定规则交互，上层组件通常依赖下层组件提供的服务）

## 6. Architectural Patterns（架构模式）

架构模式针对系统设计中的特定问题场景提供解决方案，常见模式如下：

### 6.1 并发模式（Concurrency）

适用于需模拟并行方式处理多个任务的应用场景，常见模式：

  

- 操作系统进程管理模式（operating system process management pattern）
- 任务调度器模式（task scheduler pattern）

### 6.2 持久化模式（Persistence）

“持久化” 指数据在创建它的进程执行结束后仍能保留，常见模式：

  

- 数据库管理系统模式：将数据库管理系统（DBMS）的存储和检索能力应用于应用架构；
- 应用级持久化模式：在应用架构中内置持久化功能。

### 6.3 分布式模式（Distribution）

解决分布式环境中系统或系统内组件间的通信方式问题，核心模式：

  

- 中介者模式（Broker）：“中介者” 作为客户端组件与服务器组件之间的 “中间人”，协调两者的通信。

## 7. Architectural Design（架构设计流程）

架构设计需遵循以下关键步骤：

### 7.1 明确软件上下文（Place Software into Context）

设计需定义与软件交互的外部实体（其他系统、设备、人员）以及交互的性质。

### 7.2 识别架构原型（Identify Architectural Archetypes）

“原型（Archetype）” 是一种抽象（类似类），代表系统行为的一个元素。

### 7.3 定义并细化组件结构（Define & Refine Component Structure）

设计人员通过定义和细化实现每个原型的软件组件，明确系统的结构。

#### 7.3.1 Architectural Context（架构上下文）

![](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%27400%27%20height=%27256%27/%3e)![SafeHome架构上下文图](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjI1NiIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg==)![](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%27400%27%20height=%27256%27/%3e)![image](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjI1NiIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg==)

  
（以 SafeHome 安全系统为例，展示了目标系统（target system: SafeHome surveillance Security Function）与外部实体如互联网产品控制面板（Internet-based Product control panel）、房主（homeowner）、传感器（sensors）等的交互关系）

#### 7.3.2 Archetypes（原型示例）

![](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%27400%27%20height=%27256%27/%3e)![SafeHome安全功能原型UML图](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjI1NiIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg==)![](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%27400%27%20height=%27256%27/%3e)![image](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjI1NiIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg==)

  
（图 10.7：SafeHome 安全功能原型的 UML 关系图（改编自 [BOS00]），包含控制器（Controller）、节点（Node）、检测器（Detector）、指示器（Indicator）等原型及其通信关系）

#### 7.3.3 Component Structure（组件结构）

![](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%27400%27%20height=%27256%27/%3e)![组件结构图](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjI1NiIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg==)![](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%27400%27%20height=%27256%27/%3e)![image](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjI1NiIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg==)

#### 7.3.4 Refined Component Structure（细化的组件结构）

![](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%27400%27%20height=%27256%27/%3e)![细化组件结构图](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjI1NiIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg==)![](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%27400%27%20height=%27256%27/%3e)![image](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjI1NiIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg==)

  
（以 SafeHome 系统为例，细化后的组件包括执行器（Executive）、外部通信管理（External Communication Management）、安全控制（Security Control）、互联网接口（Internet Interface）、检测器处理（detector processing）、警报面板管理（alarm panel management）等）

## 8. Architectural Considerations（架构设计需考量的因素）

- **简洁性（Economy）**：优秀的软件架构应简洁明了，依靠抽象减少不必要的细节。
- **可追溯性（Visibility）**：架构决策及其背后的原因，应对后续审查模型的软件工程师清晰可见。
- **关注点分离（Spacing）**：在设计中分离不同关注点，同时避免引入隐藏的依赖关系。
- **一致性（Symmetry）**：架构的对称性意味着系统在属性上是一致且平衡的。
- **自组织性（Emergence）**：系统应具备涌现式、自组织的行为和控制能力。

## 9. Architectural Decision Documentation（架构决策文档）

文档编写需遵循以下原则：

  

1. 确定每个决策所需的信息项；
2. 定义每个决策与相应需求之间的关联；
3. 提供状态变更机制，以支持评估替代决策；
4. 定义决策间的前置依赖关系，确保可追溯性；
5. 将重要决策与决策产生的架构视图关联；
6. 在决策制定过程中及时记录并沟通所有决策。

## 10. Architectural Tradeoff Analysis（架构权衡分析）

分析流程分为 6 个步骤：

  

1. 收集场景（Collect scenarios）；
2. 获取需求、约束条件和环境描述（Elicit requirements, constraints, and environment description）；
3. 描述为满足场景和需求而选择的架构风格 / 模式，包括模块视图（module view）、进程视图（process view）、数据流视图（data flow view）；
4. 单独评估每个质量属性（Evaluate quality attributes in isolation）；
5. 确定特定架构风格下，质量属性对各种架构属性的敏感性（Identify sensitivity of quality attributes to architectural attributes）；
6. 利用步骤 5 的敏感性分析，评估步骤 3 中提出的候选架构（Critique candidate architectures）。

## 11. Architectural Complexity（架构复杂性）

如 [Zha98] 所述，可通过分析架构内组件间的依赖关系来评估拟议架构的整体复杂性，主要依赖类型包括：

  

- **共享依赖（Sharing dependencies）**：使用同一资源的消费者之间，或为同一消费者提供资源的生产者之间的依赖关系；
- **流依赖（Flow dependencies）**：资源的生产者与消费者之间的依赖关系；
- **约束依赖（Constrained dependencies）**：一组活动间控制流相对顺序的约束关系。

## 12. ADL（架构描述语言）

**架构描述语言（Architectural Description Language, ADL）** 为描述软件架构提供了语义和语法支持，能帮助设计人员实现以下操作：

  

- 分解架构组件（decompose architectural components）；
- 将单个组件组合成更大的架构块（compose components into larger architectural blocks）；
- 表示组件间的接口（连接机制）（represent interfaces between components）。

## 13. Architecture Reviews（架构评审）

架构评审的核心目标和方式：

  

- 评估软件架构满足系统质量需求的能力，并识别潜在风险；
- 可在早期发现设计问题，从而降低项目成本；
- 通常采用基于经验的评审、原型评估、场景评审和检查清单等方法。

### 13.1 Pattern-Based Architecture Review（基于模式的架构评审）

评审流程如下：

  

1. 通过梳理用例，识别并讨论质量属性；
2. 结合系统需求，讨论系统架构图；
3. 识别所使用的架构模式，并将系统结构与模式结构进行匹配；
4. 利用现有文档和用例，确定每个模式对质量属性的影响；
5. 识别设计中使用的架构模式所引发的所有质量问题；
6. 总结会议中发现的问题，并修订 “行走骨架（walking skeleton）”（即系统的核心架构原型）。

## 14. Agility and Architecture（敏捷与架构）

在敏捷开发中整合架构设计的方法：

  

- 为避免返工，在编码前利用用户故事（user stories）创建并演进架构模型（“行走骨架”）；
- 采用混合模型，允许软件架构师向不断演进的故事板中贡献用户故事；
- 运行良好的敏捷项目会在每个冲刺（sprint）中交付可工作的产物；
- 对冲刺中产生的代码进行评审，可作为一种有效的架构评审方式。