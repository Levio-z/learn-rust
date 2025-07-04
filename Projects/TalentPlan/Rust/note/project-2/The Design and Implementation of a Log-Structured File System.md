# 摘要

本文提出了一种新的磁盘存储管理技术，称为日志结构文件系统。日志结构文件系统以类似日志的结构顺序地将所有修改写入磁盘，从而**加快文件写入和崩溃恢复**。日志是磁盘上唯一的结构;它包含索引信息，以便可以有效地从日志中读回文件。为了在磁盘上保持大的空闲区域以进行快速写入，我们将日志划分为段，并使用段清理器来压缩碎片严重的段中的实时信息。我们提出了一系列的模拟，证明了一个简单的清洁政策的成本和效益的基础上的效率。我们已经实现了一个原型 logstructured 文件系统称为 Sprite LFS，它优于当前的 Unix 文件系统的一个数量级的小文件写入，同时匹配或超过 Unix 的性能读取和大写。 即使包括清理的开销，Sprite LFS 也可以使用 70%的磁盘带宽进行写入，而 Unix 文件系统通常只能使用 5- 10%。

1.在过去的十年中，CPU 速度急剧增加，而磁盘访问时间只是缓慢地改善。这种趋势在未来可能会继续下去，并将导致越来越多的应用程序成为磁盘绑定的。为了减轻这个问题的影响，我们设计了一种新的磁盘存储管理技术，称为日志结构文件系统，它使用磁盘的顺序是比当前的文件系统更高效。

日志结构文件系统基于这样的假设，即文件被缓存在主存中，并且增加内存大小将使缓存在满足读取请求方面越来越有效[1]。因此，磁盘流量将由写操作主导。日志结构的文件系统将所有新信息以称为日志的顺序结构写入磁盘。这种方法通过消除几乎所有寻道，显著提高了写入性能。日志的顺序特性也允许更快的崩溃恢复：当前的 Unix 文件系统通常必须在崩溃后扫描整个磁盘以恢复一致性，但日志结构的文件系统只需要检查日志的最新部分。

日志记录的概念并不新鲜，许多最近的文件系统已经将日志作为辅助结构来加速写入和崩溃恢复[2，3]。然而，这些其他系统仅将日志用于临时存储;信息的永久存储器是磁盘上的传统随机存取存储结构。相比之下，日志结构的文件系统将数据永久存储在日志中：磁盘上没有其他结构。日志包含索引信息，以便可以以与当前文件系统相当的效率读回文件。

为了使日志结构文件系统有效地运行，它必须确保始终有大量的可用空间用于写入新数据。这是日志结构文件系统设计中最困难的挑战。在本文中，我们提出了一个解决方案的基础上大范围称为段，其中一个段清洁过程不断再生空段通过压缩的实时数据从严重碎片化的段。我们使用模拟器来探索不同的清理策略，并发现了一种基于成本和收益的简单但有效的算法：它将较旧，变化较慢的数据与快速变化的年轻数据隔离开来，并在清理过程中以不同的方式对待它们。

我们已经构建了一个名为 Sprite LFS 的原型日志结构文件系统，它现在作为 Sprite 网络操作系统的一部分在生产中使用[4]。基准程序表明，Sprite LFS 的原始写入速度比 Unix 的小文件写入速度高出一个数量级以上。即使对于其他工作负载，与包括读取和大文件访问的情况一样，Sprite LFS 在所有情况下都至少与 Unix 一样快，但有一种情况除外（文件在随机写入后按顺序读取）?。我们还测量了生产系统中清洁的长期开销。总体而言，Sprite LFS 允许磁盘原始带宽的 65-75%用于写入新数据（其余用于清理）。**相比之下，Unix 系统只能利用磁盘原始带宽的 5-10%来写入新数据;其余的时间都花在搜索上。**

本文其余部分分为六个部分。第 2 节回顾了 90 年代计算机文件系统设计中的问题。第 3 节讨论了日志结构文件系统的设计方案，并推导出 Sprite LFS 的结构，特别关注清理机制。第 4 节描述了 Sprite LFS 的崩溃恢复系统。第 5 节使用基准程序和长期的清洁开销测量来评估 Sprite LFS。第 6 节将 Sprite LFS 与其他文件系统进行了比较，第 7 节结束。

20 世纪 90 年代的文件系统设计文件系统设计受两种力量的支配：技术，它提供了一组基本的构建块，以及工作负载，它决定了一组必须有效执行的操作。本节总结了正在进行的技术更改，并描述了它们对文件系统设计的影响。它还描述了影响 Sprite LFS 设计的工作负载，并展示了当前文件系统如何不足以处理工作负载和技术变化。

2.1.技术有三个技术组件对文件系统设计特别重要：处理器、磁盘和主存。处理器之所以重要，是因为它们的速度几乎是以指数级的速度增长的，而且这种改进似乎很可能在 20 世纪 90 年代的大部分时间里继续下去。这给计算机系统的所有其他元素施加了压力，要求它们也加快速度，这样系统就不会变得不平衡。

磁盘技术也在迅速改进，但改进主要是在成本和容量方面，而不是在性能方面。磁盘性能有两个组成部分：传输带宽和访问时间。虽然这两个因素都在改善，但改善的速度比 CPU 速度慢得多。磁盘传输带宽可以通过使用磁盘阵列和平行磁头磁盘来大幅提高[5]，但访问时间似乎不太可能有大的改善（这是由难以改善的机械运动决定的）。如果一个应用程序导致一系列由寻道分隔的小磁盘传输，那么该应用程序在未来十年内不太可能经历太多的加速，即使使用更快的处理器。

技术的第三个组成部分是主存储器，它的大小正在以指数速度增长。现代文件系统将最近使用的文件数据缓存在主存储器中，并且更大的主存储器使得更大的文件高速缓存成为可能。这对文件系统行为有两个影响。首先，更大的文件缓存通过吸收更大比例的读请求来改变呈现给磁盘的工作负载[1，6]。为了安全起见，大多数写请求最终必须反映在磁盘上，因此磁盘流量（和磁盘性能）将越来越多地由写控制。

大型文件缓存的第二个影响是它们可以作为写缓冲区，在将任何修改的块写入磁盘之前可以收集大量修改的块。缓冲可以使更有效地写入块成为可能，例如通过在仅具有一次寻道的单个顺序传送中将它们全部写入。当然，写缓冲的缺点是会增加崩溃时丢失的数据量。在本文中，我们将假设崩溃是罕见的，并且在每次崩溃中损失几秒钟或几分钟的工作是可以接受的;对于需要更好的崩溃恢复的应用程序，可以使用非易失性 RAM 作为写缓冲区。

2.2.工作负载在计算机应用程序中常见几种不同的文件系统工作负载。文件系统设计最难有效处理的工作负载之一是在办公室和工程环境中。 办公室和工程应用程序往往以访问小文件为主;一些研究测量的平均文件大小只有几个字节[1，6-8]。小文件通常会导致小的随机磁盘 I/O，并且此类文件的创建和删除时间通常由文件系统“元数据”（用于定位文件的属性和块的数据结构）的更新控制。

以顺序访问大文件为主的工作负载（例如超级计算环境中的工作负载）也会带来有趣的问题，但对于文件系统软件来说却不然。有许多技术可以确保这些文件在磁盘上按顺序排列，因此 I/O 性能往往受到 I/O 和内存子系统带宽的限制，而不是文件分配策略的限制。在设计日志结构的文件系统时，我们决定关注小文件访问的效率，并将其留给硬件设计人员来提高大文件访问的带宽。幸运的是，Sprite LFS 中使用的技术适用于大文件和小文件。

2.3.现有文件系统的问题当前的文件系统存在两个普遍问题，这使得它们难以科普 20 世纪 90 年代的技术和工作负载。首先，它们在磁盘上传播信息的方式会导致太多的小访问。例如，Berkeley Unix 快速文件系统（Unix FFS）[9]在将每个文件按顺序放置在磁盘上时非常有效，但它在物理上分隔了不同的文件。此外，文件的属性（“inode”）与文件的内容是分开的，包含文件名的目录条目也是如此。在中**创建一个新文件至少需要五个单独的磁盘 I/O，每个磁盘 I/O 之前都有一个寻道**

Unix FFS：对文件属性的两种不同访问加上对文件数据、目录数据和目录属性的各一种访问。在这样的系统中写入小文件时，只有不到 5%的磁盘潜在带宽用于新数据;其余时间用于查找。

当前文件系统的第二个问题是它们倾向于同步写入：**应用程序必须等待写入完成，而不是在后台处理写入时继续。**例如，即使 Unix FFS 异步写入文件数据块，文件系统元数据结构（如目录和 inode）也是同步写入的。对于包含许多小文件的工作负载，磁盘流量主要由同步元数据写入控制。同步写入将应用程序的性能与磁盘的性能结合起来，使应用程序难以从更快的 CPU 中获益。它们还阻止了将文件缓存用作写缓冲区的可能性。不幸的是，像 NFS 这样的网络文件系统[10]在以前不存在的地方引入了额外的同步行为。这简化了崩溃恢复，但降低了写入性能。

在本文中，我们使用伯克利 Unix 快速文件系统（Unix FFS）作为当前文件系统设计的一个例子，并将其与日志结构文件系统进行比较。使用 Unix FFS 设计是因为它在文献中有很好的记载，并在几个流行的 Unix 操作系统中使用。本节中出现的问题并不是 Unix FFS 独有的，在大多数其他文件系统中都可以找到。

3.日志结构文件系统日志结构文件系统的基本思想是通过在文件缓存中缓冲一系列文件系统更改，然后在单个磁盘写入操作中将所有更改顺序写入磁盘，从而提高写入性能。在写操作中写入磁盘的信息包括文件数据块、属性、索引块目录，以及几乎所有用于管理文件系统的其他信息。对于包含许多小文件的工作负载，日志结构文件系统将传统文件系统的许多小的**同步随机写入转换为可以利用几乎 100%的原始磁盘带宽的大型异步顺序传输。**

虽然日志结构文件系统的基本思想很简单，但是要实现日志方法的潜在好处，必须解决两个关键问题。第一个问题是**如何从日志中检索信息**;这是下面 3.1 节的主题。第二个问题是**如何管理磁盘上的可用空间，以便始终有大量的可用空间可用于写入新数据**。这是一个困难得多的问题;这是 3.2-3.6 节的主题。表 1 总结了 Sprite LFS 用于解决上述问题的磁盘数据结构;本文后面的部分将详细讨论这些数据结构。

3.1.文件位置和阅读尽管术语“日志结构化”可能意味着需要顺序扫描才能从日志中检索信息，但在 Sprite LFS 中并非如此。我们的目标是匹配或超过 Unix FFS 的读取性能。为了实现这一目标，Sprite LFS 在日志中输出索引结构，以允许随机访问检索。Sprite LFS 使用的基本结构与 Unix FFS 中使用的结构相同：对于每个文件，都存在一个称为 inode 的数据结构，其中包含文件的属性（类型，所有者，权限等）。加上文件的前十个块的磁盘地址;对于大于十个块的文件，inode 还包含一个或多个间接块的磁盘地址，每个间接块包含更多数据或间接块的地址。找到文件的 inode 后，读取文件所需的磁盘 I/O 数量在 Sprite LFS 和 Unix FFS 中是相同的。

在 Unix FFS 中，每个 inode 都位于磁盘上的固定位置;给定文件的标识号，
![](Pasted%20image%2020250525230217.png)
对于每个数据结构，该表指示该数据结构在 Sprite LFS 中的用途。该表还指出了数据结构是存储在日志中还是存储在磁盘上的固定位置，以及在本文中详细讨论了数据结构的位置。Inode、间接块和超级块类似于 Unix FFS 数据结构，具有相同的名称。注意，Sprite LFS 既不包含位图也不包含空闲列表。
![](Pasted%20image%2020250525230224.png)
生成文件的 inode 的磁盘地址。相反，Sprite LFS 不会将 inode 放置在固定位置;它们被写入日志。Sprite LFS 使用称为 inode map 的数据结构来维护每个 inode 的当前位置。给定文件的标识号，必须对 inode 映射进行索引，以确定 inode 的磁盘地址。**inode 映射被划分为写入日志的块;每个磁盘上的固定检查点区域标识所有 inode 映射块的位置。幸运的是，inode 映射足够紧凑，可以将活动部分缓存在主存中：inode 映射查找很少需要磁盘访问。**

图 1 显示了在不同目录中创建两个新文件后，Sprite LFS 和 Unix FFS 中的磁盘布局。虽然这两种布局具有相同的逻辑结构，但日志结构的文件系统会产生更紧凑的布局。因此，Sprite LFS 的写性能比 Unix FFS 好得多，而它的读性能也一样好。

![](Pasted%20image%2020250525230236.png)

3.2.自由空间管理：段日志结构文件系统最困难的设计问题是自由空间的管理。其目标是为写入新数据保持较大的空闲区。最初，所有可用空间都在磁盘上的单个区段中，但当日志到达磁盘的末尾时，可用空间将被分割成许多与被删除或覆盖的文件相对应的小区段。

从现在开始，文件系统有两种选择：线程和复制。如图 2 所示。第一种方法是将活动数据保留在原位，并将日志线程化到空闲区。不幸的是，线程将导致空闲空间变得严重碎片化，因此不可能进行大的连续写入，并且日志结构的文件系统将不会比传统的文件系统。第二种方法是将实时数据从日志中复制出来，以便留出较大的空闲区用于写入。在本文中，我们将假设实时数据以压缩形式写回日志的头部;它也可以移动到另一个日志结构的文件系统以形成日志的层次结构，或者它可以移动到一些完全不同的文件系统或存档。复制的缺点是它的成本，特别是对于长期文件;在最简单的情况下，日志在磁盘上循环工作，实时数据被复制回日志，所有长期文件都必须在日志在磁盘上的每次传递中复制。

Sprite LFS 使用线程和复制的组合。**磁盘被划分为称为段的大型固定大小区。任何给定的段总是从开始到结束顺序写入，并且在重写段之前，必须将所有活动数据从段中复制出来。**但是，日志是逐段线程化的;如果系统可以将长期存在的数据收集到段中，则可以跳过这些段，这样就不必重复复制数据。段的大小要选择得足够大，以使读或写整个段的传输时间远远大于查找段开头的开销。这允许整个段操作几乎以磁盘的全部带宽运行，而不管段的访问顺序如何。Sprite LFS 当前使用的段大小为 512 MB 或 1 MB。

3.3.**段清理机制将实时数据从段中复制出来的过程称为段清理。**在 Sprite LFS 中，这是一个简单的三步过程：将一些段读取到内存中，识别活动数据，并将活动数据写回少量干净的段。在此之后操作完成后，已读取的段将标记为干净，它们可用于新数据或进行额外的清理。

作为段清理的一部分，必须能够识别每个段的哪些块是活动的，以便可以再次写出它们。它还必须能够识别每个块所属的文件以及块在文件中的位置;为了更新文件的 inode 以指向块的新位置，需要此信息。Sprite LFS 通过将段摘要块作为每个段的一部分写入来解决这两个问题。摘要块标识写入段中的每条信息;例如，对于每个文件数据块，摘要块包含该块的文件号和块号。当需要多个日志写入来填充段时，段可以包含多个段摘要块。 （当文件缓存中缓冲的脏数据块数量不足以填充数据段时，会发生部分数据段写入。）段摘要块在写入过程中几乎不增加开销，并且在崩溃恢复（参见第 4 节）和清理过程中非常有用。

Sprite LFS 还使用段摘要信息来区分活动块和已被覆盖或删除的块。一旦知道了块的标识，就可以通过检查文件的 inode 或间接块来确定它的活动性，以查看适当的块指针是否仍然引用该块。如果是，那么该块是活的;如果不是，那么该块是死的。Sprite LFS 通过在每个文件的 inode 映射条目中保留一个版本号来稍微优化这种检查;每当文件被删除或截断为零长度时，版本号就会递增。版本号与 inode 号一起构成文件内容的唯一标识符（uid）。段摘要块为段;如果在清理段时块的 UID 与当前存储在 inode 映射中的 UID 不匹配，则可以立即丢弃该块而不检查文件的 inode。

这种清理方法意味着 Sprite 中没有空闲块列表或位图。除了节省内存和磁盘空间外，消除这些数据结构还简化了崩溃恢复。如果这些数据结构存在，则需要额外的代码来记录对结构的更改并在崩溃后恢复一致性。

3.4.鉴于上述基本机制，必须解决四个策略问题：

- 什么时候应该执行段清理器？一些可能的选择是让它以低优先级在后台连续运行，或者只在晚上运行，或者只在磁盘空间几乎耗尽时运行。
- 它一次应该清理多少段？段清理提供了重新组织磁盘上数据的机会;一次清理的段越多，重新排列的机会就越多。
- 应该清理哪些部分？一个明显的选择是那些最分散的，但这并不是最好的选择。
- 在写出活动块时，应如何对其进行分组？一种可能性是尝试增强未来读取的局部性，例如，将同一目录中的文件分组到单个输出段中。另一种可能性是按最后一次修改的时间对块进行排序，并将年龄相似的块分组到新的段中;我们称这种方法为年龄排序。

![image.png](attachment:7b499886-e98e-4caa-b761-bf1399daf8d8:image.png)

图 2 -日志结构文件系统的可用空间管理解决方案。

在日志结构的文件系统中，**日志的空闲空间可以通过复制旧块或围绕旧块线程化日志来生成。**图的左侧显示了线程日志方法，其中日志跳过活动块并覆盖已删除或覆盖的文件块。维护日志块之间的指针，以便在崩溃恢复期间可以跟踪日志。该图的右侧显示了复制方案，其中通过在日志结束后阅读磁盘部分并将该部分的活动块沿着新数据重写到新生成的空间中来生成日志空间。

在我们迄今为止的工作中，我们还没有系统地处理上述政策中的前两个。当干净段的数量低于阈值（通常为几十个段）时，Sprite LFS 开始清理段。它一次清理几十个段，直到清理段的数量超过另一个阈值（通常为 50-100 个清理段）。Sprite LFS 的整体性能似乎对阈值的确切选择并不十分敏感。相比之下，第三和第四个策略决策是至关重要的：根据我们的经验，它们是决定日志结构文件系统性能的主要因素。第 3 节的其余部分讨论了我们对清理哪些段以及如何对实时数据进行分组的分析。

我们使用一个称为写成本的术语来比较清理策略。写入成本是指写入每个字节的新数据时，磁盘处于忙碌状态的平均时间，包括所有清理开销。写入成本表示为如果没有清理开销并且数据可以以其全带宽写入而没有寻道时间或旋转延迟时所需时间的倍数。写入成本为 1.0 是完美的：这意味着新数据可以在整个磁盘带宽上写入，并且没有清理开销。写入成本为 10 意味着只有十分之一的磁盘最大带宽实际用于写入新数据;其余的磁盘时间用于寻道、旋转延迟或清理。

对于具有大段的日志结构文件系统，寻道和旋转延迟对于写入和清理都可以忽略不计，因此写入成本是移动到磁盘和从磁盘移动的总字节数除以表示新数据的字节数。此成本由被清理的段中的利用率（仍然存在的数据的部分）确定。在稳定状态下，清理器必须为写入的每个新数据段生成一个干净的段。为了做到这一点，它读取 N 个完整的段，并写出 N_u 个实时数据段（其中 u 是段的利用率，0 ≤ u < 1）。这将为新数据创建 N_（1-u）个连续可用空间段。因此

图 3 用图形表示写成本与 u 的函数关系。作为参考，小文件工作负载上的 Unix FFS 最多使用 5-10%的磁盘带宽，写入成本为 10-20（具体测量请参见[11]和 5.1 节中的图 8）。通过日志记录，延迟写入和磁盘请求排序，这可能会提高到大约 25%的带宽[12]或写入成本 4。图 3 表明，为了使日志结构文件系统的性能优于当前的 Unix FFS，清理的段的利用率必须小于 0.8;为了优于改进的 Unix FFS，利用率必须小于 0.5。需要注意的是，上面讨论的利用率并不是包含活动数据的磁盘的总体比例;它只是被清理的段中活动块的比例。 文件使用的变化会导致某些段的利用率低于其他段，清理器可以选择利用率最低的段进行清理;这些段的利用率将低于磁盘的总体平均值。

即使如此，日志结构文件系统的性能也可以通过降低磁盘空间的总体利用率来提高。使用的磁盘越少，被清理的段将具有越少的活动块，从而降低写入成本。日志结构文件系统提供了一个成本-性能权衡：如果磁盘空间未得到充分利用，可以实现更高的性能，但每可用字节的成本很高;如果磁盘容量利用率提高，存储成本降低，但性能也降低。这样的权衡
![](Pasted%20image%2020250525230252.png)

在上面的公式中，我们做了一个保守的假设，即必须完整地读取一个段才能恢复活动块;在实践中，只读取活动块可能会更快，特别是在利用率非常低的情况下（我们还没有在 Sprite LFS 中尝试过）。如果要清理的段没有活动块（u = 0），则根本不需要读取它，写入成本为 1.0。

图 3 用图形表示写成本与 u 的函数关系。作为参考，小文件工作负载上的 Unix FFS 最多使用 5-10%的磁盘带宽，写入成本为 10-20（具体测量请参见[11]和 5.1 节中的图 8）。通过日志记录，延迟写入和磁盘请求排序，这可能会提高到大约 25%的带宽[12]或写入成本 4。图 3 表明，为了使日志结构文件系统的性能优于当前的 Unix FFS，清理的段的利用率必须小于 0.8;为了优于改进的 Unix FFS，利用率必须小于 0.5。需要注意的是，上面讨论的利用率并不是包含活动数据的磁盘的总体比例;它只是被清理的段中活动块的比例。 文件使用的变化会导致某些段的利用率低于其他段，清理器可以选择利用率最低的段进行清理;这些段的利用率将低于磁盘的总体平均值。

即使如此，日志结构文件系统的性能也可以通过降低磁盘空间的总体利用率来提高。使用的磁盘越少，被清理的段将具有越少的活动块，从而降低写入成本。日志结构文件系统提供了一个成本-性能权衡：如果磁盘空间未得到充分利用，可以实现更高的性能，但每可用字节的成本很高;如果磁盘容量利用率提高，存储成本降低，但性能也降低。这样的权衡

![](Pasted%20image%2020250525230312.png)

图 3 -小文件的写入成本与 u 的函数关系。

在日志结构的文件系统中，写入成本很大程度上取决于被清理的段的利用率。段中清理的活动数据越多，清理所需的磁盘带宽就越多，并且无法用于写入新数据。该图还显示了两个参考点：“FFS today”，代表今天的 Unix FFS，以及“FFS improved”，这是我们对改进的 Unix FFS 的最佳性能的估计。Unix FFS 的写入成本对使用的磁盘空间量不敏感。

性能和空间利用率之间的差异不是日志结构文件系统所独有的。例如，Unix FFS 只允许 90%的磁盘空间被文件占用。剩余的 10%保持空闲，以允许空间分配算法有效地操作。

在日志结构文件系统中以低成本实现高性能的关键是强制磁盘进入双峰段分布，其中大多数段几乎已满，少数段为空或几乎为空，并且清理器几乎总是可以使用空段。这允许高的整体磁盘容量利用率，但提供低的写入成本。以下部分描述了我们如何在 Sprite LFS 中实现这种双峰分布。

3.5.模拟结果我们构建了一个简单的文件系统模拟器，以便在受控条件下分析不同的清理策略。模拟器的模型并不反映实际的文件系统使用模式（它的模型比现实更严格），但它帮助我们理解随机访问模式和局部性的影响，这两者都可以用来降低清理成本。模拟器将文件系统建模为固定数量的 4 kbyte 文件，选择数量以产生特定的整体磁盘容量利用率。在每一步中，模拟器使用两种伪随机访问模式之一来用新数据覆盖其中一个文件：

均匀：每个文件在每个步骤中被选择的可能性相等。

冷热：文件分为两组。一个组包含 10%的文件;它被称为热，因为它的文件在 90%的时间内被选中。另一组被称为 cold;它包含 90%的文件，但只有 10%的时间被选中。在组内，每个文件被选中的可能性相等。这种访问模式模拟了一种简单形式的局部性。

在这种方法中，整体磁盘容量利用率是恒定的，并且没有对读取流量进行建模。模拟器运行，直到所有干净的段被耗尽，然后模拟清洁器的动作，直到阈值数量的干净的段再次可用。在每次运行中，允许模拟器运行，直到写入成本稳定并且所有冷启动差异已经被移除。

图 4 将两组模拟的结果叠加到图 3 的曲线上。在“LFS 均匀”模拟中，使用了均匀访问模式。清洁工使用了一种简单的贪婪策略，总是选择利用率最低的路段进行清洁。当写出活动数据时，清理器没有尝试重新组织数据：活动块以它们在被清理的段中出现的相同顺序写出（对于统一的访问模式，没有理由期望从重新组织中得到任何改进）。

即使使用统一的随机存取模式，段利用率的变化也允许比从整体磁盘容量利用率和公式（1）预测的写成本低得多的写成本。例如，在总体磁盘容量利用率为 75%时，清理的段的平均利用率仅为 55%。在整体磁盘容量利用率低于 20%时，写入成本降至 2.0 以下;这意味着一些清理的段根本没有活动块，因此不需要读入。

“LFS 热-冷”曲线示出了当访问模式中存在局部性时的写入成本，如上所述。此曲线的清理策略与“LFS uniform”相同，只是在再次写出之前，活动块按年龄排序。这意味着长期（冷）数据往往会与短期（热）数据隔离在不同的段中;我们认为这种方法将导致段利用率的理想双峰分布。

图 4 显示了令人惊讶的结果，局部性和“更好”的分组导致比没有局部性的系统更差的性能！我们尝试改变局部性的程度（例如，95%的访问对应 5%的数据），发现随着局部性的增加，性能变得越来越差。图 5 显示了这种非直观结果的原因。在贪婪策略下，一个段直到它成为所有段中利用率最低的段才被清理。因此，每个段的利用率最终下降到清理阈值，包括冷段。可惜
![](Pasted%20image%2020250525230324.png)

图 5 -使用 greedy cleaner 的段利用率分布。

这些图显示了模拟期间磁盘的段利用率分布。通过测量磁盘上所有段的利用率来计算该分布，这些段在模拟过程中启动段清理时的点处。该分布示出了可用于清洁算法的段的利用率。每个发行版的总体磁盘容量利用率均为 75%。“均匀"曲线对应于图 4 中的”LFS 均匀“，”冷热“对应于图 4 中的”LFS 冷热“。局部性导致分布更偏向于发生清理时的利用率;因此，段以更高的平均利用率进行清理。

在冷段中，利用率下降非常缓慢，因此这些段往往在清洁点上方停留很长时间。图 5 显示，在具有局部性的模拟中，比在没有局部性的模拟中，更多的段聚集在清洁点周围。总的结果是，冷段倾向于长时间占用大量空闲块。

在研究了这些数据后，我们意识到清洁工必须区别对待热段和冷段。冷段中的空闲空间比热段中的空闲空间更有价值，因为一旦冷段被清理，它将花费很长时间才重新积累不可用的空闲空间。换句话说，一旦系统从具有冷数据的段中回收空闲块，它将在冷数据变得碎片化并“再次将它们收回”之前“保留”它们很长一段时间。相比之下，清理一个热段的好处就不那么多了，因为数据可能会很快消失，空闲空间也会很快重新积累;系统可能会延迟清理一段时间，让更多的块在当前段中消失。段的可用空间的值取决于段中数据的稳定性。不幸的是，如果不知道未来的访问模式，就无法预测稳定性

使用一个假设，即一个段中的数据越老，它可能保持不变的时间越长，稳定性可以通过数据的年龄来估计。

为了测试这个理论，我们模拟了一个新的策略来选择要清理的段。该策略根据清洁段的收益和清洁段的成本对每个段进行评级，并选择收益成本比最高的段。好处有两个组成部分：将被回收的可用空间量和空间可能保持空闲的时间量。空闲空间的数量是 1−u，其中 u 是段的利用率。我们使用段中任何块的最近修改时间（即。最年轻的块的年龄）作为空间可能保持空闲的时间的估计。清洁的好处就是这两个分量相乘形成的时空积。清除段的成本是 1+u（读段的成本单位，写回实时数据的成本单位）。综合所有这些因素，我们得到

![](Pasted%20image%2020250525230337.png)

我们将此策略称为成本效益策略;它允许以比热段高得多的利用率来清理冷段。

我们重新运行了模拟下的冷热访问模式与成本效益的政策和年龄排序
![](Pasted%20image%2020250525230346.png)
![](Pasted%20image%2020250525230408.png)



此图显示了在 75%的总体磁盘容量利用率下模拟冷热访问模式时的段利用率分布。“LFS 成本-收益”曲线显示了当使用成本-收益策略来选择要清理的段并在重写之前按年龄分组的活动块时发生的段分布。由于这种双峰段分布，大多数清洁段的利用率约为 15%。为了进行比较，贪婪方法选择策略产生的分布由图 5 中复制的“LFS Greedy”曲线显示。

实时数据。从图 6 中可以看出，成本效益政策产生了我们所希望的双峰分布。清理策略在大约 75%的利用率时清理冷段，但在清理热段之前要等到热段达到大约 15%的利用率。由于 90%的写操作都是针对热文件的，因此大多数被清理的数据段都是热的。图 7 显示，costbenefit 策略比 greedy 策略减少了多达 50%的写入成本，并且即使在相对较高的磁盘容量利用率下，日志结构文件系统的性能也超过了最佳的 Unix FFS。我们模拟了一些其他程度和种类的地方，发现成本效益政策变得更好的地方增加。

模拟实验说服我们在 Sprite LFS 中实施成本效益方法。从 5.2 节中可以看出，Sprite LFS 中实际文件系统的行为甚至比图 7 中预测的要好。

3.6.为了支持成本效益清理策略，Sprite LFS 维护了一个称为段使用表的数据结构。对于每个段，该表记录段中的活动字节数和段中任何块的最近修改时间。这两个值由段清理器在选择要清理的段时使用。这些值最初在写入段时设置，当删除文件或覆盖块时，活动字节的计数会递减。如果计数福尔斯下降到零，则可以在不清洁的情况下重新使用该段。段使用表的块被写入日志，块的地址存储在检查点区域（详见第 4 节）。

为了按年龄对活动块进行排序，段摘要信息记录写入段的最年轻块的年龄。目前，Sprite LFS 并不为文件中的每个块保留修改时间;它为整个文件保留单个修改时间。对于未完全修改的文件，此估计值将不正确。我们计划修改段摘要信息，以包括每个块的修改时间。

4.崩溃恢复当系统崩溃发生时，在磁盘上执行的最后几次操作可能使其处于不一致的状态（例如，可能写入了新文件，但没有写入包含该文件的目录）;在重新引导期间，操作系统必须检查这些操作，以纠正任何不一致。在没有日志的传统 Unix 文件系统中，系统无法确定最后一次更改的位置，因此必须扫描磁盘上的所有元数据结构以恢复一致性。这些扫描的成本已经很高了（在典型配置中需要几十分钟），而且随着存储系统的扩展，成本会越来越高。

在日志结构的文件系统中，最后一次磁盘操作的位置很容易确定：它们位于日志的末尾。因此，应该可以在崩溃后很快恢复。日志的这种好处是众所周知的，并且已经在数据库系统[13]和其他文件系统[2，3，14]中被用于优势。与许多其他日志记录系统一样，Sprite LFS 使用双管齐下的方法进行恢复：检查点（定义文件系统的一致状态）和前滚（用于恢复自上次检查点以来写入的信息）。

4.1.检查点检查点是日志中所有文件系统结构一致且完整的位置。Sprite LFS 使用两阶段流程来创建检查点。首先，它将所有修改的信息写入日志，包括文件数据块、间接块、inode 以及 inode 映射和段使用表的块。其次，它将检查点区域写入磁盘上的特定固定位置。检查点区域包含 inode 映射和段使用表中所有块的地址，加上当前时间和指向最后写入的段的指针。

在重新引导期间，Sprite LFS 读取检查点区域，并使用该信息初始化其主存数据结构。为了处理检查点操作期间的崩溃，实际上存在两个检查点区域，并且检查点操作在它们之间交替。检查点时间位于检查点区域的最后一个块中，因此如果检查点失败，时间将不会更新。在重新启动期间，系统将读取两个检查点区域，并使用时间最近的一个。

Sprite LFS 会定期执行检查点，并且在卸载文件系统或系统

![](Pasted%20image%2020250525230420.png)

图 7 -写入成本，包括成本效益策

此图比较了热-冷访问模式下贪婪策略与成本效益策略的写入成本。成本效益策略明显优于贪婪策略，特别是对于磁盘容量利用率超过 60%的情况。

被关闭了检查点之间的间隔较长可以减少写入检查点的开销，但会增加恢复期间前滚所需的时间;检查点间隔较短可以缩短恢复时间，但会增加正常操作的成本。Sprite LFS 当前使用 30 秒的检查点间隔，这可能太短了。定期检查点的替代方法是在将给定数量的新数据写入日志后执行检查点;这将限制恢复时间，同时减少文件系统未以最大吞吐量运行时的检查点开销。

4.2.前滚原则上，在崩溃后重新启动是安全的，只需阅读最新的检查点区域并丢弃该检查点之后日志中的任何数据。这将导致即时恢复，但自最后一个检查点以来写入的任何数据都将丢失。为了恢复尽可能多的信息，Sprite LFS 会扫描在最后一个检查点之后写入的日志段。

# 4.2 前滚

在前滚期间，Sprite LFS 使用段摘要块中的信息来恢复最近写入的文件数据。当摘要块指示存在新的 inode 时，Sprite LFS 会更新它从检查点读取的 inode 映射，以便 inode 映射引用 inode 的新副本。这会自动将文件的新数据块合并到恢复的文件系统中。如果在没有文件 inode 新副本的情况下发现文件的数据块，则前滚代码假定磁盘上文件的新版本不完整，并忽略新数据块。

前滚代码还调整从检查点读取的段使用表中的利用率。自检查点以来写入的段的利用率将为零;必须调整它们以反映前滚后留下的活动数据。旧段的利用率也必须进行调整，以反映文件删除和覆盖（这两种情况都可以通过日志中新 inode 的存在来识别）。

前滚中的最后一个问题是如何恢复目录条目和 inode 之间的一致性。每个 inode 都包含引用该 inode 的目录条目数量的计数;当计数降至零时，文件将被删除。不幸的是，当一个 inode 已经用新的引用计数写入日志，而包含相应目录条目的块还没有写入时，就可能发生崩溃，反之亦然。

为了恢复目录和 inode 之间的一致性，Sprite LFS 在日志中为每个目录更改输出一条特殊记录。该记录包括操作代码（创建、链接、重命名或取消链接）、目录条目的位置（目录的 i 编号和目录中的位置）、目录条目的内容（名称和 i 编号）以及条目中命名的 inode 的新引用计数

这些记录统称为目录操作日志; Sprite LFS 保证每个目录操作日志条目在日志中出现在相应的目录块或 inode 之前。

在前滚期间，目录操作日志用于确保目录条目和 inode 之间的一致性：如果出现日志条目但 inode 和目录块未同时写入，则前滚更新目录和/或 inode 以完成操作。前滚操作可能会导致向目录中添加条目或从目录中删除条目，以及更新信息节点上的引用计数。恢复程序将更改的目录、inode、inode 映射和段使用表块追加到日志中，并写入新的检查点区域以包含它们。唯一无法完成的操作是创建一个新文件，而该文件的 inode 从未写入;在这种情况下，目录条目将被删除。除了它的其他功能之外，目录日志还可以轻松地提供原子重命名操作。

目录操作日志和检查点之间的交互给 Sprite LFS 带来了额外的同步问题。特别是，每个检查点必须表示目录操作日志与日志中的 inode 和目录块一致的状态。这需要额外的同步，以防止在写入检查点时修改目录。

1. Sprite LFS 的经验我们于 1989 年底开始实施 Sprite LFS，到 1990 年中期，它已作为 Sprite 网络操作系统的一部分投入使用。自 1990 年秋季以来，它已被用于管理五个不同的磁盘分区，约有 30 个用户用于日常计算。本文中描述的所有特性都已在 Sprite LFS 中实现，但前滚尚未安装在生产系统中。生产磁盘使用较短的检查点间隔（30 秒），并在重新启动时丢弃最后一个检查点之后的所有信息。

当我们开始这个项目时，我们担心日志结构的文件系统可能比传统的文件系统更复杂。然而，实际上，Sprite LFS 并不比 Unix FFS 更复杂[9]：Sprite LFS 对于段清理器来说具有额外的复杂性，但这通过消除 Unix FFS 所需的位图和布局策略来补偿;此外，Sprite LFS 中的检查点和前滚代码并不比扫描 Unix FFS 磁盘以恢复一致性的 fsck 代码更复杂[15]。像 Episode[2]或 Cedar[3]这样的日志文件系统可能比 Unix FFS 或 Sprite LFS 更复杂，因为它们同时包含日志和布局代码。

在日常使用中，用户感觉 Sprite LFS 与 Sprite 中的 Unix FFS 类似的文件系统没有太大区别。原因是所使用的机器速度不够快，无法与当前工作负载进行磁盘绑定。例如，在修改后的 Andrew 基准[11]中，

![](Pasted%20image%2020250525230439.png)

图 8 -Sprite LFS 和 SunOS 下的小文件性能。

图（a）测量了一个基准测试，该基准测试创建了 10000 个一千字节的文件，然后按照与创建相同的顺序读回它们，然后删除它们。速度是通过两个文件系统上每个操作每秒的文件数来衡量的。Sprite LFS 中的日志记录方法为创建和删除提供了数量级的加速。图（B）估计了每个系统在具有相同磁盘的更快计算机上创建文件时的性能。在 SunOS 中，磁盘在（a）中达到 85%的饱和，因此更快的处理器不会显著提高性能。在 Sprite LFS 中，磁盘在（a）中仅饱和 17%，而 CPU 利用率为 100%;因此，I/O 性能将随 CPU 速度而扩展。

使用第 5.1 节中提供的配置，Sprite LFS 仅比 SunOS 快 20%。 大部分的加速是由于 Sprite LFS 中同步写入的删除。即使使用 Unix FFS 的同步写入，基准测试的 CPU 利用率也超过 80%，限制了磁盘存储管理变化可能带来的加速。

5.1.微基准测试我们使用了一系列小型基准测试程序来测量 Sprite LFS 的最佳性能，并将其与 SunOS 4.0.3 进行比较，后者的文件系统基于 Unix FFS。这些基准测试是合成的，因此它们并不代表实际的工作负载，但它们说明了这两个文件系统的优点和缺点。用于这两个系统的机器是 Sun-4/260（8.7 整数 SPECmarks），具有 32 MB 内存，Sun SCSI 3 HBA 和 Wren IV 磁盘（1.3 MB/秒最大传输带宽，17.5 毫秒平均寻道时间）。对于 LFS 和 SunOS，磁盘都是使用具有大约 300 MB 可用存储的文件系统进行格式化的。SunOS 使用的块大小为 8 MB，而 Sprite LFS 使用的块大小为 4 MB，段大小为 1 MB

在每种情况下，系统都运行多用户，但在测试期间处于静止状态。对于 Sprite LFS，在基准测试运行期间没有发生清理，因此测量结果代表了最佳情况下的性能;有关清理开销的测量结果，请参见下面的第 5.2 节。

图 8 显示了创建、读取和删除大量小文件的基准测试的结果。在基准测试的创建和删除阶段，Sprite LFS 的速度几乎是 SunOS 的十倍。Sprite LFS 在阅读回文件方面也更快;这是因为文件是按照创建的相同顺序读取的，并且日志结构化文件系统将文件密集地打包在日志中。此外，Sprite LFS 在创建阶段仅使磁盘保持 17%的忙碌，同时使 CPU 饱和。相比之下，SunOS 在创建阶段使磁盘忙碌的时间为 85%，即使只有大约 1.2%的磁盘潜在带宽用于新数据。这意味着随着 CPU 速度的提高，Sprite LFS 的性能将提高 4-6 倍（参见图 8（B））。在 SunOS 中几乎没有任何改进。

虽然 Sprite 的设计目的是提高具有许多小文件访问的工作负载的效率，但图 9 显示它也为大文件提供了具有竞争力的性能。在所有情况下，Sprite LFS 的写入带宽都比 SunOS 高。对于随机写入来说，它的速度要快得多，因为它将这些写入转换为对日志的顺序写入;对于顺序写入来说，它的速度也更快，因为它将许多数据块分组到单个大型 I/O 中，而 SunOS 执行

![](Pasted%20image%2020250525230452.png)

图 9 -Sprite LFS 和 SunOS 下的大文件性能。

该图显示了一个基准测试的速度，该基准测试使用顺序写入创建一个 100 MB 的文件，然后顺序读回该文件，然后将 100 MB 随机写入现有文件，然后从该文件中随机读取 100 MB，最后再次顺序读取该文件。五个相位中每一个的带宽分别显示。Sprite LFS 具有更高的写入带宽和与 SunOS 相同的读取带宽，但顺序阅读随机写入的文件除外。

每个块单独的磁盘操作（SunOS 的新版本组写入[16]，因此应该具有与 Sprite LFS 相当的性能）。两个系统的读取性能相似，除了在随机写入文件后按顺序阅读文件的情况;在这种情况下，读取需要在 Sprite LFS 中进行寻道，因此其性能大大低于 SunOS。

图 9 说明了日志结构文件系统在磁盘上产生的局部性形式与传统文件系统不同。传统的文件系统通过假定某些访问模式（文件的顺序阅读、在目录内使用多个文件的倾向等）来实现逻辑局部性;然后，如果需要，它在写入上支付额外的费用，以便针对假定的读取模式在盘上最佳地组织信息。相比之下，日志结构的文件系统实现了时间局部性：同时创建或修改的信息将在磁盘上紧密分组。如果时间局部性与逻辑局部性匹配，就像它对顺序写入然后顺序读取的文件所做的那样，那么日志结构文件系统在处理大文件时应该与传统文件系统具有大致相同的性能。如果时间局部性不同于逻辑局部性，那么系统将以不同的方式执行。 Sprite LFS 处理随机写入的效率更高，因为它将它们顺序写入磁盘。SunOS 为实现逻辑局部性而为随机写入付出更多，但随后它更有效地处理顺序重新读取。随机读取在两个系统中具有大致相同的性能，尽管块的布局非常不同。但是，如果非顺序读取与非顺序写入的顺序相同，那么 Sprite 会快得多。

5.2.清洁开销上一节的微基准测试结果对 Sprite LFS 的性能给出了乐观的看法，因为它们不包括任何清洁开销（

基准测试运行期间的写入成本为 1.0）。为了评估清理成本和成本效益清理策略的有效性，我们记录了几个月内有关生产日志结构文件系统的统计数据。测试了五个系统：

- **/user6**
    
    Sprite 开发者的主目录。工作负载包括程序开发、文本处理、电子通信和仿真。
    
- **/pcs**
    
    并行处理和 VLSI（超大规模集成电路）设计研究的主目录与项目区。
    
- **/src/kernel**
    
    Sprite 内核的源代码和二进制文件。
    
- **/swap2**
    
    Sprite 客户端工作站的交换文件。工作负载包括为 40 台无盘 Sprite 工作站提供虚拟内存后备存储。文件通常很大、稀疏，且访问方式是非顺序的。
    
- **/tmp**
    
    40 台 Sprite 工作站的临时文件存储区域。
    

表 2 显示了在四个月的清洁期间收集的统计数据。为了消除启动影响，我们在文件系统投入使用后等待了几个月才开始测量。生产文件系统的行为比第 3 节中的模拟所预测的要好得多。尽管总体磁盘容量利用率在 11- 75%之间，但清理的数据段中有一半以上是完全空的。即使是非空段的利用率也远低于平均磁盘利用率。总体写入成本范围为 1.2 至 1.6，而相应模拟中的写入成本为 2.5-3。图 10 显示了在/user 6 磁盘的最近快照中收集的段利用率分布。

我们认为有两个原因可以解释为什么 Sprite LFS 中的清洁成本低于模拟。第一、
![](Pasted%20image%2020250525230508.png)
表 2-生产文件系统的段清理统计信息和写入成本。

对于每个 Sprite LFS 文件系统，该表列出了磁盘大小、平均文件大小、平均每日写入流量速率、平均磁盘容量利用率、四个月内清理的段总数、清理时为空的段的比例、清理的非空段的平均利用率以及测量期间的总体写入成本。这些写入成本数字意味着清理开销将长期写入性能限制在最大顺序写入带宽的 70%左右。

所有的模拟文件都只有一个块长。在实践中，存在大量较长的文件，并且它们倾向于作为一个整体被写入和删除。这导致各个细分市场内更大的局部性。在最好的情况下，文件比段长得多，删除文件将产生一个或多个完全空的段。模拟和现实之间的第二个区别是，模拟的参考模式均匀地分布在热文件组和冷文件组中。在实践中，有大量的文件几乎从未被写入（实际上的冷段比模拟中的冷段冷得多）。日志结构的文件系统会将非常冷的文件隔离在段中，并且永远不会清理它们。 在模拟中，每个段最终都接受了修改，因此必须进行清理。

如果 5.1 节中对 Sprite LFS 的度量有点过于乐观，那么本节中的度量就过于悲观了。在实践中，可以在夜间或在其他空闲时段期间执行大部分清洁，使得清洁段在活动突发期间可用。我们还没有足够的经验与雪碧 LFS 知道这是否可以做到。此外，随着我们获得经验和调整算法，我们预计 Sprite LFS 的性能会有所提高。例如，我们还没有仔细分析一次清理多少段的策略问题，但我们认为这可能会影响系统将热数据与冷数据隔离的能力。

![](Pasted%20image%2020250525230523.png)

图 10 -/user 6 文件系统中的段利用率

此图显示了/user 6 磁盘最近快照中的段利用率分布。该分布显示了大量的完全利用的段和完全空的段。

5.3.崩溃恢复虽然崩溃恢复代码尚未安装在生产系统上，但该代码运行良好，足以对各种崩溃场景进行时间恢复。恢复时间取决于检查点间隔以及正在执行的操作的速率和类型。表 3 显示了不同文件大小和恢复的文件数据量的恢复时间。不同的崩溃配置是通过运行一个程序生成的，该程序在系统崩溃之前创建了一个、十个或五十个固定大小的文件。使用了一个特殊版本的 Sprite LFS，它具有无限的检查点间隔，并且从不将目录更改写入磁盘。在恢复前滚期间，必须将创建的文件添加到 inode 映射中，创建目录条目，并更新段使用表。

表 3 显示恢复时间随最后检查点和崩溃之间写入的文件数量和大小而变化。可以通过限制在检查点之间写入的数据量来限制恢复时间。从表 2 中的平均文件大小和每日写入流量来看，检查点间隔长达一小时将导致平均恢复时间约为一秒。如果使用 150 MB/小时的最大已覆盖写入速率，则检查点间隔长度每增加 70 秒，最大恢复时间将增加 1 秒。

5.4. Sprite LFS 中的其他开销表 4 显示了写入磁盘的各种数据的相对重要性，包括它们在磁盘上占用的活动块的数量以及它们代表的写入日志的数据的数量。磁盘上 99%以上的实时数据由文件数据块和间接数据块组成。但是，写入日志的信息中约有 13%是由 inode、inode 映射块和 segment 映射块组成的，所有这些信息往往都会很快被覆盖。仅 inode 映射就占写入日志的所有数据的 7%以上。我们怀疑这是因为 Sprite LFS 中当前使用的检查点间隔较短，这会强制将元数据存储到磁盘

![](Pasted%20image%2020250525230533.png)

表 3 -各种崩溃配置的恢复时间

该表显示了恢复 1 MB、10 MB 和 50 MB 固定大小文件的速度。测量的系统与第 5.1 节中使用的系统相同。恢复时间取决于要恢复的文件数量。

比必要的更频繁。我们希望在安装前滚恢复并增加检查点间隔时，元数据的日志带宽开销会大幅下降。

6.相关工作日志结构文件系统概念和 Sprite LFS 设计借鉴了许多不同存储管理系统的想法。具有类似日志结构的文件系统已经出现在用于在一次写入介质上构建文件系统的几个提议中[17，18]。除了以仅附加的方式写入所有更改之外，这些系统还维护索引信息，非常类似于 Sprite LFS 索引节点映射和索引节点，以便快速定位和阅读文件。它们与 Sprite LFS 的不同之处在于，介质的一次写入特性使得文件系统不必回收日志空间。

Sprite LFS 中使用的段清理方法很像为编程语言开发的清除垃圾收集器[19]。在 Sprite LFS 中，在段清理期间的成本效益段选择和块的年龄排序将文件分成几代，就像分代垃圾收集方案一样[20]。 这些垃圾收集方案与 Sprite LFS 之间的一个显著差异是，在分代垃圾收集器中可以进行有效的随机访问，而顺序访问是在文件系统中实现高性能所必需的。此外，Sprite LFS 可以利用块一次最多属于一个文件的事实，使用比编程语言系统中使用的更简单的算法来识别垃圾。

Sprite LFS 中使用的日志记录方案类似于数据库系统中开创的方案。几乎所有的数据库系统都使用写前日志来实现崩溃恢复和高性能[13]，但与 Sprite LFS 不同的是它们如何使用日志。Sprite LFS 和数据库
![](Pasted%20image%2020250525230542.png)

表 4 -/user 6 的磁盘空间和日志带宽使用情况

对于每种数据块类型，该表列出了磁盘上使用的磁盘空间百分比（实时数据）和写入此数据块类型所消耗的日志带宽百分比（日志带宽）。标记为“*”的块类型在 Unix FFS 中具有等效的数据结构。

系统将日志视为有关磁盘上数据状态的最新“真相”。主要区别在于数据库系统不使用日志作为数据的最终存储库：为此目的保留了单独的数据区域。这些数据库系统的独立数据区意味着它们不需要 Sprite LFS 的段清理机制来回收日志空间。当记录的更改已写入其最终位置时，数据库系统中日志占用的空间可以回收。由于所有读请求都是从数据区处理的，因此可以大大压缩日志，而不会影响读性能。通常情况下，只有更改的字节被写入数据库日志，而不是像 Sprite LFS 中那样写入整个块。

Sprite LFS 的检查点崩溃恢复机制和使用“重做日志”的前滚类似于数据库系统和对象存储库中使用的技术[21]。Sprite LFS 中的实现得到了简化，因为日志是数据的最终归属地。Sprite LFS 恢复不是对单独的数据副本重新执行操作，而是确保索引指向日志中数据的最新副本。

在文件缓存中收集数据并将其写入磁盘中的大型写入类似于数据库系统中的组提交概念[22]和主存数据库系统中使用的技术[23，24]。

7.日志结构文件系统的基本原理很简单：在主内存的文件缓存中收集大量新数据，然后在一个可以使用所有磁盘带宽的大型 I/O 中将数据写入磁盘。由于需要在磁盘上保持较大的空闲区域，因此实现此想法很复杂，但我们的模拟分析和 Sprite LFS 经验都表明，基于成本和收益的简单策略可以实现较低的清洁开销。虽然我们开发了一个日志结构的文件系统来支持包含许多小文件的工作负载，但这种方法也适用于大文件访问。特别是，对于完全创建和删除的非常大的文件，基本上没有清理开销。

最重要的是，日志结构的文件系统可以比现有的文件系统更有效地使用磁盘。这将使得在 I/O 限制再次威胁到计算机系统的可伸缩性之前，利用更多代更快的处理器成为可能。

8.致谢黛安格林、玛丽贝克、约翰哈特曼、迈克库普弗、肯谢里夫和吉姆莫特史密斯对本文的草稿提出了有益的意见。
# 附录
- 原文：chrome-extension://bpoadfkcbjbfhfodiogcnhhhpibjhbnh/pdf/index.html?file=https%3A%2F%2Fpages.cs.wisc.edu%2F%7Eremzi%2FOSTEP%2Ffile-lfs.pdf