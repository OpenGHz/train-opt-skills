# 数据分片、增强与传输

在取数等待、多 rank 重复样本、预处理迁移或输入裁剪时使用本页。
先验证实际数据流，再判断吞吐；sampler 配置存在不等于分片有效。

## 定义有效输入

逐字段追踪来源、有效性标记、模型/损失用途、掩码生效位置及此前已付出的计算。
不要按像素全零判定相机无效；真实黑色图像也可能是观测。
裁剪占位字段前检查其缺失模态提示、状态更新、位置索引与归一化语义。
先处理数据集级固定无效字段，再测试形状分组，最后考虑逐批动态裁剪。

| 裁剪风险 | 验证证据 |
| --- | --- |
| mask / loss 改变 | 可见关系、损失权重和归一化分母一致 |
| 位置或输出错位 | 模态编号、时间索引、位置编码与输出映射一致 |
| 批内有效性不同 | 不因部分缺失而删除其他样本的有效输入 |
| 动态分支 / 形状 | 通信顺序合法，编译复用与负载可接受 |

减少真实观测、分辨率、历史长度或动作预测长度属于训练方案调整，另做质量对照。

## 分片所有权与接口契约

map-style 必须确认 `idx` 真正决定读取对象；内部游标忽略索引会使 sampler 失效。
iterable/chunk 流在实际读取层分片；已经分片的数据不能再被另一层重复切分。
记录分片所有者、数据并行 rank/world size、worker 数和清单版本。
rank 指数据并行采样空间，混合并行中不一定等于全局进程编号。

对于各 rank 使用相同 worker 数、同一 chunk 清单和顺序的固定拓扑：

```text
W = max(1, num_workers)
global_worker = dp_rank * W + worker_id
stride = dp_world_size * W
positions = range(global_worker, num_chunks, stride)
```

无子 worker 时使用 `worker_id=0, num_workers=0`；校验参数为整数且 rank/worker 在范围内。
训练主进程取得并校验 DP 元数据，再以可序列化值传给数据集；worker 读取自身编号。
不要依赖 worker 继承通信组，也不要把预期多卡中的 rank 查询失败静默回退成 0。
共同 epoch/种子先生成相同清单排列，再按位置分片。
异构 worker 数、弹性扩缩容和动态队列需要专门协议，不能直接套此公式。
该公式只证明清单位置划分，不证明 chunk 内无重复、过滤后完整或训练步数均衡。

## 实际输出审计

在小型有限测试集记录数据版本、episode/sample/frame ID、epoch、rank、worker 和更新编号。
比较真实输出集合及计数，分清预期重复、非预期重复、过滤和尾批丢弃。
不要每步全量收集到训练热路径；审计使用独立短运行。
覆盖零 worker、多 worker、多 rank、chunk 少于加载者和空分片。
空分片或不齐步数必须有显式训练协议，不能让部分 rank 提前退出而其他 rank 继续集合通信。
比较每 rank 的样本数、任务分布和工作量，chunk 数一致不足以证明均衡。
`DistributedSampler(drop_last=False)` 的补齐可能产生预期重复；shuffle 跨 epoch 要检查 `set_epoch()`。
恢复测试记录训练已消费进度，处理预取/worker 游标领先，不能只保存读取游标。
重复样本不意味着各卡梯度逐位相同，但仍需修复非预期的数据重复。

## 增强：分别核对随机粒度与统计维度

| 维度 | 必须写明的协议 |
| --- | --- |
| 随机参数 | 按图、样本、序列或视角组独立/共享 |
| 统计量 | 均值、对比度等在哪些空间/通道维归约，是否误包含 batch |
| 变换定义 | 顺序、插值、裁剪、取值范围和归一化边界 |
| 随机状态 | rank/worker/epoch/样本的子流规则与回放方式 |

独立图像 `(B,C,H,W)` 的逐图标量可用 `(B,1,1,1)` 广播；不需要强制 Python 循环。
先固定增强参数比较参考与向量化实现，再单独测试参数抽样粒度。
固定一张图及其增强参数，仅替换其他图；逐图独立变换的该图结果应不变。
视频、历史观测和多视角先遵循共享几何/颜色变换协议，不能一律改成逐图独立。
JAX 向量化同时检查 key 拆分；`vmap` 不会替代随机语义设计。

## 数据供给与 H2D

先分别测读取、解码、增强、组批、进程间传递，再调整 worker、预取与持久 worker。
按整机训练进程数乘 worker 数预算 CPU、内存和 I/O；增加并发可能加剧争抢。
只缓存语义允许的确定性结果，不静默冻结随机增强、模型相关结果或梯度计算。
真实 8 位 RGB 可试保留 `uint8` 到设备后转换；浮点插值后压回 uint8 不能宣称无损。
同元素数 uint8 字节数是 FP32 的四分之一，不代表整个训练加速四倍。
迁移预处理后比较相同样本的最终输入，核对通道、插值、舍入、裁剪及归一化顺序。
`non_blocking=True` 不证明计算/拷贝重叠；检查锁页内存、流、硬件及实际时间线。
明确源缓冲区复用和目标读取时机，异步拷贝期间不得提前修改源数据。
不要假定主线程临时 `pin_memory()` 有益，应测完整路径。

交付实际样本覆盖表、增强协议、输入回归、真实数据 A/B 与主机资源峰值。
只有取数等待/关键路径改善且语义未变，才保留候选；合成数据仅用于隔离瓶颈。

## 官方资料与版本边界

沿用指南引用；多进程启动方式、DataLoader 参数和传输行为按项目版本与平台核对。
JAX 同样支持多进程/多主机，不能按框架名称推断无需分片。

- [PyTorch data utilities](https://docs.pytorch.org/docs/2.14/data.html)
- [Distributed communication](https://docs.pytorch.org/docs/2.14/distributed.html)
- [pin_memory and non_blocking](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)
- [JAX random numbers](https://docs.jax.dev/en/latest/random-numbers.html)
- [Multi-controller JAX](https://docs.jax.dev/en/latest/multi_process.html)
