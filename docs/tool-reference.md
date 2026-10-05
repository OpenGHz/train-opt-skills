# 工具输入与输出

所有路径从项目根目录计算。Python 3.10+；只有张量比较需要 NumPy。工具只读取输入并输出报告，不会启动、修改或停止训练。

## 退出码与输出

| 工具 | 0 | 1 | 2 |
| --- | --- | --- | --- |
| compare_benchmarks | 测量协议可比较 | 协议不一致 | 输入或输出错误 |
| audit_sample_ids | 没有检测到指定范围内的不符 | 重复/缺失/多余身份 | 输入或输出错误 |
| compare_tensors | 该快照符合给定容差 | 键/形状/数值/非有限值不符 | 输入或输出错误 |
| collect_env | 报告已生成，探测可在报告中失败 | 不使用 | 输出错误 |

默认打印 JSON；`--output path.json` 写文件。三个比较/审计工具防止直接路径、符号链接或硬链接覆盖输入。覆盖其他已存在报告是调用者的选择，请使用独立实验路径。

## 1. 环境记录

```bash
python skills/optimize-training/scripts/collect_env.py --output environment.json
python skills/optimize-training/scripts/collect_env.py --probe-torch --nvidia-smi --output gpu-environment.json
```

默认记录 Python、平台、预定义软件包版本；不读取完整环境变量、不导入 torch。第二个命令显式启用 torch 子进程探测（20 秒超时）与 nvidia-smi 查询（5 秒超时）。不输出 GPU UUID、进程列表或凭证。探测失败单独记状态，不代表训练失败。

## 2. 性能比较

推荐新实验使用 [schema v2](experiment-contract.md)，它将训练语义不变量与声明的执行变量分离。以下原格式仍作为 schema v1 向后兼容；不要把 v1 的严格全字典匹配限制误认为所有有效实验都不能改变执行设置。

```bash
python skills/optimize-training/scripts/compare_benchmarks.py examples/baseline.json examples/candidate.json --output comparison.json
```

以 [baseline.json](../examples/baseline.json) 为完整可运行模板。

| 字段 | 契约 |
| --- | --- |
| schema_version | 整数 1 |
| synthetic | 必需布尔值；真实测量填 false |
| measurement.unit | samples、nonpadding_tokens 等非空有效工作单位 |
| measurement.scope | 明确窗口，例如 training_update |
| measurement.warmup_excluded | 必须 true；该工具用于预热后的窗口 |
| measurement.synchronization | 实际同步方式的非空说明 |
| measurement.includes_data_loading / includes_validation / includes_checkpointing | 三个必需布尔值 |
| protocol | 见下方完整字段；基线与候选必须完全匹配 |
| runs | 一个或多个独立窗口；建议重复测量 |
| runs[].elapsed_seconds / effective_units | 正、有限数值 |
| runs[].update_times_ms | 可选，实际测量的完整参数更新时间列表，正、有限；不可把未同步的提交耗时当更新耗时 |

protocol 的必需非空字符串：`workload_id, dataset_id, model_id, precision, input_shape, hardware, software`。必需正整数：`global_batch_size, world_size, gradient_accumulation_steps`。另需整数 `seed`。

两份记录的 measurement/protocol 全字典（包括扩展字段）必须一致。建议额外记录数据采样、mask/loss 归一化、优化器和增强协议等不变量。将变化的代码 revision、优化选项及实验身份放顶层附加字段和 experiment 记录中，不能伪造相同版本。software 表示保持固定的依赖栈；若依赖栈本身是实验变量，这个保守比较器会拒绝，需要另作明确标记的对照分析。

聚合吞吐为：

```text
throughput = sum(effective_units) / sum(elapsed_seconds)
speedup = candidate.throughput / baseline.throughput
reduction_pct = (1 - 1 / speedup) * 100
```

同时报告每轮吞吐中位数与范围；有更新时间则报告合并样本的中位数。reduction_pct 表示同等有效工作量下的推导耗时降幅，不是包含编译、评估、保存等阶段的完整作业降幅。工具不计算显著性，不自动验收模型质量，不生成 P95。

数据并行多卡必须先在训练程序中把对齐窗口的全局工作量和最慢 rank 耗时汇总为一个 run，不能把多个并行 rank 当作串行 run 相加。增加 world_size、改变累积、改精度等会触发不可比；仍可研究这些变化，但需单独设计扩展/质量实验，不能修改元数据绕过限制。

## 3. 样本身份审计

```bash
python skills/optimize-training/scripts/audit_sample_ids.py examples/samples.jsonl --expected examples/expected_ids.json --output coverage.json
```

JSONL 每行：

```json
{"epoch": 0, "rank": 0, "worker": 0, "sample_id": "dataset-v1/episode-7/frame-42"}
```

epoch/rank/worker 为非负整数，无子 worker 时记 worker=0。sample_id 为非空字符串或整数；数值 1 与字符串 "1" 是不同身份。expected 为唯一身份的 JSON 数组，对每个已观察 epoch 应用。

默认重复为不符；有放回采样或补齐协议可用 `--allow-repeats`，但缺失/多余仍不被允许。带放回采样通常不要求单 epoch 覆盖全集，不应盲目传入全集作为 expected；按实际验收契约选择窗口与预期集合，并另验采样分布。

未提供 expected 时 coverage=unknown，退出 0 只表示没有检测到重复等已检查问题。工具无法推断完全没有记录的 epoch/rank/worker，无法验证时序、增强、过滤规则或样本统计独立性。完整计数保留，报告只列有限数量的示例。

## 4. 张量快照比较

```bash
python skills/optimize-training/scripts/compare_tensors.py reference.npz candidate.npz --atol 0.00001 --rtol 0.0001 --output parity.json
```

容差必须显式指定；上面的数值仅示意语法。仅支持 NumPy 可读取的真实整数/浮点 NPZ 数组，不支持 object、bool、complex 或 pickle。BF16 等自定义类型应在可信框架中按已记录方式导出到受支持类型，不能把导出转换后的快照误认为原始计算路径。

键和形状必须一致，dtype 会报告但不要求相同。所有 NaN/Inf 均失败，包括相同位置的两个 NaN。

```text
abs(candidate - reference) <= atol + rtol * abs(reference)
```

相对误差以 reference 为尺度，交换输入可能改变结论。参考值为零且误差非零时，相对误差记为无界。报告逐键最大误差和超差计数，不打印张量内容。当本机比较 dtype 无法保留整数精度或输入浮点范围时，工具明确拒绝比较。按块处理计算临时数组，但 NPZ 成员解压仍需内存；只使用可信且大小可控的文件。

这些检查只能证明给定快照是否通过约定；不能替代梯度、更新、断点恢复和任务质量验收。


## 5. 可选专项技能发现

```bash
python skills/optimize-training/scripts/discover_integrations.py --root /known/skills
```

这是只读发现，不下载、安装或执行第三方技能。它依据真实 SKILL.md 名称匹配，记录路径、来源和内容检查，标注缺失或冲突；前提与交接见[集成说明](integrations.md)。输出状态只表示发现/就绪检查结果，不能代替一次实际诊断、测试或 GPU 实验。
