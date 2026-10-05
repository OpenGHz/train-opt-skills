# 训练正确性与性能优化 Agent Skill：现有方案调研与设计建议

> **整理日期**：2026-10-05。  
> **对应基础文档**：《深度学习训练全链路性能优化指南》v2。  
> **调研目标**：判断是否已有类似 Agent Skills，明确哪些能力适合直接复用、哪些工作流值得借鉴，以及将现有指南做成通用 skill 的合理定位。  
> **资料与验证边界**：本文整理自前述公开仓库调研，依据实际 `SKILL.md`、仓库目录及官方说明评价内容覆盖与工作流适配程度。未安装运行这些 skills，未复现其 GPU 实验。本次整理不构成新的在线复核；公开仓库及文档可能继续变化。

## 摘要与结论

**已经有几套相当接近的现有方案。** 最值得优先研究的是 **NVIDIA 的 MoE 训练优化工作流、`meta-pytorch/skills` 的性能诊断模块，以及 AMD Primus 的迁移验证 skill**。它们分别覆盖“优化闭环”“定位与修复”“正确性验收”。[1][2][11]

建议继续将现有指南做成 skill，但定位为：

> **正确性约束下的通用训练优化入口：面向以 PyTorch 为主的训练项目，在保持约定训练语义和质量的前提下，完成诊断、优化与验收。**

核心策略是复用现有专项 skills，而不是从零再写一套 PyTorch 优化知识库。价值不在于再次罗列 AMP、`torch.compile`、FSDP 等选项，而在于明确：**该不该改、如何证明改对了，以及什么时候必须停止或回滚。**

以下“复用建议”“局限判断”和“建议结构”属于本次调研形成的设计判断，不代表相应项目对自身能力的承诺。

## 一、最值得参考的现有 skills

### 1. NVIDIA：MoE Training Optimization Workflow——与整体思路最接近

**仓库**：`NVIDIA/skills`  
**文件**：`skills/nemo-mbridge-perf-moe-optimization-workflow/SKILL.md`  
**来源**：[1]

这是本次找到的、**在方法论上最接近现有指南的单个 skill**。它不是简单罗列优化参数，而是要求按照以下顺序工作：

```text
固定测量契约
    → 先让模型稳定地放入显存
    → 选择并行布局
    → 用性能剖析定位瓶颈
    → 做受控调优
    → 验证并打包证据
```

其要求包括固定硬件、版本、数据和批次口径，区分“保持训练语义的改动”与“仅用于 benchmark 的改动”，进行匹配条件下的 A/B 对照，并检查目标后端、精度内核或 CUDA Graph replay 是否真正生效。[1]

**最值得借鉴的是它对结论的约束。** 例如，不能把配置开关已开启当作优化生效，不能把重叠执行的 kernel 时间相加当成墙钟时间，也不能把最终多项改动的总收益归因给其中某一项。它要求最终保留命令、配置差异、运行环境、性能和健康状态等证据。[1]

**局限**：其具体决策围绕 **Megatron Bridge、Megatron Core 和 MoE** 展开，包含专家并行、dispatcher、Parallel Folding 等专用配置。因此，它适合作为工作流设计参考，不适合直接作为任意 PyTorch、视觉或 VLA 训练项目的统一入口。[1]

**复用建议**：优先精读，借鉴整体流程，但不要照搬框架参数和固定测量步数。

### 2. `meta-pytorch/skills`——最适合直接复用的诊断模块

**仓库**：`meta-pytorch/skills`  
**来源**：[2]–[6]

其中有三个与现有指南直接相关的 skills：

| Skill | 已有能力 | 在拟建体系中的位置 |
| --- | --- | --- |
| `profile-model` | 分析 PyTorch profiler 的 JSON/JSON.gz，支持多 rank，识别慢 kernel、GPU 空闲、通信和负载不均衡；可进一步使用 HTA 分析关键路径 | 性能证据分析 |
| `debug-graph-breaks` | 读取或生成 graph-break 日志，定位原因，修改代码并重新运行；支持结合官方 graph-break 文档诊断 | 编译问题定位与修复 |
| `analyze-memory-snapshot` | 分析 CUDA 内存快照，定位分配来源、延迟释放、缓存、OOM、保留张量；提供快照差分辅助脚本 | 显存问题定位 |

这些功能在实际 skill 文件中有明确的输入、操作步骤和输出要求，不只是项目 README 中的宣传。[3][4][5]

这里有两点特别值得借鉴。

**第一，专项任务可以独立完成。** 例如，`debug-graph-breaks` 不仅解释错误，还要求读取用户代码、选择修复方案并重新运行；它也区分“消除所有 graph break”和“优先修复真正影响性能的 graph break”。PyTorch 官方 DevLog 对这个 skill 及其基于真实开源模型场景的评测有专门介绍。[4][6]

**第二，仓库把 skill 评测纳入了工程结构。** 新建 skill 的脚手架包含 `SKILL.md`、`evals/eval.yaml`、检查器和测试输入目录，并配有评测框架。这比“写完 Markdown 就认为 skill 完成”更值得参考。[2]

**局限**：这些是专项诊断与修复能力，不是完整的训练验收流程。

**复用建议**：由总入口决定何时调用它们，并在修复后补上输入、梯度、参数更新及端到端性能回归。优先将其作为实际依赖或可选集成，而不是重新实现。

### 3. NVIDIA TensorRT-LLM：性能分析与优化案例库——值得借鉴技能拆分方式

**仓库**：`NVIDIA/TensorRT-LLM`  
**目录**：`.claude/skills/`  
**来源**：[7]–[10]

重点关注以下三个 skills：

| Skill | 作用 |
| --- | --- |
| `perf-analysis` | 协调性能测量、分类瓶颈并生成结构化报告 |
| `perf-workload-profiling` | 给训练循环或独立算子增加计时与 NVTX 标记 |
| `perf-optimization-casebook` | 根据瓶颈信号选择已有优化案例，记录适用条件、风险、验证与回滚方式 |

`perf-analysis` 明确要求性能数字来自工具输出，并将 profiling 委派给专项执行者；`perf-workload-profiling` 则明确限定自己负责“如何测量”，不负责运行所有 profiler 或直接实施优化。[8][9]

**其中最值得借鉴的是 `perf-optimization-casebook`。** 它将每项优化组织成“决策案例”，而不是一句“建议开启 FlashAttention”。一个案例需要说明：

> 什么证据支持使用它、什么情况不该使用、为什么可能有效、如何适配当前项目、有哪些精度风险、怎样验证、何时回滚。

它明确区分“案例预期效果”与“当前项目实测结果”，并允许记录有价值的失败案例。[10]

**局限**：案例主体偏向 TensorRT-LLM 推理与运行时；`perf-analysis` 还依赖其 specialist 体系。不能只复制一个 `SKILL.md`，就假定相关子 agent 和执行能力也存在。[8][10]

**复用建议**：借鉴“协调入口＋测量模块＋案例库”的组织方式，具体推理案例不应直接套到训练。

### 4. AMD Primus：`port-validation-guide`——最接近跨实现正确性验证部分

**仓库**：`AMD-AGI/Primus`  
**文件**：`skills/port-validation-guide/SKILL.md`  
**来源**：[11]

它针对的问题是：将 Primus 或 Primus-Turbo 中的优化迁移到自己的训练框架后，如何验证迁移是否成功。其验证分为三层：

| 层次 | 验证内容 |
| --- | --- |
| 正确性 | 选择参考实现，比较输出，需要时比较梯度，按精度设置容差 |
| 性能 | 相同条件下比较 feature OFF/ON，包含预热、计时和峰值显存 |
| 集成 | 相关回归测试、短程训练、非有限值检查、checkpoint 保存与加载 |

它要求先查明被替换的符号、两侧实现、开关、版本和依赖约束，然后据此生成测试，而不是套用完全通用的测试模板。[11]

**重要边界：这个 skill 明确只生成验证计划和测试骨架，不运行测试，也不保证迁移正确。** 因此，它适合参考“验证计划生成”模块，不能被当成已经具备自动验收能力。[11]

**复用建议**：借鉴测试分层、参考对象选择，以及“未执行就不得声称通过”的状态管理。

### 5. `stas00/ml-engineering`：`ml-engineering`——领域覆盖广的知识型 skill

**仓库**：`stas00/ml-engineering`  
**当前入口**：`skills/ml-engineering/SKILL.md`  
**来源**：[12][13]

仓库根目录原来的 `SKILL.md` 已变为迁移提示，指向上述入口。[13]

该 skill 覆盖训练吞吐、显存、数据加载、网络与存储、并行策略、数值不稳定、故障恢复、checkpoint 和多机调试。其定位明确是 **Machine Learning Engineering Open Book 的浓缩索引**，通过链接进入详细章节、脚本和 benchmark。[12]

它与现有指南在“不要只盯 GPU 算力，要检查完整系统”这一点上高度一致。区别在于，它更像工程知识库和运行手册入口，而不是要求每次优化都严格提交一套统一验收产物的执行流程。

**复用建议**：作为补充参考库，不建议将全文复制进总入口。

**许可注意事项**：该文件注明来源采用 **CC BY-SA 4.0**。直接改编时需要单独处理许可要求，不能默认与其他 skill 使用同一种许可证。[12]

### 6. Orchestra Research：AI Research Skills——补充框架专项知识

**仓库**：`Orchestra-Research/AI-Research-SKILLs`  
**来源**：[14]

相关类别主要是：

| 类别 | 与本项目相关的条目 |
| --- | --- |
| 分布式训练 | FSDP2、DeepSpeed、Accelerate、Megatron-Core、Lightning、Ray Train |
| 计算与精度优化 | Flash Attention、bitsandbytes 等 |
| 实验管理 | Weights & Biases、MLflow、TensorBoard |

这些条目可以作为“已经确定要使用某项技术之后”的专项操作知识。需要注意，`optimization` 分类中也包含 GPTQ、AWQ、GGUF 等内容，不能把整个分类都等同于训练性能优化。[14]

该仓库也有 `autoresearch` 编排层，因此不能简单将其描述为“只有工具说明”。不过，科研流程编排与“固定训练语义下的正确性及性能验收”仍是两个不同目标。[14]

**复用建议**：选择性引用相关专项 skill，没必要为了本项目引入整套科研技能库。

### 7. Nsight Systems 自带的官方 skill pack——可直接利用的工具入口

**官方说明**：Nsight Systems Analysis Guide  
**本地入口**：`<nsys-install-folder>/skills/nsight-systems/SKILL.md`  
**来源**：[15]

前述调研查阅的 Nsight Systems 官方文档已经介绍了随产品提供的 agent skill pack。它使用渐进式加载：从 `SKILL.md` 进入知识库和脚本，帮助 agent 使用 CLI、运行分析 recipes，并根据采集到的性能数据回答问题。[15]

其分析范围覆盖 CPU/GPU、CUDA API、NVTX、MPI、NCCL 等。官方将该能力标为 **Preview**，实际使用要确认本地安装版本是否包含。[15]

**复用建议**：拟建 skill 可优先发现并使用本地官方 skill pack，而不是长期维护一份复制过来的 Nsight 命令大全。

## 二、现有指南还有没有单独做成 skill 的价值？

**有，但价值不应宣称为“首次把 profiling、A/B 测试和正确性检查放在一起”。** 这些思路已经有很接近的实现，尤其是 NVIDIA 的 MoE 工作流。[1]

真正值得做的是：

> **把当前散落在不同框架、不同工具里的能力，组织成面向普通训练项目、带统一验收协议的流程。**

现有 v2 指南已经具备几个很好的落脚点：

| 可形成特色的部分 | 做成 skill 时应落地为什么 |
| --- | --- |
| 跨实现对齐契约 | 自动整理参考实现、固定测试输入、比较边界，定位第一处分歧 |
| 实际数据正确性 | 检查 rank × worker 的真实样本输出，而不只审查 sampler 配置 |
| 增强语义 | 分别验证随机参数共享范围、统计维度和时序/多视角约定 |
| 权重与恢复语义 | 检查张量映射、加载后输出，以及部署权重和精确续训状态的区别 |
| 完整训练收益 | 统一记录稳态吞吐、完整作业耗时、质量门槛和回滚条件 |

这些要求已经分别体现在《深度学习训练全链路性能优化指南》v2 的跨实现对齐、数据覆盖与增强审计，以及权重导出和最终验收章节中。

**在本次重点核查的这些入口中，没有找到一个无需改造、就能完整替代上述组合的单一通用 skill。** 这是对本次核查范围的判断，不代表全网不存在其他实现。

## 三、建议怎样组织，而不是直接把文档改名为 `SKILL.md`

### 3.1 一个总入口，按需加载参考模块

建议第一版采用 **一个总入口、若干按需加载的参考模块，以及少量可执行辅助脚本**。先不要拆成十几个必须互相依赖的 skills。

下面是建议结构，尚未创建或实现：

```text
training-correctness-and-performance/
├── SKILL.md
├── references/
│   ├── correctness-and-alignment.md
│   ├── measurement-contract.md
│   ├── data-pipeline-audit.md
│   ├── optimization-casebook.md
│   └── checkpoint-and-resume.md
├── scripts/
│   ├── collect_environment.py
│   ├── compare_runs.py
│   └── audit_sample_coverage.py
├── assets/
│   ├── experiment-record.yaml
│   └── acceptance-report.md
└── evals/
    ├── eval.yaml
    ├── fixtures/
    └── checkers.py
```

`SKILL.md` 只负责决定：**现在处于哪个阶段、缺少什么证据、该加载哪个模块、允许做什么修改、需要满足哪些验收条件。** 现有指南中的详细知识放进 `references/`，实验模板放进 `assets/`。

### 3.2 明确运行能力与证据状态

建议把运行能力明确分成三种状态：

| 状态 | 允许交付的结论 |
| --- | --- |
| 只有代码和配置 | 静态审计、风险清单、验证计划 |
| 有日志、trace 或快照 | 基于已有证据的诊断；明确证据覆盖范围 |
| 有可用执行环境 | 在约定资源预算内运行测试与 A/B 实验，输出实测结果 |

**不能把“生成了测试代码”“测试已执行”“测试通过”混成同一个完成状态。** AMD 的 skill 对这一边界的处理值得继承。[11]

### 3.3 区分训练程序测试与 skill 自身评测

应分别建立两种测试：一种测试被优化的训练程序；另一种测试 **skill 是否真的让 agent 更可靠**。

后一种可以参考 `meta-pytorch/skills` 的评测结构。[2] 建议设计带有已知问题的任务，例如：

| 评测场景 | 要观察的 agent 行为 |
| --- | --- |
| 异步计时错误 | 是否识别计时边界错误，而不是直接采信错误加速比 |
| 跨 rank 重复采样 | 是否检查真实样本身份，而不只看配置或 GPU 利用率 |
| batch 级增强统计错误 | 是否区分统计维度错误与随机参数共享规则 |
| 同形状权重重排错误 | 是否在加载成功之外继续检查张量语义和输出 |
| 提高吞吐但改变训练协议的“假优化” | 是否识别语义变化，要求重新验证质量，而非宣称等价加速 |

上述评测场景是拟建 skill 的设计建议，不是对现有项目评测覆盖范围的描述，也不是已经完成的测试。

## 四、最终建议与参考优先级

推荐名称：**`training-correctness-and-performance`**。

推荐定位：

> 面向以 PyTorch 为主的训练项目，在保持约定训练语义和质量的前提下，完成诊断、优化与验收。

参考与复用优先级如下：

| 优先方向 | 参考对象 | 采用方式 |
| --- | --- | --- |
| 总体优化流程 | NVIDIA MoE Training Optimization Workflow | 精读测量契约、受控调优、验证和证据交付流程，不照搬框架专用参数 |
| 专项诊断与修复 | `meta-pytorch/skills` | 优先接入性能分析、graph-break 修复和内存快照分析能力 |
| 验证分层与状态管理 | AMD `port-validation-guide` | 借鉴正确性、性能、集成三层验证，以及未执行不得声称通过的边界 |
| 优化案例沉淀 | TensorRT-LLM `perf-optimization-casebook` | 借鉴适用信号、反向信号、机制、风险、验证和回滚的案例组织方式 |
| 补充知识与工具 | `stas00/ml-engineering`、Orchestra Research、Nsight Systems skill pack | 按需查询或集成，避免将所有知识和工具说明塞进总入口 |

现有 v2 指南则作为通用约束和验收协议的主体。

这样做的成果不是“又一个告诉 agent 开启 AMP、compile 和 FSDP 的 skill”，而是 **让 agent 知道该不该改、如何证明改对了，以及什么时候必须停止或回滚的执行规范**。

---

## 参考资料与入口

以下链接保留前述调研的来源入口。`main`、`master` 和产品在线文档可能变化，实际集成时应记录所采用的提交或版本。本文没有因整理而重新核实仓库状态。

[1] NVIDIA，**MoE Training Optimization Workflow**。  
<https://github.com/NVIDIA/skills/blob/main/skills/nemo-mbridge-perf-moe-optimization-workflow/SKILL.md>

[2] meta-pytorch，**skills 仓库**。专项 skills、脚手架与评测结构。  
<https://github.com/meta-pytorch/skills>

[3] meta-pytorch，**profile-model**。  
<https://github.com/meta-pytorch/skills/blob/main/skills/profile-model/SKILL.md>

[4] meta-pytorch，**debug-graph-breaks**。  
<https://github.com/meta-pytorch/skills/blob/main/skills/debug-graph-breaks/SKILL.md>

[5] meta-pytorch，**analyze-memory-snapshot**。  
<https://github.com/meta-pytorch/skills/blob/main/skills/analyze-memory-snapshot/SKILL.md>

[6] PyTorch，**Dynamo DevLog**。前述调研引用的 graph-break skill 及评测介绍入口。  
<https://docs.pytorch.org/devlogs/dynamo/>

[7] NVIDIA TensorRT-LLM，**skills 目录**。  
<https://github.com/NVIDIA/TensorRT-LLM/tree/main/.claude/skills>

[8] NVIDIA TensorRT-LLM，**perf-analysis**。  
<https://github.com/NVIDIA/TensorRT-LLM/blob/main/.claude/skills/perf-analysis/SKILL.md>

[9] NVIDIA TensorRT-LLM，**perf-workload-profiling**。  
<https://github.com/NVIDIA/TensorRT-LLM/blob/main/.claude/skills/perf-workload-profiling/SKILL.md>

[10] NVIDIA TensorRT-LLM，**perf-optimization-casebook**。  
<https://github.com/NVIDIA/TensorRT-LLM/blob/main/.claude/skills/perf-optimization-casebook/SKILL.md>

[11] AMD Primus，**port-validation-guide**。  
<https://github.com/AMD-AGI/Primus/blob/main/skills/port-validation-guide/SKILL.md>

[12] Stas Bekman，**Machine Learning Engineering skill**。  
<https://github.com/stas00/ml-engineering/blob/master/skills/ml-engineering/SKILL.md>

[13] Stas Bekman，**原根目录 SKILL.md 的迁移说明**。  
<https://github.com/stas00/ml-engineering/blob/master/SKILL.md>

[14] Orchestra Research，**AI Research Skills**。  
<https://github.com/Orchestra-Research/AI-Research-SKILLs>

[15] NVIDIA，**Nsight Systems Analysis Guide**。Agent skill pack 的官方说明入口。  
<https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html>

**内部依据**：《深度学习训练全链路性能优化指南》v2，文件名 `training_performance_optimization_guide_zh_v2.md`，2026-10-05。本文为独立调研文档，不替代或修改该指南。
