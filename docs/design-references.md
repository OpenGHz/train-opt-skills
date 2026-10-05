# 设计参考与来源

核查日期：2026-10-05。以下均为项目自身的 GitHub 页面或标准、平台的官方文档。分别借鉴领域工作流和工程组织方式；本项目的说明和工具独立编写，没有复制下列仓库的技能正文或代码。

## 领域设计与复用优先级

以用户提供的[专项技能调研](training_optimization_agent_skills_survey_zh.md)为起点，重新核对实际技能文件。以下项目决定功能分工；后面的热门通用项目决定工程包装。

| 来源 | 采用方式 | 明确边界 |
| --- | --- | --- |
| [NVIDIA MoE training workflow](https://github.com/NVIDIA/skills/blob/main/skills/nemo-mbridge-perf-moe-optimization-workflow/SKILL.md) | 固定测量契约、受控调优、验证优化实际生效并保留证据 | 不继承 MoE/Megatron 参数、固定步数或特定硬件推荐 |
| [meta-pytorch/skills](https://github.com/meta-pytorch/skills) | 可选复用 profile-model、debug-graph-breaks、analyze-memory-snapshot；借鉴 fixture/checker/A-B 评测结构 | 技能不等于可调用 API；依赖、任务输入和宿主执行能力单独核查 |
| [AMD Primus port-validation-guide](https://github.com/AMD-AGI/Primus/blob/main/skills/port-validation-guide/SKILL.md) | 正确性、性能、集成分层，记录计划/生成/执行/通过状态 | 上游只输出验证计划与骨架，不能把调用该技能记作测试通过 |
| [TensorRT-LLM casebook](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.claude/skills/perf-optimization-casebook/SKILL.md) | 适用信号、反向信号、机制、风险、验证、回滚与交互作用 | 本项目案例面向训练，均明确证据成熟度；不照搬推理的 lossless 判定或固定回滚百分比 |
| [Nsight Systems 官方说明](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html) | 若用户提供的本地技能包实际存在，按需交接系统级分析 | 版本与本地文件优先，不假设已安装或支持某个命令 |

Meta PyTorch 可选集成登记的已核查仓库提交为 `15ea40020f7ab0ad6645f78dca3b5634383e93c9`，并为三个入口记录独立内容 SHA-256。仓库提交与 Git 文件 blob SHA 不混用；这只是核查来源，不代表本项目已经执行其 GPU 功能。注册信息位于技能 assets/integrations.json，具体规则见[集成文档](integrations.md)。

本项目未直接捆绑第三方技能；Meta 仓库为 BSD-3-Clause。如果未来改成内置副本，必须保留对应来源和许可，并复制实际依赖资源而非只有 SKILL.md。知识型参考如 stas00/ml-engineering 按需查阅；不把其 CC BY-SA 内容并入本项目 MIT 正文。

## 工程结构参考项目

GitHub 星标仅是核查时页面显示的近似快照，用于说明选择了有广泛使用基础的项目；不是质量排名，也不表示这些项目验证或推荐了本项目。

| 项目 | 星标快照 | 本项目采用的做法 | 注意事项 |
|---|---:|---|---|
| [anthropics/skills](https://github.com/anthropics/skills) | 约 179.7k | 每项技能自包含；入口、脚本、参考资料、静态模板分离；清晰描述触发条件 | 仓库内并非统一许可证：不少技能为 Apache-2.0，文档处理技能包括仅开放源码可阅的条款；不能把整个仓库视为可任意复制的 Apache 项目 |
| [obra/superpowers](https://github.com/obra/superpowers) | 约 295.5k | 把工作流变成可执行步骤；区分技能行为评估与工具测试；用失败场景验证指令是否有效 | MIT；本项目未引入它的全局强制加载规则或完整开发流程 |
| [vercel-labs/agent-skills](https://github.com/vercel-labs/agent-skills) | 约 31.9k | 按影响与适用场景组织知识；入口给索引，细节按需读取；规则含解释、示例与参考资料 | MIT；训练优化收益取决于实测瓶颈，不能把前端规则的固定优先级直接移植到训练任务 |
| [K-Dense-AI/scientific-agent-skills](https://github.com/K-Dense-AI/scientific-agent-skills) | 约 47.6k | 技能版本、科学证据、边界条件、依赖与验证状态显式记录；脚本测试放在仓库级测试目录 | MIT；原 `claude-scientific-skills` 地址目前重定向到此仓库 |
| [openai/skills](https://github.com/openai/skills) | 约 27.9k | 参考其技能目录、可复用资源和按技能标注许可证的组织方式 | 当前 README 已声明弃用，新的插件示例转向 [openai/plugins](https://github.com/openai/plugins)；不将旧仓库作为当前推荐发布渠道 |

## 从文档到可运行技能

本项目以 `skills/optimize-training/SKILL.md` 为统一入口，保留“训练正确性优先、基于测量迭代”的原始目标。入口负责识别任务、选择参考材料、约定证据和验收输出；完整知识不全部塞入入口。

采用的组织原则：

1. **按需加载**：先读入口与相关主题，再在需要时读取数值对齐、数据流水线、分布式或权重转换细节。
2. **测量驱动**：每个候选优化都应说明瓶颈证据、适用条件、正确性检查、测量口径和回滚条件。
3. **工具可独立运行**：脚本提供命令行帮助、明确的输入输出和失败信息，不要求安装整套训练框架才能使用所有辅助功能。
4. **区分两类验证**：自动测试证明脚本对给定输入的行为；技能评估检查 agent 是否遵守流程。两者都不能替代真实 GPU 训练与任务质量验证。
5. **结果可追溯**：保存环境、实验配置、基线、候选改动、正确性结果、性能统计及未验证事项。

对应的一手参考：

- [Agent Skills 格式规范](https://agentskills.io/specification)：`SKILL.md` 元数据、`scripts/`、`references/`、`assets/` 与渐进加载。规范建议入口少于 500 行，并通过相对路径引用资源。
- [Anthropic skill-creator](https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md)：围绕真实任务提示设计评估，再记录预期结果与可检查断言。
- [Superpowers writing-skills](https://github.com/obra/superpowers/blob/main/skills/writing-skills/SKILL.md)：用具体触发条件提高发现准确度，用情境测试暴露流程缺口。
- [Vercel React Best Practices 入口](https://github.com/vercel-labs/agent-skills/blob/main/skills/react-best-practices/SKILL.md)：分类索引加独立规则文件，规则解释原因并附具体示例。
- [K-Dense 贡献规范](https://github.com/K-Dense-AI/scientific-agent-skills/blob/main/CONTRIBUTING.md)：技能版本、仓库级脚本测试、本地链接、元数据与可运行示例检查。

## 兼容与分发依据

本项目的基础分发单元是标准技能目录；插件清单属于外围包装。安装方式与宿主版本相关，不能把某一种插件安装命令当作所有 agent 的通用行为。

| 场景 | 已核查的支持方式 | 一手来源 |
|---|---|---|
| 解压后的本地技能仓库 | Skills CLI 支持 `npx skills add ./train-opt-skills` 形式的本地路径，再选择目标 agent | [vercel-labs/skills：Source Formats](https://github.com/vercel-labs/skills#source-formats) |
| Codex 项目级技能 | `.agents/skills/` | [Skills CLI 支持的 agent 路径表](https://github.com/vercel-labs/skills#supported-agents) |
| Claude Code 项目级技能 | `.claude/skills/` | [Skills CLI 支持的 agent 路径表](https://github.com/vercel-labs/skills#supported-agents) |
| Cursor 项目级技能 | `.agents/skills/` | [Skills CLI 支持的 agent 路径表](https://github.com/vercel-labs/skills#supported-agents) |
| 当前 OpenAI 可移植插件 | 根目录 `plugin.json`，声明 Agent Plugins schema；技能位于根目录 `skills/`，无需在可移植清单中重复声明 `skills` 字段 | [OpenAI：Package your plugin](https://developers.openai.com/plugins/build/plugins) |
| 旧 Codex 插件兼容布局 | `.codex-plugin/plugin.json` 仍作为兼容回退受支持，不代表新包必须采用该布局 | [OpenAI：Package your plugin](https://developers.openai.com/plugins/build/plugins) |

上述 Skills CLI 路径表用于说明该工具的安装目标。若手动安装，以所用 agent 版本的官方发现规则为准。`npx` 方式需要 Node.js/npm 及下载 Skills CLI 的网络；本项目不假设用户已创建或发布 GitHub 仓库。

## 来源和许可证边界

原训练指南是内容起点；GitHub 项目是工程结构参考，二者用途不同。原指南引用的框架文档及文章仍应保留对应归属。对第三方工具或训练框架的调用不意味着本项目重新授予其许可证。本项目未捆绑第三方技能代码；将来若复制具体实现，应先核对该文件及其上级目录的许可证，并保留要求的声明。
