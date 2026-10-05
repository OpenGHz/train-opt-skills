# Train Opt Skills

[English](README.md) · [简体中文](README.zh-CN.md)

**让 Agent 在明确的正确性、测量和质量约束下，完成训练诊断、优化与验收。**

主技能 `optimize-training` 面向以 PyTorch 为主的训练项目，覆盖视觉、多模态和 VLA，也支持跨框架对齐。它组织现有专项能力，决定该检查什么、实施什么改动，以及依据哪些证据保留或回滚。

## 工作流程

1. 读取项目，区分正确性修复、系统优化和任务设置变化。
2. 固定训练语义，明确声明本次改变的执行设置。
3. 根据现有代码、日志或运行环境，选择诊断方法与可用专项技能。
4. 实施有依据的小改动，验证输入、数值、梯度和参数更新。
5. 比较性能、任务质量和恢复行为，给出保留、回滚或证据不足的结论。

无 GPU 时仍能做静态审计、已有产物分析和本项目的 CPU 测试。生成了测试、执行了测试、测试通过分别记录；最终训练验收需要对应硬件和任务证据。

## 安装与使用

解压后，在项目的父目录运行：

```bash
npx skills add ./train-opt-skills --skill optimize-training
```

选择 Agent 和作用域。需要 Node.js 与网络。离线复制可在项目根目录运行：

```bash
python tools/install_skill.py --target /你的训练项目/.agents/skills --dry-run
python tools/install_skill.py --target /你的训练项目/.agents/skills
```

Claude Code 使用 `/你的训练项目/.claude/skills`。已有目录不会自动覆盖。完整说明见[安装文档](docs/installation.md)。

示例请求：

> 使用 optimize-training 检查这个训练项目。先明确实验契约、定位瓶颈，再实施并验证一个可回滚优化。保持任务语义，未测量的结果明确标注。

## 专项技能复用

[可选集成](docs/integrations.md)支持发现已安装技能、核对来源、检查前提并准备交接。核心功能不依赖这些技能；不会自动下载、安装或执行它们。

| 能力 | 可选专项技能 |
| --- | --- |
| 分析性能 trace | meta-pytorch `profile-model` |
| 编译 graph break 定位与修复 | meta-pytorch `debug-graph-breaks` |
| 显存快照与增长分析 | meta-pytorch `analyze-memory-snapshot` |
| 系统级 CPU/GPU 分析 | 本地实际提供的 Nsight Systems skill pack |

登记了核查过的上游提交和内容哈希，但“找到技能”不等于“依赖齐全、已经执行”。专项分析完成后，仍回到本项目的训练正确性与端到端验收。

## 包含内容

- 精简技能入口、按需参考页，以及 **5 个训练优化决策案例**。
- **5 个命令行工具**：环境记录、性能比较、样本审计、张量比较和专项技能发现。
- [实验契约 v2](docs/experiment-contract.md)：固定不变量、声明实验变量，兼容旧版记录。
- 实验记录、优化案例和验收报告模板。
- [可执行技能评测](docs/evaluation.md)：真实错误输入、产物检查、无技能/有技能两组实验及运行器接口。
- 自动化测试、CI、离线安装、可重复 ZIP 打包与逐文件校验清单。
- 完整的[原始 v2 指南](docs/training_performance_optimization_guide_zh_v2.md)和[专项技能调研](docs/training_optimization_agent_skills_survey_zh.md)。

## 本地检查

Python 3.10+；张量比较需要 NumPy，部分评测任务也会注明 NumPy 前提。

```bash
python skills/optimize-training/scripts/compare_benchmarks.py examples/controlled-baseline.json examples/controlled-candidate.json
python skills/optimize-training/scripts/audit_sample_ids.py examples/samples.jsonl --expected examples/expected_ids.json
python skills/optimize-training/scripts/discover_integrations.py --root /已安装技能的实际目录
python -m unittest discover -s tests -v
python tools/validate_project.py
python tools/build_release.py
```

[工具契约](docs/tool-reference.md)解释输入、退出码与局限；[验证记录](docs/validation.md)列明实际执行范围。示例数据为合成数据，不代表 GPU 实测加速。

领域设计参考 NVIDIA、Meta PyTorch、AMD Primus 与 TensorRT-LLM；工程组织参考 Anthropic Skills、Superpowers 和 Vercel。详见[来源与设计](docs/design-references.md)。自有代码采用 MIT 许可，外部链接和实现保留各自权利；未打包第三方技能源码。
