# 安装与兼容性

本项目的可移植核心是 `skills/optimize-training/`。整个项目目录包含面向人的文档和开发工具，不应全部塞进单个技能目录。

## 方式一：Skills CLI

在解压目录的父目录运行：

```bash
npx skills add ./train-opt-skills --skill optimize-training
```

本地路径安装是 [Vercel Skills CLI](https://github.com/vercel-labs/skills) 支持的形式。安装过程选择 Agent 和作用域；需要 Node.js 及获取 CLI 的网络条件。项目尚无实际发布地址时不要使用虚构的 owner/repo。

## 方式二：离线复制

进入完整项目目录，指定“存放多个技能的父目录”：

```bash
python tools/install_skill.py --target /absolute/project/.agents/skills --dry-run
python tools/install_skill.py --target /absolute/project/.agents/skills
```

| 客户端 | 常见项目级目标 |
| --- | --- |
| Codex | `<project>/.agents/skills` |
| Claude Code | `<project>/.claude/skills` |
| Cursor（Skills CLI 映射） | `<project>/.agents/skills` |

以上目录映射核对日期为 2026-10-05，来源为 [Skills CLI 客户端列表](https://github.com/vercel-labs/skills)。不同宿主版本可能有不同发现方式，优先遵循当前客户端说明。安装器只负责复制，不替客户端完成注册或保证自动触发。

安装器默认不覆盖；已有同名目录或符号链接时停止。先检查旧版本并备份，再由你决定替换。卸载只需在确认路径后移除自己安装的 `optimize-training` 目录；工具不自动删除。

## 方式三：显式读取

不支持自动发现技能的 Agent，可让它读取：

> 阅读 /absolute/path/train-opt-skills/skills/optimize-training/SKILL.md，按照其中流程处理当前训练项目，并按需读取其相对路径资源。

这不代表所有客户端都支持相同的自动触发语法。Codex 中可使用 `$optimize-training`；通用提示直接写“使用 optimize-training”。

## 插件与 ZIP

根目录 `plugin.json` 按 [OpenAI 可移植插件文档](https://developers.openai.com/plugins/build/plugins) 提供名称、版本和描述，由标准 `skills/` 目录发现技能。它是可选的分发元数据；具体宿主是否支持此清单需按版本确认。

`python tools/build_release.py` 生成：

- `dist/train-opt-skills-0.2.0.zip`：完整源项目。
- `dist/optimize-training-0.2.0.zip`：以技能名为顶层目录的独立技能。
- `dist/SHA256SUMS`：两个压缩包的校验和。

仅当宿主提供自定义技能 ZIP 导入时使用独立技能包；完整项目包用于开发和本地安装。本项目不依赖远程 MCP、账号密钥或云计算服务。

