# Train Opt Skills

[English](README.md) · [简体中文](README.zh-CN.md)

**Coordinate training optimization with explicit correctness, measurement, and acceptance contracts.**

`optimize-training` is a portable Agent Skill for PyTorch-focused training projects, including vision, multimodal and VLA workloads. It routes specialist diagnosis, chooses controlled experiments, and requires evidence before retaining a change. It also supports cross-framework alignment.

## What the agent does

1. Inspect the project and distinguish a correctness repair, system optimization, or task change.
2. Freeze semantic invariants; declare the execution settings under test.
3. Use existing evidence, optional installed specialists, and small local checks to diagnose the bottleneck.
4. Make a bounded, reversible change and verify the affected inputs, numerics, gradients and updates.
5. Compare measured performance, task quality and checkpoint behavior; keep, revert, or report uncertainty.

No GPU is needed for static audits, existing artifact analysis, or the packaged tests. Actual training acceptance requires suitable hardware and task evidence. A generated test is labeled separately from an executed or passed test.

## Quick start

From the extracted project's parent directory:

```bash
npx skills add ./train-opt-skills --skill optimize-training
```

Select your agent and scope. This CLI route requires Node.js and network access. For an offline copy, from the repository:

```bash
python tools/install_skill.py --target /absolute/path/to/your-project/.agents/skills --dry-run
python tools/install_skill.py --target /absolute/path/to/your-project/.agents/skills
```

Existing content is never overwritten. See [installation](docs/installation.md).

Ask your agent:

> Use optimize-training to inspect this training project, fix the experimental contract, diagnose the bottleneck, and validate one reversible improvement. Preserve task semantics and report unmeasured results explicitly.

## Optional specialists

The core skill works independently. [Integration support](docs/integrations.md) discovers explicitly supplied local skill roots, records provenance and prerequisites, and defines artifact handoffs. It does not install or execute another skill automatically.

| Specialist | Input and role |
| --- | --- |
| meta-pytorch `profile-model` | Existing PyTorch profiler traces; kernel, idle and communication analysis |
| meta-pytorch `debug-graph-breaks` | Logs/code; scoped `torch.compile` diagnosis and repair |
| meta-pytorch `analyze-memory-snapshot` | Existing memory snapshots; allocation and growth analysis |
| Local Nsight Systems skill pack | Optional deeper system profiling if actually present |

The registry records a reviewed upstream commit and content hashes. Skill availability is not runtime readiness. Every specialist result returns through the core correctness and performance gates. No third-party skill implementation is vendored.

## Included

| Component | Purpose |
| --- | --- |
| [SKILL.md](skills/optimize-training/SKILL.md) | Short workflow and resource routing |
| Topic references and five decision cases | Applicability, counter-signals, verification, interactions and rollback |
| Five CLI tools | Environment inventory, controlled benchmark comparison, sample audit, NPZ parity, integration discovery |
| [Experiment contract](docs/experiment-contract.md) | Fixed invariants versus explicitly declared execution variables; legacy v1 support |
| Experiment, case and report templates | Explicit evidence states and reproducible decisions |
| [Executable evaluations](docs/evaluation.md) | Task fixtures, deterministic artifact checks, baseline/treatment preparation and runtime protocol |
| Tests, CI, installer and release builder | Validated local package and reproducible ZIP manifests |
| Source guide and survey | Complete supplied [v2 guide](docs/training_performance_optimization_guide_zh_v2.md) and [specialist survey](docs/training_optimization_agent_skills_survey_zh.md) |

## Try locally

Python 3.10+ is required. The core tools use the standard library; NPZ comparison needs NumPy. Some evaluation tasks use NumPy as stated in their instructions.

```bash
python skills/optimize-training/scripts/compare_benchmarks.py examples/controlled-baseline.json examples/controlled-candidate.json
python skills/optimize-training/scripts/audit_sample_ids.py examples/samples.jsonl --expected examples/expected_ids.json
python skills/optimize-training/scripts/discover_integrations.py --root /path/to/installed/skills
python -m unittest discover -s tests -v
python tools/validate_project.py
python tools/build_release.py
```

Synthetic example numbers are not training results. See [tool reference](docs/tool-reference.md), [evaluation guide](docs/evaluation.md), and the actual [validation record](docs/validation.md).

## Project background

The supplied v2 guide defines the acceptance principles. NVIDIA's training workflow, Meta PyTorch's specialist skills/evaluation structure, AMD Primus's validation tiers, and TensorRT-LLM's casebook informed the domain design. Anthropic Skills, Superpowers and Vercel informed packaging. See [design references](docs/design-references.md) for scope and provenance.

[Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md) · [MIT license](LICENSE) · [Attribution](NOTICE)
