# 可运行示例

此目录所有性能和样本数据均为**合成示例**，只演示工具契约，不代表模型训练实验。

从项目根目录运行：

```bash
python skills/optimize-training/scripts/collect_env.py
python skills/optimize-training/scripts/compare_benchmarks.py examples/baseline.json examples/candidate.json
python skills/optimize-training/scripts/audit_sample_ids.py examples/samples.jsonl --expected examples/expected_ids.json
```

旧版 baseline/candidate 中的 measurement 和 protocol 完全相同，只改变测量值。正式实验应填真实元数据，并采用重复测量。样本示例有两个 rank、各一个逻辑 worker，每个 epoch 完整覆盖同一预期身份集合。

生成一次最小张量示例（需 NumPy）：

```bash
python -c "import numpy as np; np.savez('/tmp/reference.npz', output=np.array([1.,2.])); np.savez('/tmp/candidate.npz', output=np.array([1.,2.000001]))"
python skills/optimize-training/scripts/compare_tensors.py /tmp/reference.npz /tmp/candidate.npz --atol 0.00001 --rtol 0.0001
```

临时路径和容差只用于演示；真实任务应使用自己的工作目录及预先确定的阈值。

## 提示词

### 数据瓶颈
> 使用 optimize-training 检查这个训练项目。GPU 利用率波动，先看数据等待、样本覆盖和预处理，再给出一个可验证的优化。

### 多卡
> 四卡速度只有单卡的 1.5 倍。按固定全局批大小检查扩展性，区分通信暴露、数据供给和尾批问题。

### 跨框架
> 两个实现都是 BF16，但输出不一致。固定参考权重与真实随机输入，先找第一处分歧，再检查梯度和更新。

### 无 GPU
> 当前只有 CPU。完成能做的源码与样本审计，准备 GPU 上的计时和正确性验收命令，不要假装已测到训练收益。

### 权重转换
> 转换后的 checkpoint strict=True 能加载。检查它是否满足部署要求，另列精确续训仍缺哪些状态。


## 受控执行变量（schema v2）

```bash
python skills/optimize-training/scripts/compare_benchmarks.py examples/controlled-baseline.json examples/controlled-candidate.json
```

这组也是合成示例。它保留训练语义不变量，在两份记录中一致声明执行设置变化；代码 revision 作为来源信息保留。具体允许字段、批次乘积和不可比行为见[实验契约](../docs/experiment-contract.md)。
