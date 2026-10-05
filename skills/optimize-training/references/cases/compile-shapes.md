# Compile boundaries and varying shapes

- **ID:** `compile-shapes`
- **Maturity:** `engineering_pattern`
- **Empirical status:** No project measurement supplied; no speedup claimed.
- **Track:** System optimization unless input content or training organization changes beyond contract.

## Applicability and counter-signals

Select this case when compiler diagnostics and traces show that a repeated hot computation suffers from graph breaks, recompilation, or excessive specialization. Save the location and trigger of each relevant break or recompilation, together with representative shapes and control-flow branches.

**Counter-signals:** the hot region is data loading or network waiting; input shapes rarely recur; compilation startup dominates the entire short job; or the break is outside a meaningful hot path. Fewer graphs alone does not establish a benefit.

## Mechanism and allowed changes

Test a bounded repeated module first. Candidate changes include moving diagnostic-only Python work outside that region, rewriting a supported operation without altering behavior, or selecting an appropriate dynamic-shape policy. Preserve an eager reference path.

Shape bucketing is a separate declared variable: retain valid tokens/views, mask and position semantics, sample weights, and the specified sampling/update policy. If regrouping changes update composition, evaluate it under an explicit training-policy contract. Truncating valid sequences, reducing effective resolution, or removing real views is a task change. Do not hide unsupported branches or data-dependent work merely to obtain a single graph.

## Correctness checks

Compare reference and candidate using saved inputs, initial parameters, actual stochastic inputs, masks, and mode. Cover supported shape families, padding and edge cases; then compare forward outputs, loss normalization, gradients, and a full optimizer update with predeclared tolerances. Test stateful behavior and resume if affected. Document random-path differences explicitly: equal seeds do not establish eager/compiled draw parity. A successful graph capture does not certify numerical behavior or task quality.

## Measurement and decision

Use diagnostic compiler/profiler runs to locate the issue, then lightweight acceptance runs. Record cold-start compilation, all subsequent recompilations, warm-window throughput, peak memory, and complete job duration over a representative shape distribution. Keep the effective-work denominator constant. A long-run estimate may help plan, but report it as an estimate; include additional compilation cost and validate against the intended job length. Keep only a candidate that passes its required checks and benefits the user's workload.

## Rollback and interactions

Restore eager execution or the previous compile configuration and code boundary; retain diagnostics. Re-evaluate after changes to microbatching, dynamic input cropping, attention backend, precision, or checkpoint boundaries. Do not combine all these changes in one attribution experiment.

## Sources and version boundary

Engineering workflow: user's v2 guide, sections 3, 4, 7 and 9. Primary mechanisms: [torch.compile programming model](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.html) and [CUDA timing semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html). Checked 2026-10-05; these pages identify PyTorch 2.14. Compiler logging, shape APIs, and supported operations are version dependent; inspect the installed version's official documentation.
