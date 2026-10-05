# Failed candidate: batch statistics in per-image augmentation

- **ID:** `augmentation-batch-statistics`
- **Maturity:** `synthetic_counterexample`
- **Empirical status:** Constructed mathematical failure; no measured training run or speed claim.
- **Track:** Reject a non-equivalent system candidate; repair before measuring it as an optimization.

## Applicability and counter-signals

Select this case when a vectorized image transform unexpectedly makes one image depend on its batch neighbors. Inspect random-parameter granularity and reduction dimensions separately.

**Counter-signals:** the task deliberately specifies a batch-coupled transform; or a sequence/view group deliberately shares parameters. This case does not require independent per-image randomness for every temporal or multiview task. Shared parameters do not, by themselves, justify pooling image statistics.

## Mechanism and failed candidate

Consider a deliberately simplified grayscale contrast transform, with a fixed per-image factor `a_i` and no clipping:

`y_i = a_i * (x_i - mean(x_i)) + mean(x_i)`

For `(B, 1, H, W)`, the intended mean reduces spatial/image dimensions while preserving `B`. Replacing it with `mean(x)` over the entire batch changes the function even if the code is fully vectorized and every `a_i` is still independent.

Synthetic arithmetic: image A contains only `0.2`, image B only `0.8`, and both factors are `0.5`. Their own means give unchanged outputs `0.2` and `0.8`. The pooled mean is `0.5`, yielding `0.35` and `0.65`. These values are an analytic counterexample, not a benchmark or a claim about a particular library implementation.

## Allowed repair and correctness checks

Restore the baseline's exact reduction axes, channel/luminance convention, dtype, range, clipping and operation order; do not substitute this simplified formula for a library transform. Vectorize only after these semantics are explicit. Preserve the required sampling/sharing granularity, using broadcast shapes that represent it.

Replay saved images and enhancement factors. Hold image A and its factors fixed while replacing only unrelated batch neighbors; its output must remain unchanged when the contract specifies independent images. Also compare against the reference loop and test factors, channels, ranges, and supported layouts. Equal seeds or visually plausible output cannot replace these checks. Then test downstream forward/loss/update behavior and required task quality.

## Measurement, rollback and interactions

Reject the pooled-statistics candidate before interpreting its speed as valid training acceleration. Restore the reference transform immediately; retain the failing input and output artifact. Benchmark a corrected vectorized candidate with device completion accounted for and with real end-to-end training. Include preprocessing placement, transfers and memory; report a transform microbenchmark separately. Recheck temporal/view sharing, random streams, and shapes after batching or compilation changes.

## Sources and version boundary

Original counterexample and workflow based on the user's v2 guide, section 6.3. [Torchvision `adjust_contrast`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.functional.adjust_contrast.html) documents its image layout and factor interface; it does not establish that this simplified transform matches every backend. Checked 2026-10-05 (page identifies Torchvision 0.29). Inspect the actual version's implementation to determine reduction and luminance details before translating a real transform.
