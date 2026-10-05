# Contributing

Use Python 3.10 or later. The base tools have no third-party runtime dependencies; install NumPy in a virtual environment for tensor comparisons and their tests.

1. Describe the user problem and the evidence needed to resolve it.
2. Edit the smallest relevant script or reference. Keep the skill entry concise.
3. Add a behavioral regression test for a real failure mode when changing executable behavior.
4. Run `python -m unittest discover -s tests -v` and `python tools/validate_project.py`.
5. For skill changes, run relevant prompts from `evals/cases.json` in fresh agent conversations, retaining the prompt, tool actions, output and rubric score.
6. Build archives with `python tools/build_release.py`; review the file manifest before release.

Separate numerical equivalence, task quality, and performance evidence. Mark unavailable GPU/distributed results as unrun. Do not add a universal tolerance, guaranteed speedup, fixed worker count, or version-independent FSDP recipe.

For a release, align VERSION, plugin.json and CHANGELOG.md. Existing installation directories are never overwritten automatically; back up and review changes before replacing them. External references stay attributed in NOTICE and docs/design-references.md.

