# Contributor instructions

Read README.md and the relevant skill references before edits. Keep SKILL.md short and resource links relative; do not flatten all references into the entry point.

Core scripts support Python 3.10+ with the standard library. NumPy is optional and isolated to tensor comparisons. Never read or print arbitrary environment variables, credentials, raw training data, or unsafe pickle checkpoints. Preserve unrelated user changes.

Test behavior at scientific failure boundaries: incomparable measurement protocols, weighted throughput, intentional versus accidental sampling repeats, nonfinite values, reference-asymmetric tolerances, and incomplete evidence. Do not change metadata simply to make comparisons pass.

After a change run the affected tests, validate_project.py, and build_release.py if packaging changed. Keep synthetic examples labeled. Do not claim measured training gains without actual artifacts. Match framework API statements to the installed version and primary documentation.

Keep root human documentation separate from the installable skill. Bump VERSION, plugin.json, CHANGELOG.md together when releasing. Never commit generated checkpoints, private traces, credentials, or dist archives to source control.


Treat specialist integrations as optional. Discovery is not execution; preserve reviewed source pins and show local divergence. Keep fixtures independent of hidden checks and do not leak treatment skills into baseline workdirs. Report actual A/B observations even when both arms succeed; never tune evaluation expectations to manufacture a benefit.
