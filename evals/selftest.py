#!/usr/bin/env python3
"""Run synthetic harness tests. Does not invoke or evaluate an agent."""
from pathlib import Path
import sys
import unittest

if __name__ == '__main__':
    print('HARNESS SELFTEST ONLY: synthetic test doubles; no agent-performance claim.', flush=True)
    suite=unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),pattern='test_evals.py')
    outcome=unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if outcome.wasSuccessful() else 1)
