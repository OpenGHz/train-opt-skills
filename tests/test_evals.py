"""Synthetic test doubles verify the harness, never model or skill performance."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('eval_runner', PROJECT / 'evals/run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
SOLUTIONS = {
'async_timing': '''def measure(step, synchronize, clock, steps=10, warmup=2):
    if steps <= 0 or warmup < 0: raise ValueError('window')
    for _ in range(warmup): step()
    synchronize()
    start = clock()
    for _ in range(steps): step()
    synchronize()
    return clock() - start
''',
'duplicate_shards': '''def shard(data, rank, world_size, worker_id=0, num_workers=0):
    workers = max(1, num_workers)
    if world_size < 1 or not 0 <= rank < world_size or num_workers < 0 or not 0 <= worker_id < workers: raise ValueError('topology')
    return data[rank * workers + worker_id::world_size * workers]
''',
'augmentation_stats': '''import math
def normalize(images):
    out = []
    for row in images:
        if not row: raise ValueError('empty image')
        mean = sum(row) / len(row)
        var = sum((x-mean)**2 for x in row) / len(row)
        out.append([(x-mean)/math.sqrt(var+1e-5) for x in row])
    return out
''',
'weight_reorder': '''def convert_qkv(weight, num_heads):
    if num_heads < 1 or len(weight) != num_heads * 3 or not weight or not weight[0] or any(len(r) != len(weight[0]) for r in weight): raise ValueError('layout')
    return [weight[3*h+p][:] for p in range(3) for h in range(num_heads)]
'''}
ASSESSMENT = {'observed_step_rate_ratio': 2.0, 'equivalent_training_speedup_established': False,
              'changed_fields': ['real_cameras', 'image_resolution'], 'task_quality_verified': False,
              'time_to_target_speedup': None}
AUDIT_SCRIPT = '''import argparse, collections, json
from pathlib import Path
p=argparse.ArgumentParser()
for name in ['samples','expected','output']: p.add_argument('--'+name,required=True)
a=p.parse_args()
ids=[json.loads(line)['sample_id'] for line in Path(a.samples).read_text().splitlines() if line.strip()]
counts=collections.Counter(ids)
expected=json.loads(Path(a.expected).read_text())
Path(a.output).write_text(json.dumps(dict(records=len(ids),unique_ids=len(counts),missing_ids=sorted(set(expected)-set(ids)),repeated_ids=sorted(k for k,v in counts.items() if v>1))))
'''

def response(case):
    return {'schema_version': 1, 'case_id': case, 'status': 'fixed', 'summary': 'Synthetic harness test double.',
            'unverified': ['This is not an agent evaluation or a GPU experiment.']}

class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
    def prepare(self, case, variant='baseline', suffix=''):
        return runner.prepare(case, variant, self.root / (case + variant + suffix))
    def write_response(self, workdir, case):
        runner.write_json(workdir / 'response.json', response(case))
    def test_fixture_isolation_and_treatment_skill(self):
        baseline = self.prepare('async_timing')
        treatment = self.prepare('async_timing', 'treatment')
        self.assertFalse((baseline / '_skill').exists())
        self.assertTrue((treatment / '_skill/optimize-training/SKILL.md').is_file())
        for workspace in [baseline, treatment]:
            for name in ['checkers.py', 'rubric.json', 'tests']:
                self.assertFalse((workspace / name).exists())
            self.assertTrue((workspace / 'TASK.md').is_file())
    def test_existing_and_source_destinations_rejected(self):
        path = self.prepare('async_timing')
        with self.assertRaises(FileExistsError):
            runner.prepare('async_timing', 'baseline', path)
        with self.assertRaises(ValueError):
            runner.prepare('async_timing', 'baseline', PROJECT / 'should-not-exist-eval-test')
    def test_no_response_cannot_pass(self):
        workdir = self.prepare('protocol_change')
        runner.write_json(workdir / 'assessment.json', ASSESSMENT)
        self.assertFalse(runner.score('protocol_change', workdir)['passed'])
    def test_original_bugs_fail_semantic_checks(self):
        for case in SOLUTIONS:
            with self.subTest(case=case):
                workdir = self.prepare(case)
                self.write_response(workdir, case)
                result = runner.score(case, workdir)
                self.assertFalse(result['passed'])
                self.assertTrue(any(not c['passed'] for c in result['checks'] if c['name'] != 'response_schema'))
    def test_correct_repairs_pass_on_unseen_inputs(self):
        for case, code in SOLUTIONS.items():
            with self.subTest(case=case):
                workdir = self.prepare(case)
                (workdir / 'pipeline.py').write_text(code)
                self.write_response(workdir, case)
                result = runner.score(case, workdir)
                self.assertTrue(result['passed'], result)
    def test_protocol_conclusion_and_input_tampering(self):
        workdir = self.prepare('protocol_change')
        self.write_response(workdir, 'protocol_change')
        runner.write_json(workdir / 'assessment.json', ASSESSMENT)
        self.assertTrue(runner.score('protocol_change', workdir)['passed'])
        runner.write_json(workdir / 'assessment.json', {**ASSESSMENT, 'equivalent_training_speedup_established': True})
        self.assertFalse(runner.score('protocol_change', workdir)['passed'])
        runner.write_json(workdir / 'assessment.json', ASSESSMENT)
        (workdir / 'runs.json').write_text('{}')
        self.assertFalse(runner.score('protocol_change', workdir)['passed'])
    def test_optional_integration_requires_reproducible_script(self):
        workdir = self.prepare('optional_integration')
        self.write_response(workdir, 'optional_integration')
        runner.write_json(workdir / 'audit.json', {'records':4,'unique_ids':2,'missing_ids':['c','d'],'repeated_ids':['a','b']})
        runner.write_json(workdir / 'integration-result.json', {'integration_status':'unavailable','external_skill_executed':False,'data_contract_passed':False,'gpu_performance_measured':False})
        self.assertFalse(runner.score('optional_integration', workdir)['passed'])
        (workdir / 'audit.py').write_text(AUDIT_SCRIPT)
        self.assertTrue(runner.score('optional_integration', workdir)['passed'])
    def test_runtime_protocol_preserves_logs_and_env(self):
        for variant in ['baseline','treatment']:
            workdir = self.prepare('protocol_change', variant)
            program = ('import os,json,pathlib,sys; p=pathlib.Path(os.environ["TASK_WORKDIR"]); '
                       f'assert ("TASK_SKILL_PATH" in os.environ) == {variant == "treatment"}; '
                       'assert pathlib.Path(os.environ["TASK_PROMPT_PATH"]).is_file(); '
                       f'(p/"response.json").write_text({json.dumps(response("protocol_change"))!r}); '
                       f'(p/"assessment.json").write_text({json.dumps(ASSESSMENT)!r}); '
                       'print("synthetic stdout"); print("synthetic stderr",file=sys.stderr)')
            runtime = runner.execute([sys.executable,'-c',program],workdir,variant,5)
            self.assertEqual(runtime['status'],'completed')
            self.assertTrue(runner.score('protocol_change', workdir)['passed'])
            self.assertIn('synthetic stdout',(workdir/'.eval/stdout.txt').read_text())
            self.assertIn('synthetic stderr',(workdir/'.eval/stderr.txt').read_text())
    def test_timeout_launch_error_and_missing_result_fail(self):
        workdir = self.prepare('protocol_change')
        timed = runner.execute([sys.executable,'-c','import time;time.sleep(3)'],workdir,'baseline',.05)
        self.assertEqual(timed['status'],'timeout')
        self.assertFalse(runner.score('protocol_change',workdir)['passed'])
        launch = runner.execute([str(self.root/'missing-executable')],workdir,'baseline',1)
        self.assertEqual(launch['status'],'launch_failed')
        result = runner.execute([sys.executable,'-c','pass'],workdir,'baseline',1)
        self.assertEqual(result['returncode'],0)
        self.assertFalse(runner.score('protocol_change',workdir)['passed'])
    def test_run_suite_records_failed_attempts(self):
        result = runner.run_suite([sys.executable,'-c','pass'],'both',self.root/'suite',1,['protocol_change'])
        self.assertEqual(result['total_attempts'],2)
        self.assertEqual(result['passed_attempts'],0)

    def test_nonfinite_timeouts_never_launch_runtime(self):
        workdir = self.prepare('protocol_change')
        with mock.patch.object(runner.subprocess, 'Popen') as popen:
            for timeout in [float('nan'), float('inf'), -float('inf'), 0., -1.]:
                with self.subTest(timeout=timeout):
                    with self.assertRaisesRegex(ValueError, 'finite and positive'):
                        runner.execute([sys.executable, '-c', 'pass'], workdir, 'baseline', timeout)
                    with self.assertRaisesRegex(ValueError, 'finite and positive'):
                        runner.score('protocol_change', workdir, checker_timeout=timeout)
            popen.assert_not_called()

    def test_unexpected_communicate_exception_reaps_child(self):
        workdir = self.prepare('protocol_change')
        real_popen = runner.subprocess.Popen
        children = []
        def factory(*args, **kwargs):
            child = real_popen(*args, **kwargs)
            children.append(child)
            original_communicate = child.communicate
            calls = []
            def communicate(*a, **kw):
                if not calls:
                    calls.append(True)
                    raise ValueError('synthetic communication failure')
                return original_communicate(*a, **kw)
            child.communicate = communicate
            return child
        with mock.patch.object(runner.subprocess, 'Popen', side_effect=factory):
            result = runner.execute([sys.executable, '-c', 'import time;time.sleep(3)'], workdir, 'baseline', 1)
        self.assertEqual(result['status'], 'runtime_error')
        self.assertIsNotNone(children[0].poll())
        self.assertIn('synthetic communication failure', (workdir / '.eval/stderr.txt').read_text())

    def test_participant_manifest_cannot_disable_input_validation(self):
        for replacement in [{}, {'runs.json': '0' * 64}]:
            workdir = self.prepare('protocol_change', suffix=str(len(replacement)))
            self.write_response(workdir, 'protocol_change')
            runner.write_json(workdir / 'assessment.json', ASSESSMENT)
            (workdir / 'runs.json').write_text('{}')
            path = workdir / '.eval/manifest.json'
            manifest = json.loads(path.read_text())
            manifest['immutable_inputs'] = replacement
            runner.write_json(path, manifest)
            result = runner.score('protocol_change', workdir)
            self.assertFalse(result['passed'])
            states = {check['name']: check['passed'] for check in result['checks']}
            self.assertFalse(states['manifest_inputs_match_harness'])
            self.assertFalse(states['immutable_inputs_preserved'])

    @unittest.skipUnless(hasattr(os, 'symlink'), 'symlinks unsupported')
    def test_metadata_symlinks_cannot_overwrite_external_files(self):
        workdir = self.prepare('protocol_change')
        target = self.root / 'valuable.json'
        target.write_text('preserve me')
        score_path = workdir / '.eval/score.json'
        score_path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            runner.score('protocol_change', workdir)
        self.assertEqual(target.read_text(), 'preserve me')
        score_path.unlink()
        metadata = workdir / '.eval'
        metadata.rename(workdir / 'real-eval')
        metadata.symlink_to(workdir / 'real-eval', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'real directory'):
            runner.score('protocol_change', workdir)

if __name__ == '__main__': unittest.main()
