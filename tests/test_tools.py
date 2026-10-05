"""Run: python -m unittest discover -s tests -v from the project root.

Set TRAINING_SKILL_SCRIPTS to override the install location during development.
NumPy-dependent tests skip when NumPy is unavailable.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
CANDIDATES = [HERE.parent / 'skills' / 'optimize-training' / 'scripts',
              HERE.parent / 'scripts']
SCRIPTS = Path(os.environ['TRAINING_SKILL_SCRIPTS']) if 'TRAINING_SKILL_SCRIPTS' in os.environ else next(p for p in CANDIDATES if p.exists())
sys.path.insert(0, str(SCRIPTS))
from _common import InvalidInput, load_json
from compare_benchmarks import compare
from audit_sample_ids import audit


def record(runs=None):
    return {'schema_version':1, 'synthetic':True,
            'measurement':{'unit':'samples', 'scope':'training_update', 'warmup_excluded':True,
                           'synchronization':'device_synchronized_window', 'includes_data_loading':True,
                           'includes_validation':False, 'includes_checkpointing':False},
            'protocol':{'workload_id':'toy-v1','dataset_id':'fixture-v1','model_id':'toy-mlp',
                        'precision':'fp32','global_batch_size':32,'world_size':1,
                        'gradient_accumulation_steps':1,'input_shape':'32x128',
                        'hardware':'synthetic-cpu','software':'fixture-generator-v1','seed':42},
            'runs':runs or [{'elapsed_seconds':2.0,'effective_units':100}]}


class TestHelpers(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()
    def save(self, name, value):
        path = self.directory / name
        path.write_text(json.dumps(value), encoding='utf-8')
        return path
    def invoke(self, script, *args):
        return subprocess.run([sys.executable,str(SCRIPTS/script),*map(str,args)], capture_output=True,text=True)
    def samples(self, rows):
        path = self.directory/'samples.jsonl'
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        return path


class ToolTests(TestHelpers):
    def test_aggregation_uses_total_work_total_time(self):
        before = record([{'elapsed_seconds':1,'effective_units':100},{'elapsed_seconds':9,'effective_units':100}])
        after = record([{'elapsed_seconds':2,'effective_units':100},{'elapsed_seconds':3,'effective_units':100}])
        report, code = compare(before,after)
        self.assertEqual(code,0)
        self.assertEqual(report['baseline']['throughput'],20)
        self.assertAlmostEqual(report['baseline']['median_run_throughput'],(100+100/9)/2)
        self.assertEqual(report['speedup'],2)
        self.assertEqual(report['reduction_pct'],50)
        self.assertEqual(report['correctness'],'not_assessed')
        self.assertTrue(report['warnings'])
    def test_protocol_and_measurement_mismatch(self):
        for section,field,value in [('measurement','unit','tokens'),('protocol','global_batch_size',64),('protocol','precision','bf16')]:
            before, after = record(),record()
            after[section][field]=value
            result,code = compare(before,after)
            self.assertEqual(code,1)
            self.assertNotIn('speedup',result)
    def test_extra_protocol_fields_are_compared_strictly(self):
        before,after=record(),record()
        before['protocol']['additional']=True
        after['protocol']['additional']=1
        self.assertEqual(compare(before,after)[1],1)
    def test_invalid_measurements(self):
        for bad in [0,-1,True,float('nan'),float('inf'),'1']:
            data=record();data['runs'][0]['elapsed_seconds']=bad
            with self.subTest(bad=bad),self.assertRaises(InvalidInput):
                compare(data,record())
    def test_required_fields(self):
        data=record();del data['protocol']['seed']
        with self.assertRaises(InvalidInput):compare(data,record())
    def test_duplicate_json_keys_and_overflow_rejected(self):
        path=self.directory/'bad.json'
        for raw in ['{"x":1,"x":2}','{"x":1e999}','{"x":NaN}']:
            path.write_text(raw)
            with self.assertRaises(InvalidInput):load_json(path)
    def test_oversized_json_integer_is_malformed_without_traceback(self):
        limit = getattr(sys, 'get_int_max_str_digits', lambda: 0)()
        if not limit:
            self.skipTest('Python integer-string digit guard is unavailable or disabled')
        path = self.directory/'huge-id.jsonl'
        path.write_text('{"epoch":0,"rank":0,"worker":0,"sample_id":' + '9'*(limit+1) + '}\n')
        with self.assertRaises(InvalidInput):
            load_json(path)
        result = self.invoke('audit_sample_ids.py', path)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertNotIn('Traceback', result.stderr)
        self.assertIn('Invalid JSON', result.stderr)
    def test_output_cannot_replace_input(self):
        path=self.save('base.json',record())
        original=path.read_bytes()
        result=self.invoke('compare_benchmarks.py',path,path,'--output',path)
        self.assertEqual(result.returncode,2)
        self.assertEqual(path.read_bytes(),original)
        link=self.directory/'alias.json';link.symlink_to(path)
        result=self.invoke('compare_benchmarks.py',path,path,'--output',link)
        self.assertEqual(result.returncode,2)
    def test_repeats_scoped_by_epoch_across_owners(self):
        rows=[{'epoch':0,'rank':0,'worker':0,'sample_id':'a'},
              {'epoch':1,'rank':0,'worker':0,'sample_id':'a'}]
        report,code=audit(self.samples(rows),['a'])
        self.assertEqual(code,0)
        rows.append({'epoch':1,'rank':1,'worker':0,'sample_id':'a'})
        report,code=audit(self.samples(rows),['a'])
        self.assertEqual(code,1)
        self.assertEqual(report['epochs'][1]['cross_owner_repeated_ids'],1)
        self.assertEqual(audit(self.samples(rows),['a'],True)[1],0)
    def test_allow_repeats_does_not_excuse_coverage(self):
        rows=[{'epoch':0,'rank':0,'worker':0,'sample_id':'a'}]*2
        report,code=audit(self.samples(rows),['a','b'],True)
        self.assertEqual(code,1)
        self.assertEqual(report['epochs'][0]['missing_count'],1)
    def test_unknown_coverage_and_typed_ids(self):
        rows=[{'epoch':0,'rank':0,'worker':0,'sample_id':1},
              {'epoch':0,'rank':0,'worker':0,'sample_id':'1'}]
        report,code=audit(self.samples(rows))
        self.assertEqual(code,0)
        self.assertEqual(report['coverage'],'unknown')
        self.assertIsNone(report['epochs'][0]['missing_count'])
    def test_invalid_owner_and_expected(self):
        for field,bad in [('rank',1.2),('worker',-1),('epoch',True)]:
            row={'epoch':0,'rank':0,'worker':0,'sample_id':'a'};row[field]=bad
            with self.subTest(field=field),self.assertRaises(InvalidInput):audit(self.samples([row]))
        with self.assertRaises(InvalidInput):audit(self.samples([]))
        row={'epoch':0,'rank':0,'worker':0,'sample_id':'a'}
        with self.assertRaises(InvalidInput):audit(self.samples([row]),['a','a'])
    def test_collect_env_does_not_expose_env_or_import_torch(self):
        script='import sys;sys.path.insert(0,'+repr(str(SCRIPTS))+');import collect_env; r=collect_env.collect();assert "torch" not in sys.modules;import json;print(json.dumps(r))'
        env=dict(os.environ,TRAINING_SECRET='do-not-print-this-secret')
        result=subprocess.run([sys.executable,'-c',script],env=env,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertNotIn('do-not-print-this-secret',result.stdout)
        self.assertFalse(json.loads(result.stdout)['environment_variables_collected'])
    def test_all_help_commands(self):
        for script in ['collect_env.py','compare_benchmarks.py','audit_sample_ids.py','compare_tensors.py']:
            result=self.invoke(script,'--help')
            self.assertEqual(result.returncode,0,result.stderr)


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'NumPy optional dependency unavailable')
class TensorTests(TestHelpers):
    def snapshots(self,left,right):
        import numpy as np
        a=self.directory/'reference.npz';b=self.directory/'candidate.npz'
        np.savez(a,**left);np.savez(b,**right)
        return a,b
    def tensor_compare(self,left,right,atol=0,rtol=0):
        from compare_tensors import compare
        a,b=self.snapshots(left,right)
        return compare(a,b,atol,rtol)
    def test_asymmetric_relative_tolerance(self):
        self.assertEqual(self.tensor_compare({'x':[1.]},{'x':[2.]},rtol=.6)[1],1)
        self.assertEqual(self.tensor_compare({'x':[2.]},{'x':[1.]},rtol=.6)[1],0)
    def test_nan_and_infinity_fail_even_when_identical(self):
        for value in [float('nan'),float('inf')]:
            report,code=self.tensor_compare({'x':[value]},{'x':[value]})
            self.assertEqual(code,1)
            self.assertEqual(report['tensors'][0]['nonfinite_pairs'],1)
    def test_zero_reference_relative_error(self):
        report,code=self.tensor_compare({'x':[0.]},{'x':[.01]},atol=.1)
        self.assertEqual(code,0)
        self.assertTrue(report['tensors'][0]['relative_error_unbounded'])
        self.assertIsNone(report['tensors'][0]['max_relative_error'])
    def test_keys_and_shapes(self):
        self.assertEqual(self.tensor_compare({'x':[1]},{'y':[1]})[1],1)
        self.assertEqual(self.tensor_compare({'x':[1]},{'x':[[1]]})[1],1)
    def test_pickle_and_boolean_rejected(self):
        import numpy as np
        for value in [np.array([{}],dtype=object),np.array([True])]:
            with self.assertRaises(InvalidInput):self.tensor_compare({'x':value},{'x':value})
    def test_integer_subtraction_does_not_wrap(self):
        import numpy as np
        a=np.array([2**63-1],dtype=np.int64);b=np.array([-2**63],dtype=np.int64)
        report,code=self.tensor_compare({'x':a},{'x':b})
        self.assertEqual(code,1)
        self.assertGreater(report['tensors'][0]['max_abs_error'],1e19)
    def test_narrow_longdouble_rejects_precision_unsafe_integers(self):
        import numpy as np
        from unittest.mock import patch
        left = np.array([2**53], dtype=np.int64)
        right = np.array([2**53 + 1], dtype=np.int64)
        pairs = [(left, right), (right, left.astype(np.float64)),
                 (left.astype(np.float64), right)]
        for reference, candidate in pairs:
            with self.subTest(reference=reference.dtype, candidate=candidate.dtype):
                with patch.object(np, 'longdouble', np.float64):
                    with self.assertRaisesRegex(InvalidInput, 'precision loss'):
                        self.tensor_compare({'x':reference}, {'x':candidate})
    def test_narrow_longdouble_preserves_safe_integer_comparison(self):
        import numpy as np
        from unittest.mock import patch
        with patch.object(np, 'longdouble', np.float64):
            self.assertEqual(self.tensor_compare({'x':[2**53-1]}, {'x':[2**53]})[1], 1)
    def test_non_tensor_zip_member_is_malformed_without_traceback(self):
        import zipfile
        a,b = self.snapshots({'x':[1]}, {'x':[1]})
        # Also reject unmatched malformed entries, before key mismatch reporting.
        for path in (a,b):
            with zipfile.ZipFile(path, 'a') as archive:
                archive.writestr('readme.txt', 'not a tensor')
            result = self.invoke('compare_tensors.py', a, b, '--atol', '0', '--rtol', '0')
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertNotIn('Traceback', result.stderr)
            self.assertIn('only .npy', result.stderr)
    def test_invalid_npy_member_is_malformed_without_traceback(self):
        import zipfile
        a,b = self.snapshots({'x':[1]}, {'x':[1]})
        for path in (a,b):
            with zipfile.ZipFile(path, 'a') as archive:
                archive.writestr('bad.npy', 'not a valid npy array')
        result = self.invoke('compare_tensors.py', a, b, '--atol', '0', '--rtol', '0')
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertNotIn('Traceback', result.stderr)
    def test_requires_explicit_finite_tolerances(self):
        a,b=self.snapshots({'x':[1]},{'x':[1]})
        self.assertEqual(self.invoke('compare_tensors.py',a,b).returncode,2)
        self.assertEqual(self.invoke('compare_tensors.py',a,b,'--atol','nan','--rtol','0').returncode,2)
    def test_tensor_cli_match(self):
        a,b=self.snapshots({'x':[1,2,3]},{'x':[1,2,3]})
        result=self.invoke('compare_tensors.py',a,b,'--atol','0','--rtol','0')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'],'match')

if __name__=='__main__':unittest.main()
