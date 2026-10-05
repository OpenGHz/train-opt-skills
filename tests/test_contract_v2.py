"""Contract tests: declarations can permit implementation changes, never semantics."""
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(os.environ.get('TRAINING_SKILL_SCRIPTS', ROOT / 'skills' / 'optimize-training' / 'scripts'))
sys.path.insert(0, str(SCRIPTS))
from _common import InvalidInput
from compare_benchmarks import compare


def record():
    return {
        'schema_version': 2, 'synthetic': True,
        'comparison': {'track': 'system_optimization', 'changed_execution_settings': []},
        'measurement': {
            'unit': 'samples', 'scope': 'training_update', 'warmup_excluded': True,
            'synchronization': 'device_synchronized_window', 'includes_data_loading': True,
            'includes_validation': False, 'includes_checkpointing': False,
            'window': '100 optimizer updates', 'warmup': '20 optimizer updates',
        },
        'invariants': {
            'workload_id': 'toy-v2', 'dataset_id': 'fixture-v1', 'model_id': 'toy-mlp',
            'precision': 'fp32; TF32 off', 'input_shape': 'per sample: 128',
            'loss': 'mean squared error; mean over samples', 'optimizer': 'SGD lr=0.01',
            'sampling_and_augmentation': 'fixed order; no augmentation', 'initial_weights': 'fixture-init-v1',
            'hardware': 'synthetic-cpu', 'software': 'fixture-generator-v1',
            'global_batch_size': 32, 'views': 1, 'seed': 42,
        },
        'execution': {'microbatch_size': 32, 'gradient_accumulation_steps': 1,
                      'world_size': 1, 'data_parallel_size': 1, 'compile': 'disabled', 'num_workers': 0},
        'provenance': {'code_revision': 'baseline-fixture'},
        'runs': [{'elapsed_seconds': 2, 'effective_units': 100}] * 3,
    }


def declare(left, right, *keys):
    for item in (left, right):
        item['comparison']['changed_execution_settings'] = list(keys)


class ContractV2Tests(unittest.TestCase):
    def assertIncomparable(self, left, right, reason=None):
        report, code = compare(left, right)
        self.assertEqual(code, 1)
        self.assertEqual(report['status'], 'incomparable')
        self.assertEqual(report['correctness'], 'not_assessed')
        self.assertNotIn('speedup', report)
        if reason:
            self.assertEqual(report['reason'], reason)
        return report

    def test_declared_compile_and_worker_changes(self):
        before, after = record(), record()
        after['execution'].update(compile='default', num_workers=4)
        after['provenance']['code_revision'] = 'candidate-fixture'
        after['runs'] = [{'elapsed_seconds': 1, 'effective_units': 100}] * 3
        declare(before, after, 'compile', 'num_workers')
        report, code = compare(before, after)
        self.assertEqual(code, 0)
        self.assertEqual(report['speedup'], 2)
        self.assertEqual(report['correctness'], 'not_assessed')
        self.assertIn('not verification', ' '.join(report['warnings']))

    def test_changed_revision_is_provenance_not_invariant(self):
        before, after = record(), record()
        after['provenance']['code_revision'] = 'refactor-v2'
        self.assertEqual(compare(before, after)[1], 0)

    def test_undeclared_change_blocked(self):
        before, after = record(), record()
        after['execution'].update(compile='default', num_workers=4)
        declare(before, after, 'compile')
        report = self.assertIncomparable(before, after, 'declaration_mismatch')
        self.assertEqual(report['actual_changed_execution_settings'], ['compile', 'num_workers'])

    def test_unperformed_declared_change_blocked(self):
        before, after = record(), record()
        declare(before, after, 'num_workers')
        self.assertIncomparable(before, after, 'declaration_mismatch')

    def test_both_records_need_same_declaration(self):
        before, after = record(), record()
        after['execution']['compile'] = 'default'
        after['comparison']['changed_execution_settings'] = ['compile']
        self.assertIncomparable(before, after, 'declaration_mismatch')

    def test_resolution_views_loss_dataset_and_precision_always_fixed(self):
        changes = {'input_shape': 'per sample: 64', 'views': 2, 'loss': 'sum squared error',
                   'dataset_id': 'other-data', 'precision': 'bf16'}
        for field, value in changes.items():
            with self.subTest(field=field):
                before, after = record(), record()
                after['execution']['compile'] = 'default'
                declare(before, after, 'compile')
                after['invariants'][field] = value
                self.assertIncomparable(before, after, 'fixed_contract_changed')

    def test_global_batch_cannot_be_changed_with_consistent_product(self):
        before, after = record(), record()
        after['invariants']['global_batch_size'] = 64
        after['execution']['microbatch_size'] = 64
        declare(before, after, 'microbatch_size')
        self.assertIncomparable(before, after, 'fixed_contract_changed')

    def test_semantic_fields_cannot_be_permitted_changes(self):
        for name in ('input_shape', 'resolution', 'views', 'loss', 'dataset_id', 'global_batch_size', 'precision'):
            with self.subTest(name=name):
                before, after = record(), record()
                declare(before, after, name)
                with self.assertRaisesRegex(InvalidInput, 'cannot be declared'):
                    compare(before, after)

    def test_semantics_cannot_be_moved_into_execution(self):
        before, after = record(), record()
        for item in (before, after):
            item['execution']['precision'] = item['invariants'].pop('precision')
        with self.assertRaises(InvalidInput):
            compare(before, after)

    def test_nested_settings_cannot_hide_changes(self):
        before, after = record(), record()
        after['execution']['compile'] = {'mode': 'default', 'resolution': 64}
        declare(before, after, 'compile')
        with self.assertRaises(InvalidInput):
            compare(before, after)

    def test_microbatch_accumulation_preserves_global_product(self):
        before, after = record(), record()
        for item in (before, after):
            item['execution'].update(world_size=4, data_parallel_size=2, microbatch_size=16)
        after['execution'].update(microbatch_size=8, gradient_accumulation_steps=2)
        declare(before, after, 'microbatch_size', 'gradient_accumulation_steps')
        report, code = compare(before, after)
        self.assertEqual(code, 0)
        self.assertEqual(report['correctness'], 'not_assessed')
        self.assertIn('batch-dependent', ' '.join(report['warnings']))

    def test_bad_batch_product_rejected(self):
        before, after = record(), record()
        after['execution']['microbatch_size'] = 16
        declare(before, after, 'microbatch_size')
        with self.assertRaisesRegex(InvalidInput, 'global_batch_size must equal'):
            compare(before, after)

    def test_data_parallel_size_is_not_world_size(self):
        before, after = record(), record()
        for item in (before, after):
            item['execution'].update(world_size=4, data_parallel_size=2, microbatch_size=8)
        with self.assertRaisesRegex(InvalidInput, 'global_batch_size must equal'):
            compare(before, after)

    def test_topology_cannot_silently_change(self):
        before, after = record(), record()
        after['execution'].update(world_size=2, data_parallel_size=2, microbatch_size=16)
        declare(before, after, 'microbatch_size')
        self.assertIncomparable(before, after, 'parallel_topology_changed')

    def test_measurement_is_fixed_even_when_execution_changes_allowed(self):
        before, after = record(), record()
        after['execution']['compile'] = 'default'
        declare(before, after, 'compile')
        after['measurement']['includes_data_loading'] = False
        self.assertIncomparable(before, after, 'fixed_contract_changed')

    def test_additional_invariants_and_protocol_metadata_are_fixed(self):
        for section in ('measurement', 'invariants'):
            before, after = record(), record()
            before[section]['additional'] = True
            after[section]['additional'] = 1
            self.assertIncomparable(before, after, 'fixed_contract_changed')

    def test_explicit_values_required_on_both_sides(self):
        before, after = record(), record()
        after['execution']['activation_checkpointing'] = True
        declare(before, after, 'activation_checkpointing')
        self.assertIncomparable(before, after, 'execution_fields_differ')

    def test_task_and_scaling_tracks_do_not_report_equivalent_speedup(self):
        for track in ('task_change', 'scaling'):
            before, after = record(), record()
            for item in (before, after):
                item['comparison']['track'] = track
            self.assertIncomparable(before, after, 'unsupported_comparison_track')

    def test_scaling_with_topology_declared_is_explicitly_unsupported(self):
        before, after = record(), record()
        after['execution'].update(world_size=2, data_parallel_size=2, microbatch_size=16)
        declare(before, after, 'world_size', 'data_parallel_size', 'microbatch_size')
        for item in (before, after):
            item['comparison']['track'] = 'scaling'
        self.assertIncomparable(before, after, 'unsupported_comparison_track')

    def test_unknown_top_level_settings_rejected(self):
        before, after = record(), record()
        after['precision'] = 'bf16'
        with self.assertRaisesRegex(InvalidInput, 'unknown v2 root'):
            compare(before, after)

    def test_duplicate_declarations_rejected(self):
        before, after = record(), record()
        declare(before, after, 'compile', 'compile')
        with self.assertRaisesRegex(InvalidInput, 'duplicate'):
            compare(before, after)

    def test_examples_are_synthetic_and_comparable(self):
        inputs = [json.loads((ROOT / 'examples' / f'controlled-{name}.json').read_text())
                  for name in ('baseline', 'candidate')]
        self.assertTrue(all(item['synthetic'] for item in inputs))
        self.assertEqual(compare(*inputs)[1], 0)


if __name__ == '__main__':
    unittest.main()
