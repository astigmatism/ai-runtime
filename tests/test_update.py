import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from runtime.config import read
from runtime.system import atomic_json
from runtime.update import REMOTE, Updater

A, B = 'a' * 40, 'b' * 40


class Harness:
    def __init__(self, root):
        self.root = root; self.calls = []; self.head = A; self.remote_head = B; self.upstream_head = A
        self.branch = 'main'; self.upstream = 'origin/main'; self.remote = REMOTE
        self.dirty = ''; self.diverged = False; self.fail_build = False; self.fail_health = False

    def __call__(self, args, **kw):
        args = tuple(args); self.calls.append(args)
        stdout, code = '', 0
        values = {
            ('git', 'rev-parse', '--show-toplevel'): str(self.root),
            ('git', 'symbolic-ref', '--quiet', '--short', 'HEAD'): self.branch,
            ('git', 'rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{upstream}'): self.upstream,
            ('git', 'remote', 'get-url', 'origin'): self.remote,
            ('git', 'status', '--porcelain', '--untracked-files=normal'): self.dirty,
            ('git', 'rev-parse', 'HEAD'): self.head,
            ('git', 'rev-parse', 'refs/remotes/origin/main'): self.upstream_head,
            ('git', 'rev-parse', 'FETCH_HEAD'): self.remote_head,
        }
        if args in values: stdout = values[args]
        elif args[:3] == ('git', 'merge-base', '--is-ancestor'): code = int(self.diverged)
        elif args[:3] == ('git', 'merge', '--ff-only'): self.head = args[-1]
        elif args[:3] == ('git', 'update-ref', 'refs/remotes/origin/main'): self.upstream_head = args[-2]
        elif args[:2] == ('docker', 'build'): code = int(self.fail_build)
        elif args[:3] == ('docker', 'image', 'inspect'):
            stdout = json.dumps([{'Config': {'Labels': {'org.opencontainers.image.revision': B}}}])
        elif args[:2] == ('docker', 'compose') and 'up' in args and kw['env']['RUNTIME_IMAGE'].endswith(B):
            code = int(self.fail_health)
        return SimpleNamespace(returncode=code, stdout=stdout, stderr='simulated failure' if code else '')


class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.state = self.root / '.state'; self.state.mkdir()
        (self.root / '.env').write_text('RUNTIME_IMAGE=old-image\nKEEP_THIS=value\n')
        self.active = {'revision': A, 'image': 'old-image', 'bundle': {'profile': 'daytime-swift'}}
        atomic_json(self.state / 'active.json', self.active)
        self.h = Harness(self.root); self.u = Updater(self.root, execute=self.h)
        def export(revision):
            source = self.state / revision; source.mkdir(exist_ok=True)
            return source
        self.u.export = export
        def candidate(image, action, **kwargs):
            self.h.calls.append(('candidate', image, action))
            if action == 'deploy':
                atomic_json(self.state / 'previous.json', self.active)
                atomic_json(self.state / 'active.json', {**self.active, 'revision': B, 'image': image})
            if action == 'rollback-release': atomic_json(self.state / 'active.json', self.active)
        self.u.candidate = candidate

    def test_rejects_dirty_branch_upstream_and_remote_before_fetch_or_docker(self):
        for attribute, value in [('dirty', ' M config/shared.json'), ('branch', ''), ('branch', 'feature'),
                ('upstream', 'elsewhere/main'), ('remote', 'https://example.com/other.git')]:
            with self.subTest(attribute=attribute, value=value):
                original = getattr(self.h, attribute); setattr(self.h, attribute, value); self.h.calls.clear()
                with self.assertRaises(RuntimeError): self.u.update()
                self.assertFalse(any(x[0] == 'docker' or x[:2] == ('git', 'fetch') for x in self.h.calls))
                setattr(self.h, attribute, original)

    def test_rewritten_history_fails_before_build(self):
        self.h.diverged = True
        with self.assertRaisesRegex(RuntimeError, 'merge-base'): self.u.update()
        self.assertFalse(any(x[0] == 'docker' for x in self.h.calls))
        self.assertEqual(self.h.upstream_head, A)

    def test_build_failure_leaves_deployed_runtime_and_source_unchanged(self):
        self.h.fail_build = True
        with self.assertRaisesRegex(RuntimeError, 'docker build'): self.u.update()
        self.assertFalse(any(x[0] == 'candidate' for x in self.h.calls))
        self.assertEqual(read(self.state / 'active.json'), self.active)
        self.assertEqual(self.h.head, A)

    def test_preparation_precedes_backend_apply_and_bounded_controller_replacement(self):
        self.u.update(); calls = self.h.calls
        build = next(i for i, x in enumerate(calls) if x[:2] == ('docker', 'build'))
        validate = next(i for i, x in enumerate(calls) if x[0] == 'candidate' and x[-1] == 'validate')
        deploy = next(i for i, x in enumerate(calls) if x[0] == 'candidate' and x[-1] == 'deploy')
        up = next(i for i, x in enumerate(calls) if x[:2] == ('docker', 'compose') and 'up' in x)
        merge = next(i for i, x in enumerate(calls) if x[:3] == ('git', 'merge', '--ff-only'))
        self.assertTrue(build < validate < deploy < up < merge)
        self.assertIn('--wait-timeout', calls[up]); self.assertIn('150', calls[up]); self.assertIn('--no-deps', calls[up])
        self.assertEqual(read(self.state / 'update-job.json')['phase'], 'succeeded')
        self.assertIn('KEEP_THIS=value', (self.root / '.env').read_text())
        self.assertFalse(any('down' in call or 'prune' in call or 'reset' in call for call in calls))

    def test_controller_health_failure_recovers_prior_release_without_rewinding_git(self):
        self.h.fail_health = True
        with self.assertRaisesRegex(RuntimeError, 'compose'): self.u.update()
        self.assertIn(('candidate', 'local/ai-runtime:git-' + B, 'rollback-release'), self.h.calls)
        self.assertEqual(read(self.state / 'active.json'), self.active)
        self.assertEqual(self.h.head, A)
        self.assertEqual(read(self.state / 'update-job.json')['phase'], 'failed')

    def test_no_new_revision_requires_a_healthy_runtime(self):
        self.h.remote_head = A; self.u.update()
        self.assertIn(('candidate', 'old-image', 'check'), self.h.calls)
        self.assertFalse(any(x[:2] == ('docker', 'build') for x in self.h.calls))

    def test_interrupted_controller_replacement_can_finish_without_backend_reapply(self):
        atomic_json(self.state / 'active.json', {**self.active, 'revision': B, 'image': 'new-image'})
        atomic_json(self.state / 'transaction.json', {'phase': 'succeeded'})
        atomic_json(self.state / 'update-job.json', {'phase': 'controller-replacement',
            'previous': A, 'target': B, 'image': 'new-image'})
        self.u.recover_update()
        self.assertEqual(self.h.head, B)
        self.assertFalse(any(x[0] == 'candidate' and x[-1] == 'deploy' for x in self.h.calls))
        self.assertEqual(read(self.state / 'update-job.json')['phase'], 'succeeded')
