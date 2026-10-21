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
        self.stale_old_health = False
        self.target_config = 'CONFIGURATIONS = ("daytime", "daytime-27b", "daytime-flash-f16")\n'

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
        elif args[:2] == ('git', 'show') and len(args) == 3 and args[2].endswith(':runtime/config.py'):
            stdout = self.target_config
        elif args[:3] == ('git', 'merge-base', '--is-ancestor'): code = int(self.diverged)
        elif args[:3] == ('git', 'merge', '--ff-only'): self.head = args[-1]
        elif args[:3] == ('git', 'update-ref', 'refs/remotes/origin/main'): self.upstream_head = args[-2]
        elif args[:2] == ('docker', 'build'): code = int(self.fail_build)
        elif args[:3] == ('docker', 'image', 'inspect'):
            stdout = json.dumps([{'Config': {'Labels': {'org.opencontainers.image.revision': B}}}])
        elif args[:2] == ('docker', 'compose') and 'up' in args and kw['env']['RUNTIME_IMAGE'].endswith(B):
            code = int(self.fail_health)
        elif args[:2] == ('docker', 'compose') and 'up' in args and self.stale_old_health:
            code = int('--force-recreate' not in args)
        return SimpleNamespace(returncode=code, stdout=stdout, stderr='simulated failure' if code else '')


class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.state = self.root / '.state'; self.state.mkdir()
        (self.root / '.env').write_text('RUNTIME_IMAGE=old-image\nKEEP_THIS=value\n')
        self.active = {'revision': A, 'image': 'old-image', 'bundle': {'profile': 'daytime'}}
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
                ('upstream', 'elsewhere/main'), ('remote', 'https://example.com/other.git'),
                ('remote', 'https://github.com/another-owner/ai-runtime.git'),
                ('remote', 'https://github.com/astigmatism/ai-runtime-other.git')]:
            with self.subTest(attribute=attribute, value=value):
                original = getattr(self.h, attribute); setattr(self.h, attribute, value); self.h.calls.clear()
                with self.assertRaises(RuntimeError): self.u.update()
                self.assertFalse(any(x[0] == 'docker' or x[:2] == ('git', 'fetch') for x in self.h.calls))
                setattr(self.h, attribute, original)

    def test_current_and_legacy_checkout_origins_fetch_the_canonical_repository(self):
        for slug in ('ai-runtime', 'local-ai-runtime'):
            for remote in (f'https://github.com/astigmatism/{slug}.git',
                    f'https://github.com/astigmatism/{slug}',
                    f'git@github.com:astigmatism/{slug}.git'):
                with self.subTest(remote=remote):
                    self.h.remote = remote
                    self.h.calls.clear()
                    before = self.u.source_preflight()
                    self.assertEqual(self.u.fetch(before), B)
                    self.assertIn(('git', 'fetch', '--no-tags',
                        'https://github.com/astigmatism/ai-runtime.git', 'refs/heads/main'), self.h.calls)

    def test_legacy_checkout_update_preserves_the_existing_controller_project(self):
        self.h.remote = 'https://github.com/astigmatism/local-ai-runtime.git'
        self.u.update()
        replacements = [call for call in self.h.calls if call[:2] == ('docker', 'compose') and 'up' in call]
        self.assertEqual(len(replacements), 1)
        self.assertEqual(replacements[0][replacements[0].index('-p') + 1], 'local-ai-runtime')
        self.assertEqual(replacements[0][-1], 'controller')
        self.assertEqual(self.h.head, B)

    def test_rewritten_history_fails_before_build(self):
        self.h.diverged = True
        with self.assertRaisesRegex(RuntimeError, 'merge-base'): self.u.update()
        self.assertFalse(any(x[0] == 'docker' for x in self.h.calls))
        self.assertEqual(self.h.upstream_head, A)

    def test_retired_active_configuration_refuses_before_build_with_actionable_error(self):
        self.active = {'revision': A, 'image': 'old-image', 'bundle': {'profile': 'daytime-27b-ram64'}}
        atomic_json(self.state / 'active.json', self.active)
        self.h.target_config = 'CONFIGURATIONS = ("daytime", "daytime-27b", "daytime-flash-f16")\n'
        with self.assertRaises(RuntimeError) as raised:
            self.u.update()
        message = str(raised.exception)
        self.assertIn("Active configuration 'daytime-27b-ram64' is retired in target revision " + B, message)
        # The list is the actionable half: the operator reading this is still on the old revision,
        # where the target's names do not exist yet, so the refusal has to say what to switch to
        # before the retry rather than leave the retry to guess.
        self.assertIn('daytime, daytime-27b, daytime-flash-f16', message)
        self.assertIn('switch to a configuration selectable there', message)
        self.assertFalse(any(x[:2] == ('docker', 'build') for x in self.h.calls))
        self.assertFalse(any(x[0] == 'candidate' for x in self.h.calls))
        self.assertEqual(read(self.state / 'active.json'), self.active)
        self.assertEqual(self.h.head, A)

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
        self.h.fail_health = True; self.h.stale_old_health = True
        with self.assertRaisesRegex(RuntimeError, 'compose'): self.u.update()
        self.assertIn(('candidate', 'local/ai-runtime:git-' + B, 'rollback-release'), self.h.calls)
        self.assertEqual(read(self.state / 'active.json'), self.active)
        self.assertEqual(self.h.head, A)
        self.assertEqual(read(self.state / 'update-job.json')['phase'], 'failed')
        recovery = [x for x in self.h.calls if x[:2] == ('docker', 'compose') and '--force-recreate' in x]
        self.assertEqual(len(recovery), 1)
        self.assertEqual(recovery[0][-1], 'controller'); self.assertIn('--no-deps', recovery[0])
        self.assertIn(('candidate', 'old-image', 'check'), self.h.calls)

    def test_no_new_revision_requires_a_healthy_runtime(self):
        self.h.remote_head = A; self.u.update()
        self.assertIn(('candidate', 'old-image', 'check'), self.h.calls)
        self.assertFalse(any(x[:2] == ('docker', 'build') for x in self.h.calls))

    def test_interrupted_controller_replacement_can_finish_without_backend_reapply(self):
        self.h.stale_old_health = True
        atomic_json(self.state / 'active.json', {**self.active, 'revision': B, 'image': 'new-image'})
        atomic_json(self.state / 'transaction.json', {'phase': 'succeeded'})
        atomic_json(self.state / 'update-job.json', {'phase': 'controller-replacement',
            'previous': A, 'target': B, 'image': 'new-image'})
        self.u.recover_update()
        self.assertEqual(self.h.head, B)
        self.assertFalse(any(x[0] == 'candidate' and x[-1] == 'deploy' for x in self.h.calls))
        self.assertEqual(read(self.state / 'update-job.json')['phase'], 'succeeded')
        recovery = next(x for x in self.h.calls if x[:2] == ('docker', 'compose') and 'up' in x)
        self.assertIn('--force-recreate', recovery)
        self.assertEqual(recovery[-1], 'controller')

    def test_the_deployment_environment_reaches_a_one_off_action(self):
        # The credential a push to another machine's router needs is a secret, so it lives in the
        # deployment's env file and nowhere that is tracked. An action started by the updater — a
        # switch, a republish after a router release — therefore has to be started with that file, or
        # it can verify a catalog and then be unable to deliver it, which is a failure with nothing
        # wrong in it. A checkout that has no env file yet is still a checkout that can run.
        atomic_json(self.state / 'host.json', {'uid': 1000, 'gid': 1000, 'model_root': '/models'})
        real = Updater(self.root, execute=self.h)
        self.h.calls.clear()
        real.candidate('new-image', 'apply')
        run = [call for call in self.h.calls if call[:2] == ('docker', 'run')][0]
        self.assertEqual(run[run.index('--env-file') + 1], str(self.root / '.env'))
        # The settings the action is about still win over anything the file happens to carry.
        self.assertEqual(run[run.index('--env') + 1], 'RUNTIME_STATE_DIR=' + str(self.state))
        (self.root / '.env').unlink(); self.h.calls.clear()
        real.candidate('new-image', 'status')
        self.assertNotIn('--env-file', [call for call in self.h.calls if call[:2] == ('docker', 'run')][0])

    def test_the_router_directory_is_mounted_only_for_a_machine_that_writes_one(self):
        # The directory is the router on this machine. A machine that publishes to a router elsewhere
        # has no such path, and Docker refuses a container whose bind source does not exist — so the
        # mount travels with the setting that says this machine writes a file, and a machine that only
        # pushes starts with a container that asks for nothing it does not have.
        real = Updater(self.root, execute=self.h)
        atomic_json(self.state / 'host.json',
            {'uid': 1000, 'gid': 1000, 'model_root': '/models', 'router_target':
                {'url': 'http://192.168.1.23:11435', 'token_env': 'CATALOG_PUSH_TOKEN'}})
        self.h.calls.clear()
        real.candidate('new-image', 'apply')
        run = [call for call in self.h.calls if call[:2] == ('docker', 'run')][0]
        self.assertEqual([item for item in run if '/router' in str(item)], [])
        atomic_json(self.state / 'host.json', {'uid': 1000, 'gid': 1000, 'model_root': '/models',
            'router_state_dir': '/router'})
        self.h.calls.clear()
        real.candidate('new-image', 'apply')
        run = [call for call in self.h.calls if call[:2] == ('docker', 'run')][0]
        self.assertIn('type=bind,source=/router,target=/router', run)
