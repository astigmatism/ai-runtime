import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from models import CODING, EVERYDAY
from runtime.config import CONFIGURATIONS, engine_interface, read, render
from runtime.controller import Controller
from runtime.system import atomic_json
from tests.test_controller import FakeSystem

ROOT = Path(__file__).resolve().parents[1]
BASE = read(ROOT / 'tests/fixtures/legacy-fingerprints.json')['host']
VISION = 'GPU-465a9a1e-fcb9-27f2-8f6a-cca1c30de981'
ENGINES = read(ROOT / 'config/shared.json')['engines']


def argv_engine(profile):
    """Whether this configuration's backend is handed an argument vector, which is what the UUID
    notation below describes. An environment-interface backend is addressed by index instead, so the
    CUDA_VISIBLE_DEVICES this test compares is a UUID list only for the former -- and since 2026-10-20
    a configuration is not the same thing as an argument-vector one; its models are."""
    return engine_interface(ENGINES[read(ROOT / 'config/profiles' / (profile + '.json'))['engine']]) == 'argv'


def vision_host():
    host = copy.deepcopy(BASE)
    host.update(vision_gpu_id=VISION, vision_gpu_name='RTX 3080', vision_cuda_device=4,
        cuda_order={'daytime': host['gpu_ids']['daytime'][::-1], 'nighttime': host['gpu_ids']['nighttime'][:]})
    return host


class VisionTests(unittest.TestCase):
    def test_all_profiles_only_change_encoder_and_visibility(self):
        host = vision_host()
        gpu_night = render(ROOT / 'config', host, 'qwen27b-q8-pair')['compose']['services'][EVERYDAY]
        # Exclusive profiles do not share the encoder GPU; they are covered by the solo tests. Nighttime's MTP
        # draft also lives on the vision GPU, so on a CPU-vision host it runs without the draft.
        for profile in (p for p in CONFIGURATIONS if not read(ROOT / 'config/profiles' / (p + '.json')).get('exclusive')):
            cpu = render(ROOT / 'config', BASE, profile)
            gpu = render(ROOT / 'config', host, profile)
            # The everyday model is the same launch whichever configuration loads it, and that is
            # worth asserting across a configuration built on a different engine as well as the
            # two that share one.
            self.assertEqual(gpu['compose']['services'][EVERYDAY], gpu_night)
            if not argv_engine(profile):
                continue
            for model, cfg in gpu['compose']['services'].items():
                prior = cpu['compose']['services'][model]
                # Which GPU pair a launch is on is a fact of the cards it reserved, not of the id it is
                # filed under: several of this host's models load one after another on the same pair and
                # in the same container.
                ids = cfg['deploy']['resources']['reservations']['devices'][0]['device_ids']
                group = 'nighttime' if set(ids) & set(BASE['gpu_ids']['nighttime']) else 'daytime'
                self.assertEqual(cfg['environment']['CUDA_VISIBLE_DEVICES'], ','.join([*host['cuda_order'][group], VISION]))
                self.assertEqual(ids, [*BASE['gpu_ids'][group], VISION])
                restored = copy.deepcopy(cfg)
                restored.pop('environment')
                restored['deploy'] = prior['deploy']
                argv = restored['command']
                argv[argv.index('--mmproj-offload')] = '--no-mmproj-offload'
                argv[argv.index('--mmproj-device') + 1] = 'none'
                if model == EVERYDAY:
                    i = argv.index('--spec-type'); del argv[i:i + 14]  # seven draft flags and their values
                    restored['volumes'] = [m for m in restored['volumes'] if m['target'] != '/weights/mtp.gguf']
                self.assertEqual(restored, prior)
            for m in gpu['catalog']['models']:
                self.assertEqual(m['mmproj_offload'], 'gpu')
                self.assertEqual(m['vision_device'], 'CUDA2')
                self.assertEqual(m['vision_gpu_uuid'], VISION)
                self.assertEqual(len(m['text_gpu_uuids']), 2)
                self.assertNotIn(VISION, m['text_gpu_uuids'])
            self.assertEqual(cpu['artifacts'], gpu['artifacts'])  # Nighttime's MTP head is the file Daytime also uses

    def test_overlap_invalid_uuid_and_ambiguous_cuda_order_are_rejected(self):
        bad = []
        h = vision_host(); h['vision_gpu_id'] = BASE['gpu_ids']['daytime'][0]; bad.append(h)
        h = vision_host(); h['vision_gpu_id'] = ''; bad.append(h)
        h = vision_host(); h['cuda_order'].pop('nighttime'); bad.append(h)
        h = vision_host(); h['cuda_order']['daytime'] = h['gpu_ids']['nighttime']; bad.append(h)
        h = vision_host(); h['gpu_ids']['nighttime'][0] = h['gpu_ids']['daytime'][0]; h['cuda_order']['nighttime'] = h['gpu_ids']['nighttime'][:]; bad.append(h)
        for host in bad:
            with self.subTest(host=host), self.assertRaises(RuntimeError):render(ROOT / 'config', host, 'qwen27b-q8-pair')

    def test_text_draft_and_tensor_placement_cannot_use_vision_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
            p = config / 'profiles/qwen27b-q8-pair.json'; original = read(p)
            for key, value in [('--device', 'CUDA0,CUDA1,CUDA2'), ('--spec-draft-device', 'CUDA2'),
                    ('--tensor-split', '60,40,0'), ('--override-tensor', ['blk.*=CUDA2'])]:
                definition = copy.deepcopy(original); definition['arguments'][key] = value
                p.write_text(json.dumps(definition))
                with self.subTest(key=key), self.assertRaises(RuntimeError):render(config, vision_host(), 'qwen27b-q8-pair')

    def test_live_gpu_order_reservations_and_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp); host = vision_host(); atomic_json(state/'host.json', host)
            (state/'router-token').write_text('synthetic')
            bundle = render(ROOT/'config', host, 'qwen27b-q8-pair'); system = FakeSystem(bundle)
            atomic_json(state/'active.json', {'revision': 'a'*40, 'bundle': bundle})
            c = Controller(ROOT/'config', state, 'a'*40, system)
            self.assertTrue(c.status()['ready'])
            for row in c.status()['services']:
                self.assertEqual(row['vision_gpu_id'], VISION)
                self.assertEqual(len(row['gpu_ids']), 2)
            name = 'qwen38-daytime'; original = copy.deepcopy(system.containers[name])
            for mutation, reason in [
                (lambda ci: ci['Config'].update(Env=[]), 'CUDA device order'),
                (lambda ci: ci['Config'].update(Env=['CUDA_VISIBLE_DEVICES=2,0,1']), 'CUDA device order'),
                (lambda ci: ci['HostConfig']['DeviceRequests'][0].update(DeviceIDs=[VISION]), 'GPU assignment'),
                (lambda ci: ci['HostConfig']['DeviceRequests'][0].update(Count=-1), 'GPU reservation contract')]:
                system.containers[name] = copy.deepcopy(original); mutation(system.containers[name])
                self.assertIn(reason, c.observe(bundle)[CODING]['differences'])
                self.assertFalse(c.status()['ready'])
