"""One GPU's power knobs, read and written through the driver: the half that runs as root.

    python3 -m runtime.gpucard read  --uuid GPU-… [--expect-name 'NVIDIA GeForce RTX 3090']
    python3 -m runtime.gpucard apply --uuid GPU-… --power 330 --clock-max 1905 --offset 75

This module is the card agent. It is executed by `docker run --privileged --gpus all` of this
project's own image — the same trick `ops/scripts/gpu-undervolt --docker` uses, because the
controller container has the Docker socket and no GPUs, and Rosalina's `sudo` prompts — so it has
to keep three promises that the rest of the runtime does not have to keep:

* **stdlib and `ctypes` only.** No `runtime.config`, no `runtime.controller`, no environment: the
  container it runs in mounts nothing and has no state directory, and an import that reached for
  `RUNTIME_STATE_DIR` would turn a read into a traceback.
* **Machine-readable, exactly one JSON object on stdout, and never a traceback.** `runtime/power.py`
  on the other side of the pipe is the thing that decides what a failure means; this file only
  reports what the driver did.
* **Reading needs no privileges and writing reports whether it worked.** Every knob is read back
  after it is written, because on this driver family a write can be accepted and ignored, and an
  undervolt that silently did not happen is worse than one that refused.

What the three knobs are, and what can and cannot be verified about them:

    --power      nvidia-smi --power-limit           bounds average draw and transients; readable
    --clock-max  nvidia-smi --lock-gpu-clocks      keeps the card out of the expensive top bins;
                                                 NOT readable: no query field reports the lock
    --offset     NVML nvmlDeviceSetGpcClkVfOffset   the actual undervolt — run clock N at the
                                                 voltage stock uses for N-offset. Readable.

There is no voltage readout on GeForce, so a clock cap is only ever *inferred* from the observed SM
clock, and this file says so in its own output rather than letting the page over-claim.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys

# Leave P-state transitions alone: a high clock floor is what makes an offsetting card crash at idle.
CLOCK_MIN = 210
# Driver granularity for the V/F offset. Values off the step are rounded, not refused: the driver
# would round them anyway, and rounding here is what makes the read-back comparable.
OFFSET_STEP = 15
QUERY = ('name,power.limit,power.default_limit,power.min_limit,power.max_limit,power.draw,'
    'power.draw.average,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,fan.speed,'
    'utilization.gpu,utilization.memory,pstate,memory.used,pci.bus_id,driver_version,'
    'clocks_event_reasons_counters.sw_power_cap')
# nvidia-smi answers these queries with the string '[N/A]' or 'Not Supported' where it has no value.
MISSING = ('', 'n/a', 'na', 'not supported', 'not available', '[n/a]')


def run(command):
    return subprocess.run(command, capture_output=True, text=True)


def number(value):
    """The query field as a float, or None where the driver has nothing to say."""
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in MISSING:
        return None
    try:
        return float(text)
    except ValueError:
        return None


class Nvml:
    """The two calls that are not in nvidia-smi: reading and setting the GPC V/F offset."""

    def __init__(self, uuid):
        self.lib = ctypes.CDLL('libnvidia-ml.so.1')
        if self.lib.nvmlInit_v2() != 0:
            raise RuntimeError('nvmlInit_v2 failed')
        handle = ctypes.c_void_p()
        result = self.lib.nvmlDeviceGetHandleByUUID(uuid.encode(), ctypes.byref(handle))
        if result != 0:
            raise RuntimeError('nvmlDeviceGetHandleByUUID rc=%d' % result)
        self.handle = handle

    def offset(self):
        value = ctypes.c_int(-999)
        result = self.lib.nvmlDeviceGetGpcClkVfOffset(self.handle, ctypes.byref(value))
        return (value.value if result == 0 else None), result

    def set_offset(self, mhz):
        return self.lib.nvmlDeviceSetGpcClkVfOffset(self.handle, ctypes.c_int(int(mhz)))


def query(uuid):
    result = run(['nvidia-smi', '-i', uuid, '--query-gpu=' + QUERY,
        '--format=csv,noheader,nounits'])
    body = result.stdout.strip()
    if result.returncode or ',' not in body:
        raise RuntimeError('nvidia-smi query failed for %s: %s' % (uuid,
            (result.stdout + result.stderr).strip()[:200]))
    keys = [key.strip() for key in QUERY.split(',')]
    # One value per field, and never split again on a comma a field itself contains.
    values = [value.strip() for value in body.split(',')[:len(keys)]]
    return dict(zip(keys, values))


def card_name(uuid):
    return run(['nvidia-smi', '-i', uuid, '--query-gpu=name', '--format=csv,noheader']).stdout.strip()


def read_card(uuid):
    """Everything the driver will report about this card right now, plus the offset from NVML."""
    fields = query(uuid)
    capped = number(fields.get('clocks_event_reasons_counters.sw_power_cap'))
    offset, offset_error = None, None
    try:
        offset, result = Nvml(uuid).offset()
        if result != 0 or offset is None:
            offset_error = 'nvml rc=%s' % result
    except Exception as error:  # noqa: BLE001 - a missing libnvidia-ml is a reading, not a crash
        offset_error = str(error).splitlines()[0][:120]
    return {
        'uuid': uuid, 'name': fields.get('name') or None,
        'driver': fields.get('driver_version'), 'pci': fields.get('pci.bus_id'),
        'power_limit': number(fields.get('power.limit')),
        'power_default': number(fields.get('power.default_limit')),
        'power_min': number(fields.get('power.min_limit')),
        'power_max': number(fields.get('power.max_limit')),
        'draw': number(fields.get('power.draw')), 'draw_average': number(fields.get('power.draw.average')),
        'clock_sm': number(fields.get('clocks.sm')), 'clock_max_sm': number(fields.get('clocks.max.sm')),
        'clock_mem': number(fields.get('clocks.mem')), 'pstate': fields.get('pstate'),
        'temperature': number(fields.get('temperature.gpu')), 'fan': number(fields.get('fan.speed')),
        'utilization': number(fields.get('utilization.gpu')),
        'memory_used': number(fields.get('memory.used')), 'memory_utilization': number(
            fields.get('utilization.memory')),
        # The counter is in microseconds; hours is the unit a human compares against a reboot.
        'power_capped_hours': None if capped is None else round(capped / 3.6e9, 2),
        'offset': offset, 'offset_error': offset_error,
    }


def state(read):
    """Verdict words about a read, from the two knobs that can actually be verified.

    `stock` is a claim with two parts, and the script this file replaces was explicit about it: a
    card is stock only when the power limit is at the driver's default *and* the offset is zero.
    Anything else is `modified`, and `unknown` means the offset could not be read at all — which is
    never the same as zero.
    """
    if read.get('offset') is None:
        return 'unknown'
    limit, default = read.get('power_limit'), read.get('power_default')
    at_default = limit is not None and default is not None and limit >= default
    if read['offset'] == 0 and at_default:
        return 'stock'
    return 'modified'


def apply_card(uuid, wants, retries=2, expect_name=None):
    """Write the three knobs, then read them back, retrying the whole set as a unit.

    Returns the commands that ran, the read-back, and a per-knob verdict. A failed write is data:
    the power limit and the offset can disagree, and a card sitting between two rungs is exactly the
    thing an operator needs to see, so nothing here collapses to a single ok/fail before the read.
    """
    if expect_name is not None:
        name = card_name(uuid)
        if name != expect_name:
            raise RuntimeError('%s reports %r, expected %r: a card was moved or re-enumerated'
                % (uuid, name, expect_name))
    before = read_card(uuid)
    low = before.get('power_min') or 100
    high = before.get('power_max') or 450
    power = int(round(float(wants['power'])))
    if not low <= power <= high:
        raise RuntimeError('power limit %d W outside this card\'s %.0f-%.0f W' % (power, low, high))
    offset = int(round(int(wants['offset']) / OFFSET_STEP)) * OFFSET_STEP
    clock_max = wants.get('clock_max')
    clock_min = int(wants.get('clock_min') or CLOCK_MIN)
    commands = []
    if clock_max is None:
        commands.append(['nvidia-smi', '-i', uuid, '-rgc'])
    else:
        commands.append(['nvidia-smi', '-i', uuid, '-lgc', '%d,%d' % (clock_min, int(clock_max))])
    commands.append(['nvidia-smi', '-i', uuid, '-pl', '%d' % power])
    errors, attempts = [], 0
    for attempt in range(retries + 1):
        attempts = attempt + 1
        errors = []
        for command in commands:
            result = run(command)
            if result.returncode:
                errors.append('%s: %s' % (' '.join(command[1:]),
                    (result.stdout + result.stderr).strip()[:160]))
        try:
            if Nvml(uuid).set_offset(offset) != 0:
                errors.append('nvml offset %+d MHz refused' % offset)
        except Exception as error:  # noqa: BLE001
            errors.append('nvml offset: %s' % str(error).splitlines()[0][:120])
        after = read_card(uuid)
        if after.get('power_limit') is not None and abs(after['power_limit'] - power) < 1 \
                and after.get('offset') == offset and not errors:
            break
    after = read_card(uuid)
    # Both verifiable knobs, or nothing: `ok` is the read-back verdict, not whether the calls ran.
    verified = bool(errors) is False and after.get('offset') == offset \
        and after.get('power_limit') is not None and abs(after['power_limit'] - power) < 1
    return {
        'before': before, 'after': after, 'attempts': attempts,
        'asked': {'power': power, 'clock_max': clock_max, 'offset': offset,
            'clock_min': None if clock_max is None else clock_min},
        'commands': [' '.join(command[1:]) for command in commands],
        'readback': {
            'power': 'ok' if after.get('power_limit') is not None
                and abs(after['power_limit'] - power) < 1 else 'fail',
            'offset': 'ok' if after.get('offset') == offset else 'fail',
            # Unverifiable is not a failure and not a pass, and saying 'ok' here would be the lie.
            'clock_max': 'unverifiable',
        },
        'errors': errors,
        'ok': verified,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(prog='runtime.gpucard', description=__doc__.splitlines()[0])
    parser.add_argument('verb', choices=('read', 'apply'))
    # One UUID for a write, and as many as you like for a read: the runtime reads every card this
    # machine declares in a single container start rather than paying for one per card.
    parser.add_argument('--uuid', action='append', required=True)
    parser.add_argument('--expect-name', action='append', default=None)
    parser.add_argument('--power', type=float, default=None)
    parser.add_argument('--clock-max', type=int, default=None)
    parser.add_argument('--clock-min', type=int, default=CLOCK_MIN)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--retries', type=int, default=2)
    arguments = parser.parse_args(argv)
    try:
        if arguments.verb == 'read':
            cards = {}
            for uuid in arguments.uuid:
                try:
                    cards[uuid] = read_card(uuid)
                except Exception as error:  # noqa: BLE001 - one unreadable card, not a failed pass
                    cards[uuid] = {'error': str(error)[:200]}
            return {'ok': True, 'verb': 'read', 'cards': cards}
        if arguments.power is None:
            raise RuntimeError('apply needs --power')
        if len(arguments.uuid) != 1:
            raise RuntimeError('apply takes exactly one --uuid')
        return {'ok': True, 'verb': 'apply',
            'apply': apply_card(arguments.uuid[0],
                {'power': arguments.power, 'clock_max': arguments.clock_max,
                    'clock_min': arguments.clock_min, 'offset': arguments.offset},
                retries=arguments.retries,
                # One name to check, and only when the caller knows what the card said last time:
                # this is what notices a card was physically moved and re-enumerated.
                expect_name=arguments.expect_name[0] if arguments.expect_name else None)}
    except Exception as error:  # noqa: BLE001 - the caller wants JSON, never a traceback
        return {'ok': False, 'verb': arguments.verb, 'uuid': ','.join(arguments.uuid),
            'error': str(error)}


if __name__ == '__main__':
    report = main()
    json.dump(report, sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write('\n')
    sys.exit(0 if report.get('ok') else 1)
