"""Serialized, recoverable reconciliation of the existing backend containers.

A configuration loads the models it names: one across the machine, or two side by side on the two
text GPU pairs. A model that owns every text GPU this host declares is loaded alone, and switching to
a configuration that names two recreates the pair. A switch runs the models the target names and stops
the ones it does not, whatever it was running before. On a host that declares one GPU group the
configurations that run alone are the only ones it has, and this is the same reconciliation over one
backend instead of two.
"""
import copy
from contextlib import ExitStack
import datetime
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.parse
import uuid

from . import provision as provision_tool
from .config import (ALIASES_FILE, ALIASES_VERSION, CONFIGURATIONS, PAIR_GROUPS,
    available_profiles, catalog_names, clean_alias, configurations_for, default_profile, digest,
    gpu_architectures, host_id, model_id, models_of, nameable_models, read, read_aliases, render,
    require, resolve_profile, router_target, service_engine)
from .power import Power
from .system import System, atomic_json, future, lock, now

# How long a router that was not listening has to stay not-listening before this machine offers it
# the catalog again. Five minutes is neither polite nor urgent: it is short enough that a router
# restarted during a coffee break finds its catalog on the next pass, and long enough that a router
# down for the afternoon is not asked five times a minute, every minute, for hours.
ANNOUNCE_RETRY_SECONDS = 300
DRAIN_HEAL_SECONDS = 60

class RouterRefused(RuntimeError):
    """A router answered and said no."""
    # Worth its own type because the two ways a router can fail a publication are opposite news: a
    # refusal is this machine's document being wrong, which no retry will fix, and a router that never
    # answered is a router that may be restarting, which the supervising loop is entitled to ride out.


def unreachable(error):
    """Whether an error means the router did not answer, as distinct from answering badly."""
    return isinstance(error, OSError) and not isinstance(error, urllib.error.HTTPError)


# A transaction that has not reached a terminal phase. While one exists the runtime holds its
# drain or its outcome is unknown, so nothing else may change backends.
PENDING_PHASES = ('prepared', 'draining', 'applying', 'verifying', 'restoring', 'needs-attention')
# How many consecutive quiet passes a drain must see before work is assumed finished.
DRAIN_SETTLE_ROUNDS = 3
# How often the switch re-observes a backend that is still reading its model. Docker's health
# state is the engine's own /health, which an engine like Strata answers from server start, so
# the waiting has to happen against the controller's own view, and this is the cadence of it.
READINESS_POLL_SECONDS = 15
# A failed restoration is retried on this timer until the service is back, and the wait stops
# growing at the last entry. The host keeps trying rather than giving up: an operator may be
# asleep, and deciding to stop trying is the same as nobody being there.
RESTORATION_BACKOFF = (30, 60, 120, 300, 600, 900)
# How many steps of a restoration pass are kept in its record, so a long retry history cannot
# outgrow the state file. The failure that started it is kept separately in `reason`.
RESTORATION_PHASE_LOG = 50


def services_of(bundle):
    return bundle['compose']['services'] if bundle else {}


class Controller:
    def __init__(self, config_dir=None, state_dir=None, revision=None, system=None):
        self.config_dir = Path(config_dir or os.environ.get('RUNTIME_CONFIG_DIR', '/app/config'))
        self.state = Path(state_dir or os.environ['RUNTIME_STATE_DIR'])
        self.host = read(self.state / 'host.json')
        self.system = system or System()
        self.revision = revision or Path('/app/REVISION').read_text().strip()
        self.image = os.environ.get('RUNTIME_IMAGE', 'local/ai-runtime:git-' + self.revision)
        # Where this machine sends its catalog. A runtime that starts and publishes nowhere is the
        # failure the router's contract names explicitly, so it is refused here, at the one place
        # every path constructs the controller through, rather than at the first switch — which is
        # the difference between a host that will not start and a host that drains a router, changes
        # a model, and tells nobody.
        self.target = router_target(self.host)
        require(self.target or self.host.get('router_state_dir'),
            'host.json states nowhere to publish: router_target (a router on another machine, with '
            'this machine\'s host_id) or router_state_dir (the directory this machine\'s own router '
            'reads). A runtime that starts publishing nowhere serves models nobody is told about.')

        # The cards' power state, which this process can read and write but has no GPU of its own to
        # touch: every read and write goes through a privileged one-shot container. Lazy about it --
        # constructing it opens nothing -- so a controller that only renders and switches pays nothing.
        self.power = Power(self)

    def load(self, filename, default=None):
        path = self.state / filename
        return read(path) if path.exists() else default

    def save(self, filename, value):
        require(not self.system.readonly, 'State mutation is forbidden in inspection mode')
        atomic_json(self.state / filename, value)

    def sinks(self):
        """Every router this machine publishes to, in the order it publishes to them.

        There used to be one answer to this question — the router on this machine, reached through a
        directory — and the whole of publication, drain and verification was written against it. A
        machine can now feed a router that lives elsewhere, and during the migration between those two
        states it feeds both: the same document, two sinks, and every obligation the router contract
        states has to be honoured against each of them, because a switch that drains one router while
        the other is still admitting work is a switch that interrupts someone.

        A local sink is a directory plus a reload call; a remote sink is a URL, a host id and a
        bearer token. Both are drained, both are published to, both are read back. The order is local
        first: it is the sink this runtime can put back exactly, and the one whose router refuses a
        catalog can be restored while the remote router has not yet been touched.
        """
        sinks = []
        if self.host.get('router_state_dir'):
            sinks.append({'kind': 'file', 'name': self.host.get('router_state_dir')})
        if self.target:
            sinks.append({'kind': 'http', 'name': self.target['url'], 'host': self.target['host']})
        return sinks

    def router_call(self, sink, path, body=None, method=None):
        """One admin call against one sink, authenticated the way that sink wants it."""
        if sink['kind'] == 'http':
            # Two credentials, because the router trusts them for different things: the publisher token
            # writes this machine's catalog and nothing else, and drain and state are the control plane,
            # which a write credential is not allowed to touch. A machine that switched behind a remote
            # router with only the first would fail at the moment it had already drained the local one.
            control = not path.startswith('hosts/')
            env = self.target['admin_token_env'] if control else self.target['token_env']
            token = self.target['admin_token'] if control else self.target['token']
            require(bool(token),
                'The router at ' + self.target['url'] + ' is called with the token in ' + env +
                ', which is not in this container environment. ' + (
                    'Drain and state are the router\'s control plane, and its contract keeps the '
                    'publisher credential out of them on purpose, so the control token has to be '
                    'stated: router_target.admin_token_env names it, and the value goes in the '
                    'deployment env file.' if control else
                    'The publisher credential is named in host.json and delivered in the environment; '
                    'it never belongs in the repository.'))
            headers = {'Authorization': 'Bearer ' + token}
            url = self.target['url'] + '/admin/api/' + path
        else:
            token = (self.state / 'router-token').read_text().strip()
            require(bool(token), 'Router credential is missing')
            headers = {self.host['router_admin_header']: token}
            url = self.host['router_admin_url'].rstrip('/') + '/admin/api/' + path
        # Only pass the verb when there is one: a caller that says nothing means GET, and a
        # double written against the shape this code had for two years keeps working.
        return self.system.http(url, body, headers, **({} if method is None else {'method': method}))

    def admin(self, path, body=None):
        """The legacy sink's admin call: this machine's own router, on this machine's admin port."""
        token = (self.state / 'router-token').read_text().strip()
        require(bool(token), 'Router credential is missing')
        return self.system.http(self.host['router_admin_url'].rstrip('/') + '/admin/api/' + path,
            body, {self.host['router_admin_header']: token})

    def remote_call(self, path, body=None, method=None):
        """The push sink's admin call, or None on a machine that has no push sink configured."""
        if not self.target:
            return None
        return self.router_call({'kind': 'http'}, path, body, method=method)


    def admin_healing_for(self, sink, path, body=None):
        """`admin_healing` against one named sink."""
        try:
            return self.router_call(sink, path, body)
        except urllib.error.HTTPError as error:
            if not self.heal_wedged_router(error, sink):
                raise
            return self.router_call(sink, path, body)

    def sink_drain(self, sink, healing=True):
        """One router's report about this machine: (draining, reason, work in flight).

        A host-scoped deployment reports drain per machine — `draining_hosts`, and a reason under
        `hosts[<id>]` — while `draining` stays the whole-router flag a router release of the router
        itself pulls. Reading the wrong one of those is the interesting failure: a machine that reads
        the global flag would refuse to switch while its neighbour was in a maintenance window, and a
        machine that reads the per-host flag against a router too old to have it would drain beside
        live traffic. So the presence of `draining_hosts` decides which is authoritative, and an older
        router keeps the meaning it always had.
        """
        call = self.admin_healing_for(sink, 'runtime-state') if healing else \
            self.router_call(sink, 'runtime-state')
        state = call['runtime']
        mine = self.target['host'] if sink['kind'] == 'http' else None
        if sink['kind'] == 'http' and 'draining_hosts' in state:
            report = (state.get('hosts') or {}).get(mine) or {}
            return {'draining': mine in (state.get('draining_hosts') or []),
                'reason': report.get('drain_reason'), 'scoped': True,
                'active': int((state.get('active_by_host') or {}).get(mine, 0) or 0),
                'queued': int((state.get('queued_by_host') or {}).get(mine, 0) or 0)}
        return {'draining': bool(state['draining']),
            'reason': state.get('drain_reason', state.get('reason')), 'scoped': False,
            'active': int(state.get('active_count', 0) or 0),
            'queued': int(state.get('queued_count', 0) or 0)}

    def mirror(self, sink):
        """Whether a sink is a copy being proven rather than the router the house is standing on.

        A machine feeding its own router and a new one over the road is in the migration, and the two
        are not equals: the directory is what clients are being served from and what a switch has to
        satisfy, while the push is a copy whose reachability is still being established. So a router
        over there that is unreachable is recorded as a sink that has not caught up — the supervising
        loop keeps offering it the catalog until it accepts — instead of stopping a switch that has
        nothing to do with it. A refusal is not this: an refused catalog is this machine's document
        being wrong, and it fails the switch on a mirror exactly as it would on a primary.

        The distinction is a property of the migration and not of the code: when `router_state_dir`
        leaves the host settings the push stops being a mirror, and every call below that is now
        tolerant becomes fatal without anything else having to remember to change.
        """
        return sink['kind'] == 'http' and bool(self.host.get('router_state_dir'))

    def drain_reports(self, healing=True):
        """Every sink's drain report, as (sink, report) pairs.

        A router that will not answer raises, and the caller's retry schedule is what that error is
        for. "Unknown" is not an answer here: a pass that treated an unreachable router as "not
        drained" would walk into a backend that still has somebody generating in it, and a recovery
        that treated it as "nothing to release" would leave traffic parked behind a drain nobody
        owns. Which is also why the notes in `drain_pass` are about backends and never about routers
        — an unreachable *backend* is serving nothing and must not hold a drain open forever, while
        an unreachable *router* is a question that has not been answered.
        """
        out = []
        for sink in self.sinks():
            try:
                out.append((sink, self.sink_drain(sink, healing)))
            except Exception as error:
                if not self.mirror(sink) or not unreachable(error):
                    raise
                out.append((sink, {'draining': False, 'reason': None, 'scoped': False, 'active': 0,
                    'queued': 0, 'unreachable': str(error)[:200]}))
        return out

    def draining_for_me(self, healing=True):
        """Whether a router this machine publishes to is holding traffic for this machine.

        Not the global flag: that one belongs to the router as a whole, and on a shared router it says
        nothing about whether *this* machine may replace a backend. It is a name collision waiting to
        be removed from the code, which is why nothing reads it any more.
        """
        return any(report['draining'] for _, report in self.drain_reports(healing))

    def release_orphan_drain(self):
        if self.system.readonly:
            return False
        try:
            with lock(self.state / 'runtime.lock'):
                return self._release_orphan_drain_locked()
        except Exception:
            # A busy runtime is left alone. Restoration records its own failed attempts
            # and schedules the next one before propagating an error here.
            return False

    def _release_orphan_drain_locked(self):
        """Reopen a router that is holding traffic for a machine that is not switching.

        A drain belongs to a transaction, and it is released when the transaction ends. That pairing
        breaks in one ordinary way: the pass that releases it finds the router unreachable — a router
        redeployed underneath, a controller restarted mid-switch with the record already resolved — and
        the drain is left on the router with a reason naming a run that finished an hour ago. The
        machine is then serving a model its own router refuses to hand out, and this runtime says so
        about itself (`serving()` is false while a drain is held), which means a page that looks broken
        is describing something real that nobody intended.

        Only a drain whose reason is one of this runtime's own transaction ids is released. An operator
        holding traffic to look at a card is not an orphan, and a machine that reopened the door out
        from under a human would be worse than the stall it fixed.
        """
        if self.load('transaction.json', {}).get('phase') in PENDING_PHASES:
            return False  # Something of ours is in flight, and it owns its drain.
        if self.load('router-maintenance.json', {}).get('reserved'):
            return False
        # The announcement pass runs whenever a page is looked at, and a machine with nothing to say
        # should still be quiet: this asks at most once a minute, which is far quicker than any person
        # notices a stall and far slower than any router minds being asked.
        last = self.load('drain.json', {})
        if last.get('checked_at', '') > future(-DRAIN_HEAL_SECONDS):
            return False
        try:
            reports = [report for _, report in self.drain_reports(healing=False)]
            held = [report for report in reports if report['draining']]
        except Exception:
            self.save('drain.json', {'checked_at': now(), 'error': 'the router would not answer'})
            return False  # A router that will not answer is a question, not an answer.
        self.save('drain.json', {'checked_at': now(), 'held': len(held),
            'reasons': sorted({str(r.get('reason')) for r in held})})
        if not held or any(report.get('unreachable') for report in reports) or any(
                not str(report.get('reason') or '').startswith('local-ai-runtime:') for report in held):
            return False
        # An abandoned drain can hide a stopped backend or a refused catalog. The normal
        # restoration verifies and republishes the recorded release before opening admission;
        # if that fails it leaves a durable, scheduled retry, never an open but empty router.
        self._restore_locked()
        self.save('drain.json', {'checked_at': now(), 'released_at': now(), 'held': len(held)})
        return True

    def router_view(self):
        """One router's runtime-state, for the read-only surfaces that have always shown one.

        The page shows one number for requests in flight and one drain flag, and a machine feeding two
        routers has two of each. Publication and the drain never read through here — they go to every
        sink — but a status page that printed two GPU rails and a table of routers would be answering
        a question nobody is asking yet. The local router is the one clients are still on during the
        migration, so it is the one described; a machine with only a push sink describes that one.
        """
        sinks = self.sinks()
        order = sorted(sinks, key=lambda s: 0 if s['kind'] == 'file' else 1)
        last = None
        for sink in order:
            try:
                return self.router_call(sink, 'runtime-state')
            except Exception as error:
                last = error
        raise last

    def admin_healing(self, path, body=None):
        """The admin call a mutating operation makes, with the one repair it is allowed.

        A router whose active catalog is unusable (a backend kind it cannot handle) refuses every
        admin call, including the read a recovery needs, and the state is persisted: a restart
        comes back wedged exactly as it was. The runtime is the only writer of that catalog file,
        so it is the only component that can converge it. If the catalog on disk is not the
        last-known-good release, it is the catalog of a switch the router refused: the good one is
        written back and the router is reloaded, and the call is retried once. If the router was
        not wedged that way, the error is re-raised and the caller's retry schedule is untouched.
        """
        try:
            return self.admin(path, body)
        except urllib.error.HTTPError as error:
            if not self.heal_wedged_router(error):
                raise
            return self.admin(path, body)

    def heal_wedged_router(self, error, sink=None):
        """Write the last-known-good catalog back to the router and reload it.

        Returns True only when something was actually changed. If the router was not wedged on a
        catalog this runtime refused - the error says otherwise, no last-known-good release is
        recorded, or the file on disk holds a catalog of the same backend kind as the release it
        is supposed to serve - it is left alone so the caller's retry surfaces the failure
        again. Only a foreign kind in that file is a wedge this runtime owns, because only this
        runtime writes that file, and it never published a catalog of a kind the router refused.
        The decorated fields a healthy publish adds to the file are deliberately ignored: a
        transient 503 against an otherwise-healthy file must not rewrite it. The router re-reads
        its catalog file on every call, so the reload is enough and no restart is involved.
        """
        # A push cannot wedge a router, so there is nothing here to heal for one: the document is
        # read by the same reader that would serve it before it is stored, and a refusal leaves the
        # accepted catalog in force. The wedge this repairs is specific to a directory — this runtime
        # wrote a file, a router then failed to read it, and the sink has no door to refuse at.
        if sink is not None and sink['kind'] == 'http':
            return False
        if not self.host.get('router_state_dir'):
            return False
        body = {}
        try:
            body = json.loads(error.read() or b'{}')
        except Exception:
            pass
        code = (body.get('error') or {}).get('code')
        if error.code != 503 or code not in (None, 'INVALID_BACKEND_KIND'):
            return False
        good = self.last_known_good()
        if good is None:
            return False
        catalog = copy.deepcopy(good['bundle']['catalog'])
        path = self.local_catalog_path()
        try:
            on_disk = json.loads(path.read_text())
        except Exception:
            return False
        if on_disk.get('backend_kind') == catalog.get('backend_kind'):
            return False
        atomic_json(path, catalog)
        self.admin('reload-config', {})
        return True

    def desired(self, profile=None):
        current = self.load('active.json', {})
        initial = resolve_profile(self.host.get('initial_profile'))
        selectable = configurations_for(self.config_dir, self.host)
        # A host's initial_profile may name a configuration later retired, or one that belongs to
        # another machine; start this host's own default instead. A renamed one still resolves, so a
        # rename never strands a host on its recorded release. The default is host-scoped for the
        # same reason the registry is: a host is never started on a configuration that would refuse
        # its own selection.
        selected = profile or current.get('bundle', {}).get('profile') or (
            initial if initial in selectable else default_profile(self.config_dir, self.host))
        return render(self.config_dir, self.host, selected, self.aliases()[0])

    def nameable_models(self):
        """{model id: configurations of this machine that load it} -- what may be given a name.

        Names attach to models and not to configurations, because a configuration is where the models
        go and a model is what a client, a router, and a person mean by "that thing". A model shared
        by two configurations is named once and said that way by both.
        """
        return nameable_models(self.config_dir, self.host)

    def aliases(self):
        """What this machine's models are called, and any reason they are not being used.

        Read on every use rather than once at startup: the page is where a name is edited, and a
        controller that renamed a model at boot would show one name and publish another for the
        length of a release.
        """
        return read_aliases(self.state / ALIASES_FILE, self.nameable_models())

    def set_alias(self, model, alias):
        """Name one model, or clear the name and take the default back.

        A name is not a release: nothing here drains, restarts or re-renders a launch, and the model
        id and the service IDs a client sends are untouched. The one obligation it inherits is the
        loud one -- a machine that changes what it is called has to say so, so a saved alias is
        published unless something is already mid-flight and will publish it itself.
        """
        require(not self.system.readonly, 'State mutation is forbidden in inspection mode')
        value = clean_alias(model, alias, self.nameable_models())
        # The file is written first and without the runtime lock, because a name is an operator
        # preference kept in one atomic file, not a runtime transaction: it must survive and be
        # readable even on a machine that is draining, loading, or refusing to start.
        stored = self.load(ALIASES_FILE, {})
        saved = dict(stored.get('aliases') or {}) if isinstance(stored, dict) else {}
        if value is None:
            saved.pop(model, None)
        else:
            saved[model] = value
        self.save(ALIASES_FILE, {'version': ALIASES_VERSION, 'aliases': saved})
        reply = {'saved': True, 'published': False}
        if self.load('transaction.json', {}).get('phase') in PENDING_PHASES:
            return {**reply, 'note': 'A switch or restoration is in flight; it publishes this name with its own catalog.'}
        active = self.load('active.json')
        if not active or active['revision'] != self.revision:
            return {**reply, 'note': 'Nothing of this revision is serving, so there is no catalog to rename.'}
        # The lock is taken for the publication and never waited for. What it protects is writing a
        # catalog, and a rename that arrives while a switch holds it has nowhere to publish in front
        # of that switch: the switch reads the aliases when it publishes, so it says the new name
        # itself. Waiting would make a rename cost a drain, which is the thing to refuse here.
        held = ExitStack()
        try:
            held.enter_context(lock(self.state / 'runtime.lock'))
        except RuntimeError:
            return {**reply, 'note': 'Another runtime operation holds the runtime lock; this name is published '
                'with the next event, by whoever holds it.'}
        with held:
            try:
                self.publish(active['bundle'])
            except Exception:
                # Saved either way: the publication record says what the router said, and the loop that
                # reports health keeps offering the catalog until it lands.
                return reply
        return {**reply, 'published': True}

    def gpu_inventory(self):
        """This machine's cards as cards, in the order a configuration reserves them.

        The map draws one tile per physical GPU, which needs the cards rather than a count: two cards
        of the same model are two tiles. What a configuration reserves is a whole GPU group, so a
        tile's owner is read off the group it belongs to and never off its name, which is decorative
        and collides — one host's list carries `RTX 4080` and `RTX 4080 SUPER`.
        """
        declared = self.host.get('gpu_ids') or {}
        named = self.host.get('gpu_names') or {}
        text, order = [], 0
        for group in [group for group in PAIR_GROUPS if group in declared]:
            listed = named.get(group) or []
            for position, gpu in enumerate(declared[group]):
                text.append({'uuid': gpu, 'group': group, 'order': order, 'position': position,
                    'name': listed[position] if position < len(listed)
                        else 'Text GPU ' + str(position + 1)})
                order += 1
        vision = self.host.get('vision_gpu_id')
        return {'text': text, 'vision': None if not vision else {'uuid': vision, 'order': order,
            'name': self.host.get('vision_gpu_name') or 'Vision GPU'}}

    def backend_url(self, cfg):
        return 'http://' + cfg['container_name'] + ':8080'

    def host_models(self):
        """{model id: container name} for every model this machine's registry could start.

        A switch runs the models its configuration names and stops the ones it does not, so the
        runtime has to know which containers belong to this machine's models at all and not only to
        the configuration currently loaded: a model that another of this host's configurations loads
        is a container that can be found running with these cards on them, and that is a fault.
        """
        models = {}
        for name in configurations_for(self.config_dir, self.host):
            for model in models_of(self.config_dir, read(self.config_dir / 'profiles' / (name + '.json'))):
                models[model_id(model)] = model['container_name']
        return models

    def removed_backends(self, bundle, previous_bundle):
        """Existing containers that must not run beside `bundle`, so they release their GPUs first.

        Two kinds. The previous release's launches whose model the target does not name are released
        whatever their container is called: several of this host's models render into the same
        container name -- every daytime launch is `qwen38-daytime` -- and a container named that and
        belonging to a service the target no longer declares is not the container the target wants,
        it is the one standing in its way. Compose tracks it as an orphan of a service that left the
        file, and a launch asked to publish a fixed container name onto cards a live container is
        still holding is a name conflict, not a reload.

        The second kind is a leftover: a container of a model of this host the target does not name.
        It counts only where nothing the target names starts a container of that name, because the
        name is the identity Docker gave the running container -- when the target does start one under
        that name, what is running under it is the launch this switch replaces, and Compose handles it
        under the target's own service.

        A leftover is matched by container, because that is all Docker reports: several of this host's
        models render into one name, so a container this release did not start can hide behind a name
        the target also starts. What it cannot hide from is the drift comparison in the transition,
        which reads the running launch against the one the target renders for that name.
        """
        services = services_of(bundle)
        names = [cfg['container_name'] for service, cfg in services_of(previous_bundle).items()
            if service not in services]
        wanted = {cfg['container_name'] for cfg in services.values()}
        names += [name for model, name in self.host_models().items()
            if model not in services and name not in wanted]
        return [name for name in dict.fromkeys(names) if self.system.inspect(name)]

    def release(self, names):
        """Stop, then remove, so a restart policy can never bring a backend back onto reassigned GPUs."""
        for name in names:
            if self.system.inspect(name):
                self.system.docker('stop', '--time', '60', name, timeout=180)
                self.system.docker('rm', name)
            require(not self.system.inspect(name), 'Backend could not be removed: ' + name)

    def provisioning(self, bundle):
        """Return (service, engine, contract) for a service that builds its own state on first start.

        The switch now prepares such a workspace itself before it drains, so the readiness window
        only has to cover the load, and the load is waited on for the engine's actual answer.
        """
        for service, cfg in bundle['compose']['services'].items():
            engine = service_engine(bundle, service)
            if engine.get('provisioning'):
                return service, engine, engine['provisioning']
        return None, None, None

    def provisioned(self, bundle):
        """The measured receipt for this bundle's workspace, or None if it has never been measured.

        A receipt only speaks for the engine image and the workspace directory it was taken with, so
        both have to match before it counts as evidence about the thing about to be switched to.
        """
        service, engine, provision = self.provisioning(bundle)
        if engine is None:
            return None
        receipt = self.load('provisioning/' + bundle['profile'] + '.json')
        if not receipt:
            return None
        source = next((v['source'] for v in bundle['compose']['services'][service]['volumes']
                       if v['target'] == provision.get('workspace_target')), None)
        if (receipt.get('image_id') != engine['image_id'] or receipt.get('workspace') != source
                or receipt.get('engine_tag') != engine['tag']):
            return None
        return receipt

    def preparation_state(self, bundle):
        """(service, engine, contract, workspace, cfg) for the service that builds its own state,
        read without touching anything, or None when the bundle has no such service."""
        service, engine, contract = self.provisioning(bundle)
        if service is None:
            return None
        cfg = bundle['compose']['services'][service]
        return service, engine, contract, provision_tool.workspace_target(cfg, contract), cfg

    def preparation_needed(self, bundle):
        """Whether the switch must build this bundle's workspace before it takes the drain.

        A workspace that already holds everything the engine needs is skipped without ceremony:
        the common case after the first switch is a comparison, not a build.
        """
        found = self.preparation_state(bundle)
        if found is None or found[3] is None:
            return False
        service, engine, contract, workspace, cfg = found
        return not provision_tool.workspace_ready(provision_tool.ready_files(workspace, contract))

    def ensure_provisioned(self, bundle, log=lambda message: None):
        """The half of a first start that is safe beside a loaded model, done before the drain.

        The trial host measured the split: while 94 GB was verified and packed beside a serving
        model, its slots answered in 0.003 s against a 0.002 s baseline and it never left health;
        the load that follows the pack is the half that cannot run beside the model it replaces,
        and it waits for the drain. So a switch to a self-provisioning engine does the safe half
        here, in plain sight, before it has asked for anything. If the engine cannot build its
        workspace, the switch stops at the edge of the transaction: no drain, no touched backend,
        and the reason the engine gave is the reason the operator gets.
        """
        found = self.preparation_state(bundle)
        require(found is not None and found[3] is not None,
            'This bundle declares no provisionable workspace')
        service, engine, contract, workspace, cfg = found
        name = cfg['container_name'] + '-provision'
        port = provision_tool.published_port(cfg, provision_tool.PORT_OFFSET)
        files = provision_tool.ready_files(workspace, contract)
        # The daemon refuses a bind whose source is missing, so the directory the switch is about to
        # build has to exist before compose is asked to start anything.
        created = provision_tool.ensure_workspace(self, workspace)
        free_before = provision_tool.free_bytes(workspace)
        spec = self.state / 'provisioning' / (bundle['profile'] + '-compose.json')
        provision_tool.atomic_json(spec, provision_tool.throwaway(bundle, service, name, port))
        result = provision_tool.watch(self.system, spec, service, name, port, files,
            serving=provision_tool.serving_containers(self), goal='workspace',
            stop=provision_tool.stop_seconds(cfg), log=log)
        if result['reached'] != 'workspace':
            words = (result.get('container_log') or '').strip().splitlines()
            last_line = words[-1] if words else 'no output was captured from the container'
            raise RuntimeError(bundle['profile'] + ': the engine could not build its workspace '
                '(' + result['reached'] + ' after ' + str(result['seconds']) + 's): ' + last_line)
        self._record_provision_run(bundle, service, engine, contract, workspace, name, port, result,
            free_before, created)
        return result

    def _record_provision_run(self, bundle, service, engine, contract, workspace, name, port, result,
            free_before, created=False):
        """The build the switch just did, in the same receipt the read-only measurement writes, so
        one file is the whole story of the workspace on this machine."""
        receipt = self.load('provisioning/' + bundle['profile'] + '.json') or {
            'profile': bundle['profile'], 'engine_tag': engine['tag'], 'image_id': engine['image_id'],
            'workspace': str(workspace), 'ready_files': list(contract.get('ready_files', [])), 'runs': []}
        receipt['measured_at'] = now()
        run = {'phase': 'provision', 'reached': result['reached'], 'seconds': result['seconds'],
            'container': name, 'port': port, 'via': 'switch', 'workspace_created': created,
            'pack_bytes': provision_tool.footprint(workspace),
            'free_bytes_before': free_before, 'free_bytes_after': provision_tool.free_bytes(workspace),
            'ready_paths': result['ready_paths'], 'samples': result['samples']}
        if result.get('container_log') is not None:
            run['container_log'] = result['container_log']
        receipt['runs'].append(run)
        receipt['pack_bytes'] = run['pack_bytes']
        receipt['provision_seconds'] = result['seconds']
        receipt['workspace_ready'] = result['workspace_ready']
        receipt['disk_free_bytes'] = {'before': free_before, 'after': run['free_bytes_after']}
        nearby = [sample['neighbours'] for sample in result['samples'] if sample.get('neighbours')]
        receipt['neighbour_healthy'] = bool(nearby) and all(n['health'] == 'healthy'
            for sample in nearby for n in sample)
        answered = [n['slots_seconds'] for sample in nearby for n in sample
                    if n.get('slots_seconds') is not None]
        receipt['neighbour_slots_max_seconds'] = max(answered) if answered else None
        receipt['neighbour_samples'] = nearby[-8:]
        atomic_json(self.state / 'provisioning' / (bundle['profile'] + '.json'), receipt)

    def _record_warm_boot(self, bundle, services, seconds):
        """The boot the switch just measured, where the next refusal can read it as evidence.

        A successful switch is the warm boot: the engine loaded from its own workspace with no
        other model on the cards and answered the bounded generation, so the number the switch
        waited is the number the window has to cover.
        """
        found = self.preparation_state(bundle)
        if found is None or found[0] not in services:
            return
        service, engine, contract, workspace, cfg = found
        receipt = self.load('provisioning/' + bundle['profile'] + '.json') or {
            'profile': bundle['profile'], 'engine_tag': engine['tag'], 'image_id': engine['image_id'],
            'workspace': str(workspace), 'ready_files': list(contract.get('ready_files', [])), 'runs': []}
        receipt['measured_at'] = now()
        receipt['runs'].append({'phase': 'warm', 'reached': 'healthy', 'seconds': seconds,
            'via': 'switch', 'container': cfg['container_name']})
        receipt['warm_boot_seconds'] = seconds
        receipt['healthy_warm_boot'] = True
        receipt['workspace_ready'] = True
        atomic_json(self.state / 'provisioning' / (bundle['profile'] + '.json'), receipt)

    def verdict(self, bundle):
        """The hard reasons a switch to this bundle is refused, and nothing softer.

        A refusal is evidence the machine cannot do the job: a warm boot that was measured and
        failed, a container that stopped itself at startup, or a backend kind the deployed router
        cannot serve. Everything else — a workspace that still needs building, a window that is
        thin against a measurement — is a note the gate shows, because the switch now builds the
        workspace itself and waits for the engine's actual answer, and the recovery path stands
        behind a switch that turns out to be wrong.
        """
        reasons = []
        kind = bundle['catalog'].get('backend_kind')
        kinds = self.router_supported_kinds()
        if kind and kinds is not None and kind not in kinds:
            reasons.append('The deployed router does not support backend kind ' + kind +
                '; a switch to this profile is refused until the router is updated to a release that does')
        if self.preparation_state(bundle) is None:
            return reasons
        receipt = self.provisioned(bundle)
        if receipt is None:
            return reasons
        last = (receipt.get('runs') or [{}])[-1]
        phase = str(last.get('phase') or '')
        reached = str(last.get('reached') or '')
        if phase == 'warm' and reached != 'healthy':
            detail = str(receipt.get('error') or 'see the receipt')
            reasons.append('This engine has never completed a warm boot from its own workspace on this '
                'host (' + detail + '); the switch is refused until it does, or the profile is retired')
        if phase == 'provision' and reached.startswith('exited'):
            service, engine, contract, workspace, cfg = self.preparation_state(bundle)
            reasons.append('The ' + engine['tag'] + ' container stopped by itself at startup ' + reached +
                '; its output is in the receipt. A switch would refuse to build the workspace for '
                'this engine until that is fixed, and the profile is better retired if it cannot be')
        return reasons

    def router_supported_kinds(self):
        """The backend kinds the deployed router can serve, or None if it cannot be asked.

        Cached briefly because the status page asks about every selectable profile; the switch
        itself verifies fresh, so this is the display layer and a stale entry costs at most a
        retry after a router update.
        """
        # Wall-clock, not monotonic: the switch's drain accounting owns the monotonic clock,
        # and this cache must not steal a reading from it.
        cached = getattr(self, '_kinds_cache', None)
        if cached is not None and (
                datetime.datetime.now(datetime.timezone.utc) - cached[0]).total_seconds() < 30:
            return cached[1]
        kinds = None
        try:
            state = self.router_view()
            reported = (state.get('router') or {}).get('supported_backend_kinds')
            # A router old enough not to report its kinds supports exactly the two kinds it has
            # always served; a catalog of anything else is refused below, as it should be.
            kinds = list(reported) if reported else ['ollama', 'llama_cpp']
        except Exception:
            kinds = None
        self._kinds_cache = (datetime.datetime.now(datetime.timezone.utc), kinds)
        return kinds

    def require_router_supports(self, bundle):
        """Refuse, before the drain, a catalog the deployed router cannot serve.

        Publishing a kind the router does not implement wedges the router's own health and admin
        paths, and because the wedged catalog is a file the router reads back on boot, a restart
        cannot clear it: the host would come up serving nothing, and only a hand-written file
        would put it back. Refusing at the edge is the only way a switch cannot leave that state
        behind, so the check fails closed - a router that cannot be asked cannot be switched.
        """
        kind = bundle['catalog'].get('backend_kind')
        if not kind:
            return
        # Every router this machine feeds must be able to serve the kind, and a refusal is asked
        # the same way the drain asks it: through the sink, healing the file router it might itself
        # have wedged with an earlier refused catalog. A router over the road that is merely
        # unreachable is a mirror that has not caught up, not a router that refuses this catalog, so
        # it holds the switch no more than it holds the drain -- and once `router_state_dir` leaves
        # the host settings it stops being a mirror and its silence becomes fatal by itself.
        for sink in self.sinks():
            try:
                state = self.admin_healing_for(sink, 'runtime-state')
            except Exception as error:
                if self.mirror(sink) and unreachable(error):
                    continue
                require(False, 'The router ' + sink['name'] + ' cannot be reached, so this switch cannot '
                    'verify that its backend kind ' + kind + ' is supported: ' + str(error))
            reported = (state.get('router') or {}).get('supported_backend_kinds')
            supported = tuple(reported) if reported else ('ollama', 'llama_cpp')
            require(kind in supported,
                'The deployed router ' + sink['name'] + ' does not support backend kind ' + kind + ': '
                'publishing this catalog would '
                'leave the router unable to serve the configuration or to take administration, and a restart '
                'would not clear it. The switch is refused before the drain; update the router to a release '
                'that supports the kind, or the profile must name a supported kind')

    def verdict_notes(self, bundle):
        """What a switch to this bundle will do that is worth knowing before it is selected.

        These never refuse; they are the difference between selecting a profile and knowing what
        the selection costs.
        """
        found = self.preparation_state(bundle)
        if found is None:
            return []
        service, engine, contract, workspace, cfg = found
        notes = []
        receipt = self.provisioned(bundle)
        if workspace is None or not provision_tool.workspace_ready(
                provision_tool.ready_files(workspace, contract)):
            note = ('The ' + engine['tag'] + ' workspace has not been built on this host; the '
                    'switch will build it first, beside the model that is serving, before it '
                    'takes the drain')
            if receipt:
                built = [run for run in reversed(receipt.get('runs', []))
                         if run.get('phase') == 'provision' and run.get('reached') == 'workspace']
                if built:
                    note += ' (the last build here took ' + str(built[0]['seconds']) + 's)'
            notes.append(note)
        if receipt and receipt.get('healthy_warm_boot') is True:
            window = engine.get('readiness_wait_seconds') or 0
            measured = float(receipt.get('warm_boot_seconds') or 0)
            if measured and window < 3 * measured:
                notes.append('The readiness window for ' + engine['tag'] + ' is ' + str(window) +
                    's, under three times the measured warm boot of '
                    + str(receipt['warm_boot_seconds']) + 's; raise readiness_wait_seconds to at '
                    + 'least ' + str(int(3 * measured + 1)) + 's')
        return notes

    def switchable(self, bundle):
        """Refuse a switch the evidence says will not work; let the rest proceed and prepare.

        The refusal used to be paperwork — a window no one had measured. After the switch learned
        to build the workspace itself and wait for the engine's actual answer, the refusals are:
        a warm boot that was measured and failed, an engine that stops itself at startup, and a
        backend kind the deployed router cannot serve - the one refusal the evidence cannot catch
        afterwards, because publishing it wedges the router in a state a restart cannot clear.
        """
        reasons = self.verdict(bundle)
        if reasons:
            raise RuntimeError(reasons[0])
        # The verdict's kind check is the display layer and may be a second stale; the switch
        # itself verifies against the router fresh, and fails closed if it cannot.
        self.require_router_supports(bundle)

    def validate(self, bundle, full_hash=False):
        images = {}
        for service, cfg in bundle['compose']['services'].items():
            engine = service_engine(bundle, service)
            require(cfg['image'] == engine['tag'], service + ': inference engine tag differs from manifest')
            image = json.loads(self.system.docker('image', 'inspect', engine['tag']))[0]
            require(image['Id'] == engine['image_id'] and (image['Config'].get('Labels') or {}).get(
                'org.opencontainers.image.revision') == engine['revision'], service + ': pinned inference engine identity changed')
            images[service] = image['Id']
        network = bundle['compose']['networks']['router']['name']
        self.system.docker('network', 'inspect', network)
        cache = self.load('artifact-receipts.json', {})
        receipts = {}
        for artifact in bundle['artifacts']:
            path = Path(artifact['source'])
            require(path.is_file(), 'Required model artifact is missing: ' + path.name)
            require(path.resolve().is_relative_to(Path(self.host['model_root']).resolve()), 'Artifact symlink escapes model root')
            stat = path.stat()
            require(stat.st_size == artifact['bytes'], 'Model artifact size changed: ' + path.name)
            fingerprint = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                'ctime_ns': stat.st_ctime_ns, 'inode': stat.st_ino}
            prior = cache.get(str(path), {})
            verified = prior.get('verified') is True and prior.get('fingerprint') == fingerprint and prior.get('sha256') == artifact['sha256']
            if full_hash or (not self.system.readonly and not verified):
                sha = hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                        sha.update(chunk)
                require(sha.hexdigest() == artifact['sha256'], 'Model artifact checksum mismatch: ' + path.name)
                after = path.stat()
                require((stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino) ==
                    (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino), 'Artifact changed while hashing')
                verified = True
            receipts[str(path)] = {'fingerprint': fingerprint, 'sha256': artifact['sha256'], 'verified': verified}
        if not self.system.readonly:
            self.save('artifact-receipts.json', {**cache, **receipts})
        return {'image_ids': images, 'artifacts': len(receipts),
            'all_checksums_verified': all(x['verified'] for x in receipts.values()), 'receipts': receipts}

    def observe(self, bundle):
        observed = {}
        network = bundle['compose']['networks']['router']['name']
        for service, cfg in bundle['compose']['services'].items():
            engine = service_engine(bundle, service)
            ci = self.system.inspect(cfg['container_name'])
            reasons = []
            result = {'container_name': cfg['container_name'], 'healthy': False,
                'running': bool(ci and ci['State']['Running']), 'differences': reasons}
            if not ci:
                reasons.append('container missing')
                observed[service] = result
                continue
            result.update(id=ci['Id'], started_at=ci['State']['StartedAt'],
                restart_count=ci['RestartCount'], image_id=ci['Image'])
            if not result['running']:
                # A stopped container answers nothing, and every identity check above passes for it:
                # without this line the observer would report it unhealthy with no reason at all,
                # which is exactly the record the 18:47 trial left — an engine that died mid-load
                # reads as "nothing wrong", and the switch's window is never the answer to "why".
                reasons.append('container not running')
            # Every expectation is read with .get(): a service dictionary that lost a key is a
            # configuration fault, and the operator needs to see which setting drifted, not a crash in
            # the middle of a switch. A missing key never matches, so it falls through as a reason.
            for valid, reason in [
                (ci['Image'] == engine['image_id'], 'image'),
                ((ci['Config'].get('Cmd') or []) == (cfg.get('command') or []), 'arguments'),
                (ci['Config'].get('Entrypoint') == cfg.get('entrypoint'), 'entrypoint'),
                # A service that runs as the image's own user renders no user at all and the daemon
                # reports that as an empty string; comparing them literally would call such a
                # container drifted on every check, which would schedule pointless restarts.
                ((ci['Config'].get('User') or None) == cfg.get('user'), 'user'),
                (ci['HostConfig'].get('ReadonlyRootfs') == cfg.get('read_only'), 'read-only root'),
                (ci['HostConfig'].get('RestartPolicy', {}).get('Name') == cfg.get('restart'), 'restart policy'),
                (ci['HostConfig'].get('Init') == cfg.get('init'), 'init'),
                (ci['HostConfig'].get('Privileged', False) is False, 'privileged mode')]:
                if not valid:
                    reasons.append(reason)
            requests = ci['HostConfig'].get('DeviceRequests') or []
            reserved = cfg.get('deploy', {}).get('resources', {}).get('reservations', {})
            expected_ids = (reserved.get('devices') or [{}])[0].get('device_ids')
            if len(requests) != 1 or requests[0].get('DeviceIDs') != expected_ids:
                reasons.append('GPU assignment')
            expected_cuda = cfg.get('environment', {}).get('CUDA_VISIBLE_DEVICES')
            live_env = dict(item.split('=', 1) for item in ci['Config'].get('Env', []) if '=' in item)
            if live_env.get('CUDA_VISIBLE_DEVICES') != expected_cuda:
                reasons.append('CUDA device order')
            if expected_cuda and (len(requests) != 1 or requests[0].get('Driver') != 'nvidia'
                    or requests[0].get('Count') != 0 or requests[0].get('Capabilities') != [['gpu']]):
                reasons.append('GPU reservation contract')
            mounts = {(x.get('Source'), x.get('Destination'), x.get('RW'))
                for x in ci.get('Mounts') or [] if x.get('Type') == 'bind'}
            if mounts != {(x.get('source'), x.get('target'), not x.get('read_only', True))
                    for x in cfg.get('volumes') or []}:
                reasons.append('model mounts')
            ports = cfg.get('ports') or []
            # Compare to this release's binding, including during rollback to a release
            # that predates the current host's LAN address setting.
            binding = ports[0].rsplit(':', 2) if len(ports) == 1 else []
            if len(binding) != 3 or binding[2] != '8080' or ci['HostConfig'].get('PortBindings') != {
                    '8080/tcp': [{'HostIp': binding[0], 'HostPort': binding[1]}]}:
                reasons.append('port bindings')
            connection = (ci.get('NetworkSettings') or {}).get('Networks', {}).get(network, {})
            aliases = ((cfg.get('networks') or {}).get('router') or {}).get('aliases') or []
            if not aliases or not connection or not set(aliases).issubset(connection.get('Aliases') or []):
                reasons.append('router network')
            if result['running']:
                try:
                    health = self.system.http(self.backend_url(cfg) + '/health', timeout=3)
                    slots = self.system.http(self.backend_url(cfg) + '/slots', timeout=3)
                    entry = next(x for x in bundle['manifest']['services'] if x['service'] == service)
                    if engine.get('interface') == 'environment':
                        # Both /health and /slots answer in llama.cpp's shape, but there is no argument
                        # vector to read the window from: the published configuration is the reference.
                        expected_ctx, expected_slots = entry['context_tokens'], entry['parallel_slots']
                    else:
                        argv = cfg['command']
                        expected_ctx = int(argv[argv.index('--kv-unified-per-slot') + 1])
                        expected_slots = int(argv[argv.index('--parallel') + 1])
                    reported = [x.get('n_ctx') for x in slots if isinstance(x, dict)] if isinstance(slots, list) else None
                    if (not isinstance(slots, list) or len(slots) != expected_slots
                            or len(reported) != expected_slots
                            or any(x != expected_ctx for x in reported)):
                        # What the backend answered goes in the record: a window smaller than the one the
                        # profile contracted for is permanent and needs reading, while an empty slot list
                        # is a backend that is still loading. Both look the same as a waiting engine
                        # otherwise, and the switch would sit out the whole health timeout to find out.
                        result['slot_window'] = {'contracted': expected_ctx, 'slots': expected_slots,
                            'reported_context': reported, 'loaded': health.get('loaded')
                                if isinstance(health, dict) else None}
                        reasons.append('slot context')
                    result['healthy'] = isinstance(health, dict) and health.get('status') == 'ok' and not reasons
                    result['processing'] = isinstance(slots, list) and any(
                        x.get('is_processing', False) for x in slots if isinstance(x, dict))
                except Exception:
                    result['health_error'] = 'Backend readiness unavailable'
            observed[service] = result
        # A model this configuration does not name, running on this machine's cards, is competing for
        # memory with every model that is named: it is not a second half of anything, it is a leftover.
        # Compared by container, because several of this host's models share one container name across
        # configurations -- the question is whether a container is standing that this launch did not
        # start, not which model id it is filed under.
        started = {cfg['container_name'] for cfg in bundle['compose']['services'].values()}
        for model, container in self.host_models().items():
            if container in started:
                continue
            stray = self.system.inspect(container)
            if stray and stray['State']['Running']:
                for entry in observed.values():
                    entry['differences'].append('competing backend running: ' + container)
                    entry['healthy'] = False
        legacy = self.system.inspect('local-ai-llama-cpp')
        require(not legacy or not legacy['State']['Running'], 'Legacy inference backend is running; refusing competing GPU ownership')
        return observed

    def inspect(self, profile=None, full_hash=False):
        bundle = self.desired(profile)
        validation = self.validate(bundle, full_hash=full_hash)
        services = self.observe(bundle)
        router = self.router_view()
        return {'mode': 'inspection', 'revision': self.revision, 'profile': bundle['profile'],
            'config_sha256': bundle['config_sha256'], 'validation': validation, 'services': services,
            'matches_live': all(s['healthy'] and not s['differences'] for s in services.values()),
            'router_draining': router['runtime']['draining'],
            'active_requests': router['runtime']['active_count'],
            'queued_requests': router['runtime'].get('queued_count', 0)}

    def drain_pass(self, configs):
        """One look at the router and every running backend: (work the router relays, notes).

        The router's own counters are the authority on whether a generation is being served to
        anyone. A backend is only evidence of work when it answers `/slots` and a slot says it is
        processing: a backend whose `/slots` cannot be fetched is serving nothing, and a response
        this code cannot read is not evidence either. The trial's second failure was here — the
        engine that replaced llama.cpp answered no port at all, `except` pinned the pass noisy
        forever, and the drain check could never pass when that failed backend was the only
        container left standing. That is the exact case restoration exists for, so an unreachable
        backend is recorded and never holds the drain open.
        """
        # Work is counted per router and summed, because the point of the wait is that nobody is
        # generating anywhere. A machine's own queue on a shared router is `active_by_host` for its
        # own id: another machine's long generation is not this machine's maintenance window, and a
        # drain that waited for it would be a second machine holding a switch hostage indefinitely.
        relayed, notes = 0, {'unreachable': [], 'busy': [], 'malformed': []}
        for sink, report in self.drain_reports():
            relayed += report['active'] + report['queued']
        for name, cfg in configs.items():
            ci = self.system.inspect(name)
            if not ci or not ci['State']['Running']:
                continue
            try:
                slots = self.system.http(self.backend_url(cfg) + '/slots', timeout=5)
            except Exception:
                notes['unreachable'].append(name)
                continue
            if not isinstance(slots, list):
                notes['malformed'].append(name)
                continue
            if any(x.get('is_processing', False) for x in slots if isinstance(x, dict)):
                notes['busy'].append(name)
        return relayed, notes

    def wait_idle(self, bundles, timeout=300, force_on_deadline=False):
        """Wait until nothing is in flight, and return what the last pass saw.

        `force_on_deadline` is for a caller already committed to bringing a backend back. The
        point of a drain is to not interrupt work in flight; once the deadline passes and the
        router is relaying nothing, there is nobody left to be polite to and the router already
        rejects everything, so the only remaining outcome is to continue. Aborting there would
        leave the host drained with nothing serving — which is what happened at 01:12 on
        2026-10-12 and stayed that way until someone rebooted the machine.

        Without it, a deadline is a refusal before anything is touched: the router still reports
        work, or a reachable backend is generating for a client of its own (a direct request to a
        backend port is a legitimate thing on this host), and the caller has not destroyed
        anything yet, so backing away is the polite and correct answer.
        """
        configs = {cfg['container_name']: cfg for b in bundles if b for cfg in b['compose']['services'].values()}
        deadline, stable = time.monotonic() + timeout, 0
        while time.monotonic() < deadline:
            relayed, notes = self.drain_pass(configs)
            quiet = relayed == 0 and not notes['busy']
            stable = stable + 1 if quiet else 0
            if stable >= DRAIN_SETTLE_ROUNDS:
                return notes
            time.sleep(1)
        if force_on_deadline:
            relayed, notes = self.drain_pass(configs)
            if relayed == 0:
                return {**notes, 'forced': True}
        raise RuntimeError('Drain timed out; no active generation was stopped')

    def compose_path(self, bundle):
        folder = self.state / 'generated' / bundle['config_sha256']
        self.save(str(folder.relative_to(self.state) / 'compose.json'), bundle['compose'])
        return folder / 'compose.json'

    def prepare(self, bundle):
        self.validate(bundle)
        path = self.compose_path(bundle)
        self.system.docker('compose', '-p', bundle['compose']['name'], '-f', str(path), 'config', '--quiet')

    def reconcile(self, bundle, services, sequential=False, recreate=False):
        """Start the named services; across a switch they come up one at a time, in the order the
        configuration names its models, so a launch takes the memory it was sized for before the next
        one asks for cards that may still be busy.

        `recreate` forces a container away even where Compose would keep the running one, which is
        what a switch asks for: a model whose launch is byte-identical in both configurations is
        still stopped and loaded again, because the machine is being told to run this configuration
        and a process quietly carried over from another one is precisely what this runtime exists not
        to have to reason about later. Recovery leaves it false -- bringing the previous release back
        serving is the whole job there, and an untouched healthy container is the fastest right answer
        as long as it is the right container."""
        if services:
            path = self.compose_path(bundle)
            order = [entry['service'] for entry in bundle['manifest']['services']]
            batches = [[service] for service in order if service in services] if sequential \
                else [services]
            for batch in batches:
                # Engines that prepare their own state before they answer the first health check
                # (Strata builds its expert pack and installs its draft layer on a fresh workspace)
                # name their own bounded readiness window; everything else keeps the usual one.
                wait = max([service_engine(bundle, service).get('readiness_wait_seconds', 450)
                    for service in batch])
                self.system.docker('compose', '-p', bundle['compose']['name'], '-f', str(path),
                    'up', '-d', '--no-deps', '--pull', 'never',
                    *(('--force-recreate',) if recreate else ()),
                    '--wait', '--wait-timeout', str(wait), *batch, timeout=wait + 60)
        observed = self.observe(bundle)
        # Docker's health state is the engine's own /health, and an engine whose server answers
        # before the model is resident (Strata's /health says status ok from server start, carries
        # the model state as loaded, and /slots stays empty until the read completes) is healthy
        # for Docker while it is still reading: up --wait came back long before the model did, and
        # the one-shot observe above rejected it while the read was mid-flight. For those, the
        # declared readiness window is the bound on waiting for the controller's own view. A
        # backend that loaded with the wrong window, or drifted in any other way, is not waited
        # out: the window would only turn the fault into a delay.
        if services:
            wait = max([service_engine(bundle, service).get('readiness_wait_seconds', 450) for service in services])
            deadline = time.monotonic() + wait
            while not all(s['healthy'] for s in observed.values()) and self.still_loading(observed) \
                    and time.monotonic() < deadline:
                time.sleep(READINESS_POLL_SECONDS)
                observed = self.observe(bundle)
        bad = [(service, s) for service, s in observed.items() if not s['healthy']]
        for service, s in bad:
            # The record must carry the witness while the container still exists: restoration
            # removes it, and an engine that stopped by itself has no health and no slots left to
            # explain why. Its exit code and its own last words are the only account of the
            # failure, so they are read here, before anything can delete them.
            ci = self.system.inspect(s.get('container_name') or '')
            if ci:
                if not ci['State']['Running']:
                    s['exit_code'] = ci['State'].get('ExitCode')
                try:
                    words = self.system.docker('logs', '--tail', '120', s['container_name'], check=False)
                    lines = [line.strip() for line in words.splitlines() if line.strip()]
                    s['last_words'] = [line[:180] for line in lines[-10:]]
                except Exception:
                    pass
        detail = '; '.join(service + ': ' + ', '.join(
            s['differences']
            + (['slot window ' + json.dumps(s['slot_window'])] if s.get('slot_window') else [])
            + (['health: ' + s['health_error']] if s.get('health_error') else [])
            + (['exit ' + str(s['exit_code'])] if s.get('exit_code') is not None else [])
            + (['last words: ' + ' | '.join(s['last_words'])] if s.get('last_words') else []))
            for service, s in bad)
        require(not bad, 'Backend identity/readiness verification failed' + (': ' + detail if detail else ''))
        return observed

    @staticmethod
    def still_loading(observed):
        """The one unhealthy state a readiness window can fix: an engine that answers its own
        health check before the model is resident. Strata's /health says status ok from server
        start and carries the model state as loaded; /slots stays empty until the read completes,
        so the observer records slot context with loaded=false. That is a backend mid-load, and
        waiting within the declared window is the fix. The same empty list with loaded=true, or
        any other difference at all, is a fault the window will not fix: saying so here is what
        keeps the transition from spending the window on it."""
        for service in observed.values():
            if service['healthy']:
                continue
            window = service.get('slot_window')
            if window is None or 'slot context' not in service['differences']:
                return False
            if any(reason != 'slot context' for reason in service['differences']):
                return False
            if window.get('loaded') is not False:
                return False
        return bool(observed)

    def acceptance(self, bundle, services):
        for service in services:
            cfg = bundle['compose']['services'][service]
            if service_engine(bundle, service).get('interface') == 'environment':
                # The native /completion probe belongs to llama.cpp; the same bounded-generation check
                # runs against the OpenAI chat dialect every engine in this house has to serve. A
                # reasoning model may spend its whole budget on thinking, so either channel counts.
                answer = self.system.http(self.backend_url(cfg) + '/v1/chat/completions', {
                    'messages': [{'role': 'user', 'content': 'Complete the word: O'}],
                    'max_tokens': 8, 'temperature': 0, 'stream': False}, timeout=120)
                choices = answer.get('choices') or []
                message = (choices[0].get('message') or {}) if choices else {}
                said = ' '.join(message.get(key) or '' for key in ('content', 'reasoning_content')).strip()
                require(bool(said), 'Bounded generation acceptance failed for ' + service)
                continue
            answer = self.system.http(self.backend_url(cfg) + '/completion', {
                'prompt': 'Complete the word: O', 'n_predict': 8, 'temperature': 0,
                'stream': False, 'cache_prompt': False}, timeout=90)
            require(isinstance(answer.get('content'), str) and bool(answer['content'].strip()),
                'Bounded generation acceptance failed for ' + service)
        self.wait_idle([bundle])

    def publish(self, bundle):
        observed = self.observe(bundle)
        require(all(x['healthy'] for x in observed.values()), 'Refusing to publish an unverified backend catalog')
        catalog = copy.deepcopy(bundle['catalog'])
        # The name is stamped here and not baked into the bundle: a restoration and a re-announcement
        # publish the document they were handed, and a machine that had been renamed in the meantime
        # would otherwise put the old name back and keep it, because publication is an event and a
        # successful record of a stale name is never retried.
        names = catalog_names(self.config_dir, self.host, self.aliases()[0])
        for entry in catalog.get('offline_services', []):
            entry['display_name'] = names.get(entry['model'], entry['display_name'])
        for entry in catalog['models']:
            entry['display_name'] = names.get(entry['model'], entry['display_name'])
            service = next(x for x in bundle['manifest']['services'] if x['model_alias'] == entry['model'])
            ci = self.system.inspect(service['container_name'])
            entry['updated_at'] = now()
            if service_engine(bundle, service['service']).get('interface') == 'environment':
                # No argument vector to read an output ceiling from: this engine has none, and what
                # was inspected is its environment, so the record says so instead of naming argv.
                entry['runtime_output_policy'] = {'verification': 'docker-inspect-env',
                    'engine_sets_no_output_ceiling': True, 'verified_at': now(), 'container_id': ci['Id'],
                    'started_at': ci['State']['StartedAt'],
                    'env_sha256': hashlib.sha256(json.dumps(sorted(ci['Config'].get('Env') or [])).encode()).hexdigest()}
                continue
            entry['runtime_output_policy'] = {'n_predict': -1, 'reasoning_budget': -1,
                'reasoning_effort': 'default', 'verification': 'docker-inspect-argv',
                'verified_at': now(), 'container_id': ci['Id'], 'started_at': ci['State']['StartedAt'],
                'argv_sha256': hashlib.sha256(json.dumps(ci['Config']['Cmd']).encode()).hexdigest()}
        catalog.update(catalog['models'][0])
        # Identify the live configuration for router clients. Bundles rendered before this field
        # existed (a recovered previous release) still get an accurate id and exclusivity.
        configuration = catalog.get('configuration') or {'id': bundle['profile'],
            'exclusive': any(model.get('exclusive') for model in catalog['models'])}
        catalog['configuration'] = {**configuration, 'runtime_revision': self.revision, 'published_at': now()}
        require(not self.system.readonly, 'Publication forbidden in inspection mode')
        # Publication is per sink, and so is the receipt. Each sink records the document accepted
        # there and, for a push, the revision the router computed from those bytes: the router's own
        # contract calls that revision the receipt, and a runtime that did not compare it would be
        # publishing on optimism instead of on proof.
        previous = self.read_local_catalog()
        reports = []
        try:
            for sink in self.sinks():
                try:
                    reports.append(self.publish_sink(sink, catalog))
                except Exception as error:
                    if not self.mirror(sink) or not unreachable(error):
                        raise
                    reports.append({'kind': sink['kind'], 'name': sink['name'],
                        'error': str(error)[:400], 'unreachable_at': now()})
        except BaseException:
            # Whatever a sink wrote stays written: a catalog the router refused, or a catalog of a
            # release that is not live. The local sink is the one this runtime can put back exactly,
            # and the push sink needs no equivalent — a refusal there leaves the previously accepted
            # catalog in force by construction — so restoring the file converges both. An
            # unsupported catalog can also break runtime-state itself, which is why the prior
            # document goes back before recovery asks who owns the drain; admissions and release
            # receipts wait for backend recovery to verify the previous release.
            self.restore_local_catalog(previous)
            raise
        # A sink that was written but refused, or written and unreachable, is a promise this machine
        # intends to keep: the record carries when to try it again, which is what stops the loop that
        # reports health every few seconds from offering a router the same catalog five times a
        # minute for the length of an afternoon. A switch publishes whenever it likes; the backoff is
        # read only by the loop that has nothing else to do.
        self.save('publication.json', {'published_at': now(), 'runtime_revision': self.revision,
            'configuration': catalog['configuration']['id'], 'sinks': reports,
            **({'retry_at': future(ANNOUNCE_RETRY_SECONDS)}
                if any(report.get('error') for report in reports) else {})})

    def local_catalog_path(self):
        return Path(self.host['router_state_dir']) / 'active-model.json'

    def read_local_catalog(self):
        if not self.host.get('router_state_dir'):
            return None
        path = self.local_catalog_path()
        return read(path) if path.exists() else None

    def write_local_catalog(self, catalog):
        """Put one document in the local router's directory, or remove it when handed None."""
        if not self.host.get('router_state_dir'):
            return False
        path = self.local_catalog_path()
        if catalog is None:
            path.unlink(missing_ok=True)
        else:
            atomic_json(path, catalog)
        return True

    def restore_local_catalog(self, catalog):
        """Put a specific document back and make the local router read it again.

        The reload is the half that matters: the router re-reads the file on demand, and a file
        restored while the router still holds the refused catalog in memory has restored nothing.
        A reload that fails is not the failure being reported, so it is not allowed to hide it.
        """
        if not self.write_local_catalog(catalog):
            return
        try:
            self.admin('reload-config', {})
        except Exception:
            pass  # The original failure remains the diagnostic; recovery will retry.

    def publish_sink(self, sink, catalog):
        report = {'kind': sink['kind'], 'name': sink['name']}
        if sink['kind'] == 'file':
            self.write_local_catalog(catalog)
            self.admin('reload-config', {})
            self.verify_local(catalog)
            return report
        return self.publish_http(catalog, report)

    def verify_local(self, catalog):
        """The router on this machine, holding exactly this machine's document, serving it live.

        This is the historical check, unchanged: one router, one catalog, so every model it lists is
        one this runtime published, and `available` is a fact about a backend on this machine's own
        Docker network rather than about somebody's routing table.
        """
        state = self.admin('runtime-state')
        models = state.get('models', [])
        expected = {m['model']: m['context_length'] for m in catalog['models']}
        require({m['id'] for m in models} == set(expected) and all(
            m['x_ollama_router']['health']['available'] and m['x_ollama_router']['output_policy'] == 'unrestricted'
            and m['x_ollama_router']['context_window'] == expected[m['id']]
            and m['x_ollama_router'].get('default_output_tokens') is None
            and m['x_ollama_router'].get('max_output_tokens') is None for m in models),
            'Router catalog/readiness verification failed')

    def push_host_identity(self):
        """This machine as it names itself when it pushes to a router that serves many.

        The router accepts a `host` block in the catalog body — `{id}`, plus an optional
        `display_name` for an operator-facing label — and refuses it unless the id is the machine the
        push authenticated as. Declaring the identity in the document, and not only in the URL and
        token that carry it, is what lets a shared router label each model by where it runs and lets
        a client tell two machines apart when their model names collide.
        """
        identity = {'id': self.target['host']}
        name = self.host.get('display_name')
        if name:
            identity['display_name'] = name
        return identity

    def publish_http(self, catalog, report):
        """Send the whole catalog to the router that lives elsewhere and read the receipt back.

        The response is the proof: it repeats the revision computed from the bytes accepted, the
        default model, and the models now listed for this machine. A router that refused keeps serving
        whatever it accepted last, so a failed publish is a deployment that did not happen rather than
        a house with no models — which is why there is no undo path here.
        """
        # The host block is added to the push copy only. The local file sink's router source is
        # `local`, and stamping a remote machine's id into that document would be a name the local
        # router has every right to refuse.
        payload = {**catalog, 'host': self.push_host_identity()}
        try:
            receipt = self.remote_call('hosts/' + self.target['host'] + '/catalog', payload, method='PUT')
        except urllib.error.HTTPError as error:
            # The router answers a refusal with a code and a reason, and both are the whole point of
            # the answer: `409 CATALOG_ID_CONFLICT` names the service ID another machine already
            # serves, `404 HOST_NOT_DECLARED` means this machine's id is not in the router's list, and
            # `422` names the field of the document the reader could not accept. All three are a
            # different fix, and none of them is inferable from a bare 4xx.
            detail = ''
            try:
                detail = error.read().decode('utf-8', 'replace')
            except Exception:
                pass
            raise RouterRefused('The router at ' + self.target['url'] + ' refused this machine\'s catalog ('
                + str(error.code) + '): ' + detail[:600]) from error
        require(isinstance(receipt, dict) and receipt.get('revision'),
            'The router at ' + self.target['url'] + ' answered a catalog push without a revision: ' +
            json.dumps(receipt)[:400])
        require(receipt.get('ok', True) and receipt.get('default_model') == catalog['default_model'],
            'The router at ' + self.target['url'] + ' refused this machine\'s catalog: ' +
            json.dumps(receipt.get('error') or receipt.get('warnings') or receipt)[:600])
        report.update({'revision': receipt['revision'], 'accepted_at': now(),
            'warnings': sorted(set(receipt.get('warnings') or []))})
        report.update(self.verify_remote(catalog, receipt['revision']))
        return report

    def verify_remote(self, catalog, revision):
        """What a router two hosts away can confirm, and what it cannot be asked to confirm.

        It can confirm the document: each model this machine publishes is listed with the context
        window claimed and output and thinking uncapped. A shared router lists every machine,
        so this runtime verifies its own host's configuration and revision, independently of
        the model the router serves by default.

        What it cannot confirm is that the backend is alive, not without reporting on its own network.
        `available: false` from a remote router says *it* cannot open `backend_url`, which during the
        migration is true of a backend still published on loopback and expected to be true. So
        availability is required of a remote router only once this machine states a backend address —
        only once it has claimed the address is routable — and until then it is reported, in the
        status, as the difference between "running here" and "reachable from there".
        """
        state = self.remote_call('runtime-state')
        mine = {m['model']: m['context_length'] for m in catalog['models']}
        entries = [m for m in state.get('models', []) if m['id'] in mine]
        require({m['id'] for m in entries} == set(mine),
            'The router at ' + self.target['url'] + ' is not serving this machine\'s models: it lists ' +
            str(sorted(m['id'] for m in state.get('models', []))) + ', this machine published ' +
            str(sorted(mine)))
        require(all(m['x_ollama_router']['output_policy'] == 'unrestricted'
            and m['x_ollama_router']['context_window'] == mine[m['id']]
            and m['x_ollama_router'].get('default_output_tokens') is None
            and m['x_ollama_router'].get('max_output_tokens') is None for m in entries),
            'Router catalog verification failed on ' + self.target['url'] + ': a model is served with '
            'a context window or an output quota this machine did not publish')
        available = all(m['x_ollama_router']['health']['available'] for m in entries)
        # The refusal is reserved for a machine that has claimed a routable address and cannot
        # deliver on it. A machine that has said nothing about reachability — whose catalog still
        # carries a loopback address or a container name from before the migration — is in the state
        # the migration exists to leave, and failing its switches would keep the new router empty for
        # exactly as long as it takes to move the backends.
        # Recovery publishes the previous bundle, whose addresses may predate the LAN binding.
        # Current host settings describe the attempted release, not the one being restored.
        claims_address = bool(self.host.get('backend_address')) and all(
            urllib.parse.urlsplit(m['backend_url']).hostname == self.host['backend_address']
            for m in catalog['models'])
        require(available or (self.host.get('router_state_dir') and not claims_address),
            'The router at ' + self.target['url'] + ' cannot reach this machine\'s backends, and this '
            'machine states ' + str(self.host.get('backend_address')) + ' as the address it publishes them '
            'on. Either the router opens that address or the address is wrong; a catalog that would '
            'serve nothing is not worth switching to')
        roster = self.remote_call('hosts') or {}
        accepted = next((h for h in roster.get('hosts', []) if h['id'] == self.target['host']), {})
        # active_model.profile is the model's own label ("primary"), not configuration.id.
        # The per-host roster carries both the configuration and the stored revision, including
        # for a non-default host. Compare the receipt to what the router actually serves.
        require(accepted.get('published') and accepted.get('revision') == revision
            and set(accepted.get('model_ids') or []) == set(mine)
            and (accepted.get('configuration') or {}).get('id') == catalog['configuration']['id']
            and (accepted.get('configuration') or {}).get('runtime_revision') ==
                catalog['configuration']['runtime_revision'],
            'Router host-catalog verification failed on ' + self.target['url'])
        default = roster.get('default_host')
        return {'available': available, 'router_default': default,
            **({} if available else {'warnings': ['ROUTER_CANNOT_REACH_BACKEND:' + self.target['host']]})}

    def retire(self):
        """Tell every sink this machine no longer serves these models.

        A catalog whose backends nobody starts is worse than no catalog: clients are told a model
        exists and get a 503 from a machine that has stopped trying. The push sink has an explicit
        retirement for that; the local one retires the same way it publishes — the document going
        away.
        """
        require(not self.system.readonly, 'Mutation forbidden in inspection mode')
        retired = []
        for sink in self.sinks():
            if sink['kind'] == 'http':
                self.remote_call('hosts/' + self.target['host'] + '/catalog', {}, method='DELETE')
            else:
                self.write_local_catalog(None)
                self.admin('reload-config', {})
            retired.append(sink['name'])
        self.save('publication.json', {'retired_at': now(), 'runtime_revision': self.revision,
            'retired_from': retired})
        return retired

    def set_drain(self, enabled, transaction):
        """Hold traffic for this machine, on every router that is serving it.

        A switch that drains one router and not the other does not interrupt a generation, it changes
        which router's clients get interrupted — and the reason travels with the drain because the
        reason is how a router tells its owner from an intruder. A host-scoped drain is sent scoped
        when the sink is a push: on a shared router this machine's maintenance window is not the other
        machines' window, and pausing the whole deployment to replace one card is how a second machine
        makes the first one slower to maintain instead of faster.
        """
        reason = 'local-ai-runtime:' + transaction['id'] if enabled else 'runtime ready'
        results = []
        for sink in self.sinks():
            body = {'enabled': enabled, 'reason': reason}
            if sink['kind'] == 'http':
                body['host'] = self.target['host']
            try:
                results.append((sink['name'], self.admin_healing_for(sink, 'runtime-drain', body)))
            except Exception as error:
                if not self.mirror(sink) or not unreachable(error):
                    raise
                results.append((sink['name'], {'ok': False, 'unreachable': str(error)[:200]}))
        return results

    def owns_drain(self, transaction):
        """Whether this machine's drain is held under this transaction's reason.

        Any sink that says so is enough to release, because releasing goes to every sink: the case
        this answers for is a pass that drained one router and died before reaching the other, where
        the untouched router has nothing to release and the drained one must not be left holding
        traffic with no transaction behind it.
        """
        reason = 'local-ai-runtime:' + transaction['id']
        return any(report['draining'] and report['reason'] == reason
            for _, report in self.drain_reports())

    def mark(self, transaction, phase, step=None):
        """Write the phase a pass has reached into the transaction before doing anything else.

        A restoration is the one path whose record has to be readable afterwards without a
        journal: it runs while the machine is not serving, so every step it reached is recorded
        beside the phase, and the transaction is the evidence of what was attempted.
        """
        transaction['phase'] = phase
        if step is not None:
            transaction.setdefault('restoration', {}).setdefault('phases', []).append({'step': step, 'at': now()})
        self.save('transaction.json', transaction)

    def finish(self, transaction, state, error=None, serving_profile=None):
        transaction.update(phase=state, finished_at=now(), error=error)
        self.save('transaction.json', transaction)
        # `phase` and `error` are what a person reads first, and `intent` says which kind of attempt
        # this was: a failed switch and the restoration that put the machine back afterwards
        # otherwise look identical in this file.
        record = {k: transaction.get(k) for k in
            ('id', 'intent', 'phase', 'profile', 'started_at', 'finished_at', 'error', 'changed_services', 'revision', 'config_sha256')}
        record['intent'] = transaction.get('intent') or 'deployment'
        self.save('last-deployment.json', record)
        if state in ('succeeded', 'recovered'):
            # A verified release is serving, so nothing further is owed. The restoration record
            # stays on disk as history; it simply stops being an action waiting to happen.
            # A recovered switch restored its previous release, not the target the transaction
            # records, so the note names the release that is actually serving.
            serving = serving_profile or transaction.get('profile')
            self.clear_restoration('The runtime is serving ' + str(serving) + ' and requests are open.')

    def last_known_good(self):
        """The release this host was last known serving, which is what a restoration brings back.

        `active.json` is written only after a deployment has verified its backends and reopened
        admissions, so a failed switch has not moved it: what served is still what the record
        says. That is why the boot-time recovery brought llama.cpp back on its own on
        2026-10-12 - after the reboot the controller started what the record named - and it is the
        same lever the in-process path, the CLI and the one-action button pull, not a second
        implementation of it.
        """
        return self.load('active.json') or self.load('previous.json')

    def serving(self, bundle=None):
        """Whether a healthy backend answers and admissions are open.

        This is the state the runtime must come to rest in. It is deliberately conservative: an
        unreadable router or a missing record is not serving.
        """
        try:
            bundle = (bundle if bundle is not None else (self.last_known_good() or {}).get('bundle'))
            if not bundle:
                return False
            if self.draining_for_me(healing=False):
                return False
            return bool(self.observe(bundle)) and all(s['healthy'] for s in self.observe(bundle).values())
        except Exception:
            return False

    def restoration(self):
        """The durable record that service has to be brought back, or None."""
        return self.load('restoration.json')

    def restoration_pending(self):
        record = self.restoration()
        return bool(record) and not record.get('finished_at')

    def restoration_due(self):
        record = self.restoration()
        if not record or record.get('finished_at'):
            return False
        # Same-format UTC timestamps, so a schedule compares correctly as text.
        return record.get('next_attempt_at', '') <= now()

    def schedule_restoration(self, error, transaction=None, attempted=True):
        """Record that the runtime is not serving and say when the next attempt runs.

        This is what makes a failed recovery an action instead of a condition: the host is left
        with a pending restoration and a time, so nobody has to be there for it. It is retried
        until the service comes back, with the wait capped, because stopping to try is the same
        as nobody being there.
        """
        current = self.restoration() or {}
        if current.get('finished_at'):
            current = {}
        attempts = int(current.get('attempts', 0) or 0) + (1 if attempted else 0)
        delay = RESTORATION_BACKOFF[min(attempts - 1, len(RESTORATION_BACKOFF) - 1)] if attempts else 0
        good = self.last_known_good() or {}
        phases = list(current.get('phases', []))
        phases.append({'step': 'scheduled', 'at': now(), 'after_seconds': delay, 'error': str(error)})
        record = {'id': current.get('id') or str(uuid.uuid4()), 'attempts': attempts,
            'requested_at': current.get('requested_at') or now(), 'updated_at': now(),
            'next_attempt_at': future(delay), 'profile': good.get('bundle', {}).get('profile'),
            'transaction': (transaction or {}).get('id') or current.get('transaction'),
            'reason': str((transaction or {}).get('error') or error), 'last_error': str(error),
            'phases': phases[-RESTORATION_PHASE_LOG:]}
        self.save('restoration.json', record)
        return record

    def clear_restoration(self, note=None):
        record = self.restoration()
        if not record or record.get('finished_at'):
            return record
        record.update(finished_at=now(), phase='restored', note=note)
        self.save('restoration.json', record)
        return record

    def restore_last_known_good(self, operation_id=None, progress=lambda phase: None):
        """Bring the last-known-good release back and reopen admissions, whatever the failure left.

        One action, no reboot, and no lever a person is without: it is the boot-time recovery
        path, callable at any time by the button, the CLI or the supervision loop. It uses the
        interrupted transaction when that transaction owns the router drain - it knows which
        backend it replaced - and otherwise opens its own against the release on record.
        """
        require(not self.system.readonly, 'Mutation forbidden in inspection mode')
        with lock(self.state / 'runtime.lock'):
            return self._restore_locked(operation_id, progress)

    def _restore_locked(self, operation_id=None, progress=lambda phase: None):
        """Caller must hold runtime.lock for the whole restoration, as every transition does."""
        require(not self.system.readonly, 'Mutation forbidden in inspection mode')
        before = self.restoration()
        try:
            return self._restore_attempt_locked(operation_id, progress)
        except Exception as error:
            # Failures before recover_locked (including an unreadable router admin API) must
            # advance the retry timer too. A recovery failure already scheduled its own retry.
            if self.restoration() == before and self.last_known_good() is not None:
                self.schedule_restoration(error, self.load('transaction.json', {}))
            raise

    def _restore_attempt_locked(self, operation_id=None, progress=lambda phase: None):
        transaction = self.load('transaction.json', {})
        if transaction.get('phase') in PENDING_PHASES:
            if self.owns_drain(transaction):
                self.recover_locked(transaction, restoring=True, progress=progress)
                return {'profile': transaction['previous']['bundle']['profile'], 'restored': True,
                    'serving': self.serving()}
            # It stopped before the drain, so no backend was replaced. Close the attempt; the
            # release on record is still the one that served, and that is what is brought up.
            self.finish(transaction, 'failed',
                transaction.get('error') or 'Interrupted before draining; prior runtime preserved')
        good = self.last_known_good()
        require(good is not None, 'No last-known-good release is recorded; operator attention required')
        # A router release reached through the admin API is the one owner of the drain this path
        # must respect. Everything else in this process holds runtime.lock for its whole pass, and
        # the caller holds it now, so a drain that is still set here belongs to an attempt that
        # stopped - which is exactly the flag restoration exists to clear, not a reason to refuse.
        require(not self.load('router-maintenance.json', {}).get('reserved'),
            'A router release reserves the runtime; finish or recover it first')
        if self.serving(good['bundle']):
            self.clear_restoration('A healthy backend is serving and requests are open.')
            return {'profile': good['bundle']['profile'], 'restored': True, 'already_serving': True, 'serving': True}
        transaction = {'id': operation_id or str(uuid.uuid4()), 'started_at': now(), 'phase': 'prepared',
            'intent': 'restore-last-known-good', 'previous': good, 'target': copy.deepcopy(good),
            'changed_services': [], 'revision': self.revision, 'profile': good['bundle']['profile'],
            'config_sha256': good['bundle']['config_sha256'],
            'restoration': {'operation': operation_id, 'phases': []}}
        self.save('transaction.json', transaction)
        self.set_drain(True, transaction)
        self.recover_locked(transaction, restoring=True, progress=progress)
        return {'profile': good['bundle']['profile'], 'restored': True, 'serving': self.serving()}

    def _recover_locked(self, transaction, progress, force_on_deadline):
        previous = transaction.get('previous')
        require(previous is not None, 'No previous release exists; recovery needs operator attention')
        require(self.owns_drain(transaction), 'Recovery does not own the router drain; refusing to change it')
        old = previous['bundle']
        target = transaction['target']['bundle']
        progress('restoring')
        self.mark(transaction, 'restoring', 'preparing')
        self.prepare(old)
        # A drain deadline must never strand the host: this pass exists to bring service back, and
        # the router already rejects everything, so there is nothing left to be polite to.
        self.wait_idle([old, target], force_on_deadline=force_on_deadline)
        self.mark(transaction, 'restoring', 'draining')
        topology = set(services_of(old)) != set(services_of(target))
        # Every model of the release being restored is asked for, not only the ones found unhealthy:
        # the failed switch stopped, freed, or half-started them all, and Compose recreates only what
        # is not already the right container. Nothing the failed target started is left standing beside
        # them, because a model the restored configuration does not name would be holding memory the
        # restored models were sized for.
        extra = self.removed_backends(old, target)
        self.release(extra)
        self.mark(transaction, 'restoring', 'recreating')
        services = list(services_of(old))
        after = self.reconcile(old, services, sequential=topology)
        self.acceptance(old, services)
        self.mark(transaction, 'restoring', 'verifying')
        self.publish(old)
        self.save('active.json', previous)
        self.set_drain(False, transaction)
        self.mark(transaction, 'restoring', 'publishing')
        self.finish(transaction, 'recovered', transaction.get('error'), serving_profile=old['profile'])

    def recover_locked(self, transaction, restoring=False, progress=lambda phase: None):
        """Restore the release a transaction recorded, through the path a healthy boot uses.

        Called by the recovery leg of a failed deployment, by `recover` and the boot supervisor,
        and by the one-action restore. `restoring` marks a pass whose purpose is to bring service
        back: it then refuses nothing the drain can be used to argue about, logs every step into
        the transaction, and schedules the next attempt instead of resting when it fails. Without
        it the pass is the undo a caller may still walk away from.
        """
        if not restoring:
            return self._recover_locked(transaction, progress, False)
        try:
            return self._recover_locked(transaction, progress, True)
        except Exception as error:
            self.schedule_restoration(error, transaction)
            raise

    def transition(self, profile=None, deploy=False, adopt=False, install=False):
        require(not self.system.readonly, 'Mutation forbidden in inspection mode')
        with lock(self.state / 'runtime.lock'):
            return self._transition_locked(profile, deploy=deploy, adopt=adopt, install=install)

    def _transition_locked(self, profile=None, deploy=False, adopt=False, install=False, *,
                           operation_id=None, progress=lambda phase: None):
        """Caller must hold runtime.lock for the entire transaction, including recovery."""
        require(not self.system.readonly, 'Mutation forbidden in inspection mode')
        require(not self.load('router-maintenance.json', {}).get('reserved'),
            'A router release reserves the runtime; finish or recover it first')
        pending = self.load('transaction.json', {})
        require(pending.get('phase') not in PENDING_PHASES,
            'An interrupted deployment needs explicit runtime recover before another operation')
        previous = self.load('active.json')
        require(previous is not None or adopt or install,
            'Runtime is not adopted; use the reviewed migration procedure, or install it on a machine '
            'that has never run this runtime')
        require(not previous or previous['revision'] == self.revision or deploy,
            'This controller has been superseded; it cannot reapply an older release')
        progress('checking')
        bundle = self.desired(profile)
        self.switchable(bundle)
        self.prepare(bundle)
        observed = self.observe(bundle)
        # An install is a machine with nothing on it, and only that. It exists because the migration
        # gate above is a gate about a *migrated* machine — a host whose backends were started by the
        # stack this runtime replaced, where adopting them without a switch is the reviewed act — and a
        # second machine has no such history: it has an empty Docker host and a model to load. Without
        # this, the only way onto a new machine was to start its backend by hand and then adopt it,
        # which manufactures the very receipt this runtime exists to earn.
        require(not install or (previous is None and not any(s['healthy'] for s in observed.values())),
            'An install is a machine with no active release and no healthy backend: adopt a migrated '
            'host instead')
        prior = services_of(previous['bundle']) if previous else {}
        drift = [service for service, s in observed.items() if not s['healthy'] or (previous and
            prior.get(service) != bundle['compose']['services'][service])]
        # A switch runs the models the target configuration names and nothing else. It is not a patch
        # applied to whatever happens to be loaded: a model that is identical in both bundles is still
        # stopped and started by a switch to a different configuration, because the cards under it are
        # being redistributed and a process that survived that would be surviving a lie about where it
        # runs. Nothing moves when the target launches what is already launched: the machine is left
        # alone and a redeploy of a revision that is serving costs nothing.
        #
        # The comparison is the Compose document -- the launch itself -- and not the bundle hash,
        # because a bundle also carries what things are called. An operator renaming a model changes
        # the hash, and a rename must never be a reason to stop a serving model and load it again.
        replacing = bool(previous) and previous['bundle']['compose'] != bundle['compose']
        services = list(bundle['compose']['services']) if replacing else drift
        # Cards move between models across a switch, so the models come up one at a time, in the order
        # the configuration names them: a launch takes the memory it was sized for before the next one
        # asks for cards that may still be busy.
        topology = set(prior) != set(bundle['compose']['services'])
        removed = self.removed_backends(bundle, previous['bundle'] if previous else None)
        require(not adopt or not (drift or removed), 'Adoption requires an exact match to healthy existing backends')
        require(not self.draining_for_me(),
            'A router this machine publishes to is drained for this machine by another operation')
        if previous and previous['revision'] == self.revision and previous['bundle'] == bundle \
                and not drift and not removed:
            return {'already_active': True, 'profile': bundle['profile']}
        unchanged = {service: s['id'] for service, s in observed.items() if service not in services}
        target = {'revision': self.revision, 'image': self.image, 'bundle': bundle, 'applied_at': now()}
        # Adoption's prior runtime is already known healthy and can be restored by this image.
        if adopt:
            previous = copy.deepcopy(target)
        tx = {'id': operation_id or str(uuid.uuid4()), 'started_at': now(), 'phase': 'prepared', 'previous': previous,
            'target': target, 'changed_services': services, 'revision': self.revision, 'profile': bundle['profile'],
            'config_sha256': bundle['config_sha256']}
        if removed:
            tx['removed_backends'] = removed
        self.save('transaction.json', tx)
        changed = False
        try:
            if self.preparation_needed(bundle):
                # The workspace build is safe beside the serving model, so it happens before the
                # drain: a failure here stops the switch at its edge, and nothing has been taken.
                self.mark(tx, 'preparing')
                progress('preparing')
                self.ensure_provisioned(bundle)
            self.set_drain(True, tx)
            self.mark(tx, 'draining')
            progress('draining')
            self.wait_idle([bundle, previous['bundle'] if previous else None])
            self.mark(tx, 'applying')
            progress('loading')
            changed = True
            self.release(removed)
            load_started = time.monotonic()
            after = self.reconcile(bundle, services, sequential=topology or replacing,
                recreate=replacing)
            require(all(after[r]['id'] == cid for r, cid in unchanged.items()), 'An unchanged backend was unexpectedly recreated')
            self.mark(tx, 'verifying')
            progress('verifying')
            self.acceptance(bundle, services)
            self._record_warm_boot(bundle, services, round(time.monotonic() - load_started, 1))
            self.publish(bundle)
            self.save('previous.json', previous)
            self.save('active.json', target)
            self.set_drain(False, tx)
            self.finish(tx, 'succeeded')
            return {'profile': bundle['profile'], 'changed_services': services, 'revision': self.revision}
        except BaseException as error:
            tx['error'] = str(error)
            if changed:
                # A backend has already been replaced, so this is no longer an undo the caller can
                # walk away from: it is the restoration, and it runs as one. If it cannot finish,
                # it schedules the next attempt rather than resting with the router drained.
                try:
                    progress('restoring')
                    self.recover_locked(tx, restoring=True)
                except BaseException as recovery:
                    self.finish(tx, 'needs-attention', f'Update failed: {error}; recovery failed: {recovery}')
                    raise RuntimeError('Deployment failed; router remains drained and recovery needs attention') from recovery
            else:
                if self.owns_drain(tx):
                    self.set_drain(False, tx)
                self.finish(tx, 'failed', str(error))
            raise RuntimeError('Deployment failed: ' + str(error)) from error

    def recover(self):
        require(not self.system.readonly, 'Mutation forbidden in inspection mode')
        with lock(self.state / 'runtime.lock'):
            tx = self.load('transaction.json')
            require(tx and tx['phase'] in PENDING_PHASES,
                'There is no interrupted deployment to recover')
            if tx['phase'] == 'prepared' and not self.draining_for_me():
                self.finish(tx, 'failed', 'Interrupted before draining; prior runtime preserved')
            else:
                self.recover_locked(tx, restoring=True)

    def rollback_release(self):
        """Recover a successful runtime apply if the new controller cannot become healthy."""
        require(not self.system.readonly, 'Mutation forbidden in inspection mode')
        with lock(self.state / 'runtime.lock'):
            previous = self.load('previous.json')
            active = self.load('active.json')
            require(previous and active, 'No prior release is available')
            require(active['revision'] == self.revision, 'Refusing rollback from a superseded updater')
            require(not self.draining_for_me(),
                'A router this machine publishes to is drained for this machine by another operation')
            tx = {'id': str(uuid.uuid4()), 'phase': 'prepared', 'started_at': now(), 'previous': previous,
                'target': active, 'revision': self.revision, 'profile': previous['bundle']['profile'],
                'error': 'Replacement controller failed readiness'}
            self.save('transaction.json', tx)
            self.set_drain(True, tx)
            try:
                self.recover_locked(tx, restoring=True)
            except BaseException as error:
                self.finish(tx, 'needs-attention', str(error))
                raise

    def republish(self):
        """Router-only releases may republish, but may not replace runtime source/configuration."""
        with lock(self.state / 'runtime.lock'):
            active = self.load('active.json')
            require(active and active['revision'] == self.revision, 'Controller does not own the deployed revision')
            require(self.load('transaction.json', {}).get('phase') not in PENDING_PHASES,
                'Interrupted runtime transaction blocks publication')
            self.publish(active['bundle'])

    def announce(self, status=None):
        """Say again, to every router, what this machine is serving — unless someone already was.

        Publication is an event and not a state: it happens when a configuration changes, and the
        router keeps the document afterwards. But a controller restart with nothing to change — the
        ordinary shape of a portal update, and of a machine rebooting onto a router that has been
        redeployed underneath it — goes through `transition()` reporting that everything is already
        active, and publishes nothing. A machine that only publishes on change is then healthy and
        unheard-from, which is the state this exists to prevent, so the loop that already asks
        whether this machine is serving asks whether anyone was told.

        A failure is recorded and not raised. The catalog not landing is exactly the fact the page is
        for, and a controller that refused to serve the page over it would delete the way to fix it.
        """
        active = self.load('active.json')
        if not active or active['revision'] != self.revision:
            return None  # Nothing of this revision is serving; there is nothing to claim.
        if self.load('transaction.json', {}).get('phase') in PENDING_PHASES:
            return None  # A switch is in flight and will publish with its own verification.
        # The same pass that asks whether the household was told what is serving asks whether anyone is
        # still parked behind a drain belonging to a run that has already finished.
        self.release_orphan_drain()
        record = ((status or {}).get('publication') if status is not None
            else self.load('publication.json', {})) or {}
        expected_sinks = {(s['kind'], s['name']) for s in self.sinks()}
        recorded_sinks = {(s['kind'], s['name']) for s in record.get('sinks', [])}
        if record and expected_sinks == recorded_sinks and not record.get('error') and not record.get('retired_at') \
                and not any(sink.get('error') for sink in record.get('sinks', [])):
            return None
        # Not this pass, then: the record carries when it is worth asking again. The loop that
        # publishes is the one that reports health every few seconds, and a router that is down for
        # the afternoon would otherwise be offered the catalog five times a minute by a machine that
        # is otherwise doing nothing but asking.
        if record.get('retry_at', '') > now():
            return None
        try:
            return self.publish(active['bundle'])
        except Exception as error:
            self.save('publication.json', {**(self.load('publication.json', {})),
                'failed_at': now(), 'retry_at': future(ANNOUNCE_RETRY_SECONDS),
                'runtime_revision': self.revision, 'configuration': active.get('profile'),
                'error': str(error)})
            return None

    def router_maintenance(self, begin):
        """Reserve runtime configuration across a router-only deployment's process boundary."""
        require(not self.system.readonly, 'Mutation forbidden in inspection mode')
        with lock(self.state / 'runtime.lock'):
            active = self.load('active.json')
            require(active and active['revision'] == self.revision, 'Controller does not own the deployed revision')
            reservation = self.load('router-maintenance.json', {})
            if begin:
                require(not reservation.get('reserved') and not self.draining_for_me(),
                    'Another maintenance operation is active')
                require(self.load('transaction.json', {}).get('phase') not in PENDING_PHASES,
                    'Interrupted runtime transaction')
                reservation = {'id': str(uuid.uuid4()), 'reserved': True, 'started_at': now()}
                self.save('router-maintenance.json', reservation)
                try:
                    self.set_drain(True, reservation)
                    self.wait_idle([active['bundle']])
                except BaseException:
                    if self.owns_drain(reservation):
                        self.set_drain(False, reservation)
                    self.save('router-maintenance.json', {**reservation, 'reserved': False, 'failed_at': now()})
                    raise
            else:
                require(reservation.get('reserved') and self.owns_drain(reservation), 'No owned router maintenance reservation')
                self.wait_idle([active['bundle']])
                self.publish(active['bundle'])
                self.set_drain(False, reservation)
                self.save('router-maintenance.json', {**reservation, 'reserved': False, 'finished_at': now()})

    def status(self):
        active = self.load('active.json')
        bundle = active['bundle'] if active else self.desired()
        # Which machine is answering, and what it has to place a model on. The page shows a GPU rail
        # and has to say how many cards this host has instead of assuming the number the source was
        # written against, and a person reading a screenshot cannot otherwise tell a host with two
        # cards from a host with four that lost two.
        declared = self.host.get('gpu_ids') or {}
        groups = [group for group in PAIR_GROUPS if group in declared]
        # A private setting with a typo in it is reported as one, not raised: this is what the health
        # poll calls every few seconds and what an operator reads to find out what to fix, and a
        # machine that is serving fine would otherwise be reported as a page that does not load.
        try:
            architectures = gpu_architectures(self.host)
            host_report = {'id': host_id(self.host),
                'gpu_architectures': list(architectures) if architectures else None,
                'text_gpu_groups': groups, 'text_gpus': sum(len(declared[g]) for g in groups),
                'vision_gpu_id': self.host.get('vision_gpu_id')}
        except Exception as error:
            host_report = {'id': None, 'error': str(error), 'text_gpu_groups': groups,
                'text_gpus': sum(len(declared[g]) for g in groups),
                'vision_gpu_id': self.host.get('vision_gpu_id')}
        aliases, alias_note = self.aliases()
        # `published_names` and not `names`: the loop below already has a `names`, and it maps GPU
        # UUIDs to card names, which is a different question with a similar answer.
        published_names = catalog_names(self.config_dir, self.host, aliases)
        result = {'revision': self.revision, 'deployed_revision': active['revision'] if active else None,
            'profile': bundle['profile'], 'ready': False, 'services': [],
            'host': {**host_report, 'gpu_inventory': self.gpu_inventory()},
            # What the cards are actually doing with their power, from the driver, on its own cadence
            # -- and the reason a write cannot start, when one cannot. The cache of it, never a read:
            # this is what the five-second poll calls.
            'power': self.power.status(),
            # What a client can still ask for out of this machine's catalog and nothing answers,
            # because the configuration that runs does not load the model that answers it. There is no
            # second fact about placement: a configuration loads models, a model it does not load is
            # not running, and the services list above already says which is which.
            'offline_services': copy.deepcopy(bundle['catalog'].get('offline_services', [])),
            'last_deployment': self.load('last-deployment.json'), 'updated_at': now()}
        # Read-only presentation of the profile registry. It is reported before the live
        # checks below so an unreadable definition never hides the health of the pair. Each
        # selectable configuration carries the same verdict the read-only gate gives: what a
        # switch to it would do (notes) and what would refuse it (reasons), so the page can
        # show the cost of a selection before the selection is made.
        try:
            configurations = available_profiles(self.config_dir, self.host, aliases)
            for entry in configurations.get('selectable', []):
                try:
                    candidate = render(self.config_dir, self.host, entry['profile'], aliases)
                    entry['reasons'] = self.verdict(candidate)
                    entry['notes'] = self.verdict_notes(candidate)
                except Exception as error:
                    entry['reasons'] = ['This configuration cannot be rendered: ' + str(error)]
                    entry['notes'] = []
            result['configurations'] = {'active': bundle['profile'], **configurations}
        except Exception as error:
            result['configurations'] = {'active': bundle['profile'], 'selectable': [],
     'error': str(error)}
        update = self.load('update-job.json')
        result['source_deployment'] = {k: update.get(k) for k in
            ('phase', 'started_at', 'finished_at', 'error', 'revision')} if update else None
        if update and (update.get('finished_at') or update['started_at']) > (
                (result['last_deployment'] or {}).get('finished_at') or ''):
            result['last_deployment'] = {k: update.get(k) for k in
                ('phase', 'started_at', 'finished_at', 'error', 'revision')}
        # The service-restoration contract, where a page or a person can read it: either a backend
        # serves, or a restoration is running or scheduled. Reported ahead of the live checks
        # below, so an unreadable router can never hide the fact that nothing is serving.
        good = self.last_known_good()
        result['restoration'] = self.restoration()
        result['service'] = {'serving': self.serving(bundle), 'restorable': good is not None,
            'pending': self.restoration_pending(),
            'profile': (good or {}).get('bundle', {}).get('profile')}
        # Names the operator chose, and the reason they might not be in use: a saved file that
        # cannot be read is shown as that rather than quietly falling back to defaults nobody picked.
        result['aliases'] = {'assignments': aliases, **({'note': alias_note} if alias_note else {})}
        try:
            observed = self.observe(bundle)
            names = {gpu: name for group, ids in self.host.get('gpu_ids', {}).items()
                for gpu, name in zip(ids, self.host.get('gpu_names', {}).get(group, []))}
            for model, service in zip(bundle['catalog']['models'], bundle['manifest']['services']):
                live = observed[service['service']]
                text = model.get('text_gpu_uuids', model['gpu_uuids'])
                result['services'].append({'service': service['service'],
                    # The active bundle is the document this machine was switched with, so its names
                    # are the names of that moment; an alias saved since then is what the operator is
                    # reading now, and the page must not show a name it has already retired.
                    'name': published_names.get(model['model'], model['display_name']),
                    'model': model['model'],
                    'context_tokens': model['context_length'],
                    'gpu_ids': text,
                    'vision_gpu_id': model.get('vision_gpu_uuid'),
                    'vision_gpu_name': self.host.get('vision_gpu_name') if model.get('vision_gpu_uuid') else None,
                    'vision_device': model.get('vision_device', 'CPU'),
                    'vision_gpu_shared': model.get('vision_gpu_shared', False),
                    'exclusive': model.get('exclusive', False),
                    'gpu_names': [names[gpu] for gpu in text if gpu in names],
                    **live})
            router = self.router_view()
            state = router['runtime']
            # Requests in flight as one router sees them; see router_view for which one that is.
            result['maintenance'] = {'draining': self.draining_for_me(healing=False),
                'active_requests': state['active_count'],
                'queued_requests': state.get('queued_count', 0)}
            tx = self.load('transaction.json', {})
            expected = {m['model']: m['context_length'] for m in bundle['catalog']['models']}
            # A shared router lists every machine's models, so readiness is about the ones this machine
            # publishes. Asking the whole list whether it is this machine's document turns a second
            # machine joining the deployment into this machine reporting itself unhealthy, which is the
            # same category of mistake the drain scoping fixes.
            mine = [m for m in router.get('models', []) if m['id'] in expected]
            catalog_ready = {m['id'] for m in mine} == set(expected) and all(
                m['x_ollama_router']['health']['available'] and
                m['x_ollama_router']['context_window'] == expected[m['id']] for m in mine)
            # Where this machine's catalog lives. Two different failures look identical from a
            # terminal — models running here, nothing answering there — and are fixed in opposite
            # directions: one is a backend, the other is a router that has not been told. The
            # publication record is the difference, and the warning a router reports about its own
            # network belongs on this page and not in a log line: `available: false` from a router
            # two hops away says the router cannot open the address, which is a fact about the
            # address, and never a reason to call this machine unhealthy.
            result['publication'] = self.load('publication.json')
            result['ready'] = bool(active and active['revision'] == self.revision and
                all(x['healthy'] for x in observed.values()) and catalog_ready and
                not self.draining_for_me(healing=False) and
                not self.load('router-maintenance.json', {}).get('reserved') and
                tx.get('phase') not in PENDING_PHASES)
        except Exception as error:
            result['error'] = str(error)
        return result
