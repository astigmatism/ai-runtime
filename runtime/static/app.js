'use strict';
/* The page has one job: show which physical GPU is doing what, and offer the other configurations.
   Names come from the registry in /api/status, so adding or renaming a configuration never needs a
   change here — and neither does an alias, which arrives as `alias` on the same entries.

   Two rules are load-bearing for the Content-Security-Policy this page is served under
   (`script-src 'self'; style-src 'self'`): no inline script and no inline style. Everything visual
   is a class, so nothing here writes a style attribute, and nothing writes markup — a configuration
   id and an operator's alias are text, and they are set as text. */
const el = id => document.getElementById(id);
const node = (tag, text = '', cls = '') => {
  const element = document.createElement(tag);
  element.textContent = text;
  if (cls) element.className = cls;
  return element;
};
const add = (parent, ...children) => { parent.append(...children); return parent; };

/* ------------------------------------------------------------- the registry, as the page needs it */
const registry = () => status?.configurations || {};
const configurations = () => registry().selectable || [];
const configFor = profile => configurations().find(c => c.profile === profile);
// The second resident a paired configuration loads. `everyday` is its role; `nighttime` is the
// service ID a client sends to reach it, and the two are not the same fact.
const everyday = profile => {
  const variants = registry().always_included || [];
  return variants.find(v => v.profile === profile) || variants[0];
};
// Whether a configuration answers a service ID at all, whichever resident it points it at.
const answers = (config, id) => !!config && (config.service_ids || []).includes(id);
/* What a configuration is called: the operator's alias when they saved one, and the default the
   name rule produces otherwise. The server computes both; the page never assembles a name, because
   the router and the CLI publish the same string and must not be reading a fourth opinion. */
const nameOf = entry => entry?.display_name || entry?.profile || 'Unknown configuration';
const defaultNameOf = entry => entry?.default_name || entry?.display_name || entry?.profile || '';
const gpuList = entry => entry?.gpu_names?.length ? entry.gpu_names.join(' + ') : 'GPUs unavailable';
// A machine with one resident is not "both residents ready", and a page that says it has read the
// machine wrong: the second resident does not exist on every host.
const readyLine = () => {
  const loaded = (status?.services || []).length;
  return loaded === 1 ? 'Resident ready' : loaded >= 2 ? 'Both residents ready' : 'Ready';
};
const serviceIds = entry => (entry?.service_ids || ['nighttime']).join(', ');
// How a configuration that runs alone owns the machine, from the cards this host has: "all the text
// cards" is a different number on a two-card machine than on a four-card one, and a count of two
// reads better as a word than as a digit.
const ownsTextGpus = () => {
  const count = (inventory().text || []).length || status?.host?.text_gpus || 0;
  return !count ? 'every text GPU of this host' : count === 2 ? 'both text GPUs' : `all ${count} text GPUs`;
};
const context = tokens => Number.isFinite(tokens) ? `${tokens / 1024}K context` : 'Context unavailable';
const residentRow = entry => entry ? `${nameOf(entry)} · ${gpuList(entry)}` : 'Unavailable';

/* ------------------------------------------------------------------ the machine, as a set of cards */
const inventory = () => status?.host?.gpu_inventory || {text: [], vision: null};
const cards = () => [...(inventory().text || []), ...(inventory().vision ? [inventory().vision] : [])];
// Owners keyed by GPU UUID. Two facts of very different strength feed it: the running services say
// what is placed now, by UUID; a registry entry says what a switch would place, by GPU group, which
// is the same order the host declares its cards in.
const placementNow = () => {
  const owners = {};
  for (const service of status?.services || []) {
    for (const id of service.gpu_ids || []) owners[id] = {role: service.role, service};
    if (service.vision_gpu_id) {
      owners[service.vision_gpu_id] = {role: 'vision', service, shared: service.vision_gpu_shared === true};
    }
  }
  return owners;
};
function placementOf(config) {
  const owners = {};
  if (!config) return owners;
  const second = everyday(config.everyday);
  for (const card of inventory().text) {
    if (config.exclusive) owners[card.uuid] = {role: 'coding'};
    else if (config.gpu_group && card.group === config.gpu_group) owners[card.uuid] = {role: 'coding'};
    else if (second && card.group === second.gpu_group) owners[card.uuid] = {role: 'everyday'};
  }
  // The vision card is only owned by a configuration that would put something on it, and only on a
  // host that has one: what a profile declares is what it is for, what the host has is what it gets.
  if (config.vision_gpu && inventory().vision) owners[inventory().vision.uuid] = {role: 'vision'};
  return owners;
}
const roleName = role => role === 'everyday' ? 'Everyday' : role === 'vision' ? 'Vision' : 'Coding';
function cardNode(card, owner, preview = false) {
  const tile = node('div', '', `gpu ${owner?.role || 'free'}${preview ? ' preview' : ''}`);
  add(tile, node('h3', card.name || 'GPU'), node('div', '', 'chips'));
  const chips = tile.querySelector('.chips');
  if (owner?.role) {
    add(chips, node('span', roleName(owner.role), `mini ${owner.role === 'vision' ? 'vision' : owner.role}`));
    if (owner.role === 'vision' && owner.shared === false) add(chips, node('span', 'exclusive', 'mini vision'));
    if (owner.role === 'vision' && owner.service?.vision_device) add(chips, node('span', owner.service.vision_device, 'mini vision'));
    if (owner.service && !owner.service.healthy) add(chips, node('span', 'unwell', 'mini off'));
    if (owner.service && admissionsPaused()) add(chips, node('span', 'paused', 'mini off'));
  } else add(chips, node('span', 'nothing', 'mini off'));
  // Whether the vision card is shared with a second resident is a fact about a running service, and
  // a preview has none, so a preview tile says only what a configuration is known to put there.
  const detail = owner?.role === 'vision'
    ? 'image encoder' + (owner.shared === undefined ? '' : owner.shared ? ' · shared' : ' · exclusive')
    : owner?.role ? (card.group === 'daytime' ? 'daytime pair' : 'nighttime pair') : 'free';
  add(tile, node('p', preview && !owner?.role ? 'stays free' : detail, 'own'));
  return tile;
}
function healthOf(service) {
  if (!connected) return 'Status unavailable';
  if (!service.running) return 'Not running';
  if (!service.healthy) return 'Unavailable';
  return admissionsPaused() ? 'Loaded · new requests paused' : 'Ready';
}
function residentNode(entry, {service, everyday: second = false, state, detail, tag} = {}) {
  const card = node('article', '', `resident${second ? ' everyday' : ''}${state && state !== 'Ready' ? (service && service.healthy === false ? ' bad' : ' unwell') : ''}`);
  // The text gets a column of its own: `add` returns the parent it appended to, and a left hand
  // written as `add(card, node('div'))` is the card, which turns every line of a resident into a
  // flex sibling and squeezes a model name to the width of the word beside it.
  const left = node('div');
  add(card, left);
  add(left,
    node('p', 'role', 'role'),
    node('p', '', 'name'),
    node('p', '', 'alias'));
  const role = entry?.role || (service ? service.role : (second ? 'everyday' : 'coding'));
  card.querySelector('.role').textContent = `${role} resident`;
  // The house calls the second resident's service `nighttime` whether or not the registry had an
  // entry to say so, because that is what a client sends and what an operator is looking for.
  const ids = entry?.service_ids?.length ? entry.service_ids : (role === 'everyday' ? ['nighttime'] : null);
  if (ids) card.querySelector('.role').textContent += ' · ' + ids.join(' · ');
  // A resident the configuration does not load has no name to show, and `Nothing` is the truth.
  card.querySelector('.name').textContent = entry ? nameOf(entry) : 'Nothing';
  card.querySelector('.alias').textContent = detail || '';
  add(card, node('span', state || 'Unavailable', 'state'));
  if (tag) add(card.querySelector('.name'), node('span', ' ' + tag, 'tag new'));
  return card;
}

/* --------------------------------------------------------------------- rendering: the map itself */
function renderMap() {
  const residents = el('residents'), board = el('gpus');
  const services = status?.services || [];
  const signature = JSON.stringify([
    services.map(s => [s.role, s.name, s.model, s.context_tokens, s.gpu_ids, s.gpu_names, s.vision_gpu_id,
      s.vision_gpu_name, s.exclusive, s.healthy, s.running]),
    status?.offline_roles, status?.offline_services, status?.host?.gpu_inventory,
    status?.maintenance?.draining, status?.configurations?.active, connected,
    running() ? [operation.target_profile, operation.phase] : null]);
  if (signature === mapSignature) return;
  mapSignature = signature;
  residents.replaceChildren(); board.replaceChildren();
  if (!connected) {
    add(residents, node('p', 'Runtime status is unavailable. Nothing below is a claim about this machine.', 'sub'));
    return;
  }
  if (!services.length && !status?.offline_roles?.length) {
    add(residents, node('p', 'No resident is loaded. The configuration is shown below as a wish, not a fact.', 'sub'));
  }
  const owners = placementNow();
  for (const service of services) {
    // The resident's own registry entry, so the service IDs it publishes are the ones a client
    // sends to *it* — the coding resident's IDs are not the second resident's.
    const entry = service.role === 'everyday' ? everyday(configFor(status.profile)?.everyday)
      : configFor(status.profile);
    add(residents, residentNode(entry, {
      service, state: healthOf(service),
      detail: `${service.model} · ${context(service.context_tokens)}`}));
  }
  // A resident that is not loaded is only worth a card when its service ID is this machine's
  // business at all: either the live configuration answers it — a solo model that answers `nighttime`
  // from its own backend has the role offline and no offline service, and "off" there would be a lie
  // about a model that is answering — or the catalog publishes it as deliberately unanswered. A host
  // whose configurations route to a service of their own, with no second resident in its registry,
  // has no second resident to draw at all, and a card about one would be an invention.
  const nighttime = everyday()?.service_ids?.[0] || 'nighttime';
  const stopped = (status?.offline_services || []).find(s => (s.aliases || []).includes(nighttime));
  if (status?.offline_roles?.includes('everyday') && (stopped || answers(configFor(status.profile), nighttime))) {
    // A second resident that is not loaded is only worth a card when nothing answers its service ID
    // either: a solo configuration that answers `nighttime` from its own model has the role offline
    // and no offline service, and "off" there would be a lie about a model that is answering. And
    // with nothing loaded at all, the claim that it answers is worse than a silence.
    const answered_by = stopped || !services.length ? null : configFor(status.profile)?.model;
    add(residents, residentNode(stopped || everyday(), {
      everyday: true,
      state: answered_by ? 'Answers its requests' : 'Not loaded',
      detail: answered_by ? `Not a separate process · answered by ${answered_by}`
        : stopped ? `${gpuList(everyday())} are used by ${nameOf(configFor(status.profile))}`
          : `Nothing is loaded, so nothing answers ${serviceIds(everyday())}`}));
  }
  for (const card of cards()) add(board, cardNode(card, owners[card.uuid]));
}

/* A selected configuration that is not the live one: the same machine, redrawn as a promise, with
   the difference in ownership spelled out because that difference is the whole cost of the switch. */
function renderPreview() {
  const panel = el('preview');
  const target = configFor(selected);
  const show = connected && target && target.profile !== status?.profile && !running();
  panel.hidden = !show;
  if (!show) return;
  el('preview-name').textContent = nameOf(target);
  const before = placementNow(), after = placementOf(target);
  const residents = el('preview-residents'), board = el('preview-gpus');
  residents.replaceChildren(); board.replaceChildren();
  add(residents, residentNode(target, {
    state: 'Would load', detail: `${target.model} · ${context(target.context_tokens)}`}));
  const second = everyday(target.everyday);
  if (!target.exclusive && second) {
    add(residents, residentNode(second, {everyday: true, state: 'Would load', detail: gpuList(second)}));
  } else if (target.exclusive) {
    add(residents, residentNode(null, {everyday: true, state: answers(target, 'nighttime') ? 'Answers its requests' : 'Stops',
      detail: answers(target, 'nighttime') ? `${nameOf(target)} answers nighttime from its own backend`
        : `Nothing would answer the Everyday resident's requests`}));
  }
  const moved = [], freed = [], taken = [];
  for (const card of cards()) {
    const was = before[card.uuid]?.role, now = after[card.uuid]?.role;
    if (was !== now) {
      if (!was) taken.push(card.name); else if (!now) freed.push(card.name); else moved.push(`${card.name} → ${roleName(now)}`);
    }
    add(board, cardNode(card, after[card.uuid], true));
  }
  const changes = [['Would place', taken], ['Would change owner', moved], ['Would free', freed]]
    .filter(([, list]) => list.length)
    .map(([label, list]) => `${label} ${list.join(', ')}`);
  const exclusiveNow = configFor(status.profile)?.exclusive;
  const note = [changes.length ? changes.join('; ') + '.' : 'No card changes owner.',
    target.exclusive && !exclusiveNow ? 'The Everyday resident stops for the length of the switch.'
      : !target.exclusive && exclusiveNow ? 'The Everyday resident starts again.' : '',
    // A refusal is said before anything else would be done about the cards, and it is the one thing
    // the switch would not be able to work around; a note is a notice about how it would get there.
    ...(target.reasons?.length ? [target.reasons[0]] : target.notes?.length ? [target.notes.join(' ')] : [])]
    .filter(Boolean).map(line => line.endsWith('.') ? line : line + '.');
  el('preview-note').replaceChildren(node('b', 'What would change. '), node('span', note.join(' ')));
  el('preview').hidden = false;
}

/* ------------------------------------------------------------------------- the list to switch to */
function rowOf(config) {
  const isActive = config.profile === status?.profile;
  const isSelected = config.profile === selected;
  const row = node('article', '', `cfg${isActive ? ' active' : ''}${isSelected ? ' selected' : ''}`);
  row.dataset.profile = config.profile;
  const radio = node('input', '', 'select');
  radio.type = 'radio'; radio.name = 'profile'; radio.value = config.profile;
  radio.checked = isSelected;
  radio.setAttribute('aria-label', `Preview ${nameOf(config)} on the GPU map`);
  radio.disabled = !controlsEnabled();
  radio.addEventListener('change', () => select(config.profile));
  add(row, radio);
  const body = node('div');
  const title = add(node('p', '', 'title'), node('span', nameOf(config)));
  if (config.alias) add(title, node('span', 'Alias', 'pill'));
  add(body, title, node('p', `${config.profile} · ${defaultNameOf(config)}`, 'id mono'));
  const second = everyday(config.everyday);
  const rows = [['Coding', residentRow(config)],
    ['Everyday', config.exclusive
      ? (answers(config, 'nighttime') ? `Not loaded · ${nameOf(config)} answers its requests` : 'Not loaded on this configuration')
      : residentRow(second)],
    ['Serves', (config.service_ids || []).join(' · ')]];
  if (config.variant_label) rows.push(['Variant', config.variant_label]);
  // A refusal is on the row a person is deciding not to click. What a switch would do — build a
  // workspace, restart a resident — is a property of the selection, and the preview below the map
  // says it with the cards it applies to, where it can be read against the machines it names.
  if (config.reasons?.length) rows.push(['Refuses', config.reasons[0]]);
  for (const [label, value] of rows) {
    add(body, add(node('p', '', 'row'), node('span', label), node('span', value)));
  }
  if (editing === config.profile) add(body, editor(config));
  add(row, body);
  const end = node('div', '', 'end');
  const marker = node('span', '', `state${isActive || isSelected ? '' : ' muted'}`);
  marker.textContent = running() && config.profile === operation.target_profile ? 'Selected'
    : running() && ['loading', 'verifying', 'restoring'].includes(operation.phase) && config.profile === operation.from_profile
      ? 'Previous' : isActive && !isSelected ? 'Active' : isSelected && !isActive ? 'Preview'
      : isActive ? 'Active' : '';
  const rename = node('button', editing === config.profile ? 'Close' : 'Rename', 'ghost');
  rename.type = 'button';
  rename.addEventListener('click', () => { editing = editing === config.profile ? null : config.profile; aliasError = ''; aliasDraft = config.alias || ''; render(); });
  add(end, marker, rename);
  add(row, end);
  return row;
}
function editor(config) {
  const box = node('div', '', 'edit');
  const field = node('input', '', 'name');
  field.type = 'text'; field.value = aliasDraft;
  field.placeholder = defaultNameOf(config);
  field.setAttribute('aria-label', 'Alias for ' + config.profile);
  field.maxLength = 48;
  field.addEventListener('input', () => { aliasDraft = field.value; });
  const acts = node('div', '', 'acts');
  const save = node('button', aliasSaving ? 'Saving…' : 'Save', 'ghost');
  save.type = 'button'; save.disabled = aliasSaving;
  save.addEventListener('click', () => saveAlias(config.profile, aliasDraft.trim()));
  const clear = node('button', 'Use default', 'ghost');
  clear.type = 'button'; clear.disabled = aliasSaving || !config.alias;
  clear.addEventListener('click', () => saveAlias(config.profile, null));
  const cancel = node('button', 'Cancel', 'ghost');
  cancel.type = 'button';
  cancel.addEventListener('click', () => { editing = null; aliasError = ''; render(); });
  add(acts, save, clear, cancel);
  add(box, field,
    node('p', `Shown wherever this machine names the configuration, including the catalog it publishes to `
      + `the router. Nothing is drained or reloaded to change a name. Saved on this machine only.`),
    acts);
  if (aliasError) add(box, node('p', aliasError, 'error'));
  return box;
}
function switchImpact(target) {
  const current = configFor(status?.profile);
  if (!target || !current) return 'Existing requests finish first. New requests pause during the switch.';
  const second = !target.exclusive && !current.exclusive ? 'The Everyday model stays loaded.'
    : target.exclusive && current.exclusive ? 'The Everyday model stays unloaded.'
    : target.exclusive ? 'The Everyday model will be stopped while this configuration runs.'
      : 'The Everyday model will be started again.';
  const base = `Existing requests finish first. New requests pause during the switch. ${second}`;
  // What the selection will do, from the same read-only verdict the gate reports: a refusal is shown
  // instead of inviting a click that only fails, and a build is announced before it happens.
  if (target.reasons?.length) return `${base} ${target.reasons[0]}`;
  if (target.notes?.length) return `${base} ${target.notes.join(' ')}`;
  return base;
}
function renderRegistry() {
  const configs = configurations();
  const signature = JSON.stringify([configs, registry().always_included, editing, aliasDraft, aliasError,
    running() ? operation.target_profile : null, status?.profile, selected, connected]);
  if (signature !== registrySignature) {
    registrySignature = signature;
    el('registry-error').hidden = !registry().error;
    el('registry-error').textContent = registry().error || '';
    el('config-list').replaceChildren(...configs.map(rowOf));
  }
  if (running()) selected = operation.target_profile;
  if (!selected || !configs.some(c => c.profile === selected)) {
    selected = pending?.profile || status?.profile || configs[0]?.profile;
  }
  const blocked = !controlsEnabled();
  el('profile-options').disabled = blocked;
  const active = selected === status?.profile;
  el('impact').textContent = active || blocked ? '' : switchImpact(configFor(selected));
  el('apply').disabled = blocked || active || !selected;
  el('apply').textContent = submitting ? 'Starting switch…' : running() ? 'Switch in progress'
    : !connected ? 'Status unavailable' : active ? `${nameOf(configFor(selected))} is active`
      : `Switch to ${nameOf(configFor(selected))}`;
  el('switch-reason').textContent = requestUnknown
    ? 'Checking whether the request was accepted. It will not be sent again automatically.'
    : running() ? '' : !connected ? 'Reconnect to view the current runtime before switching.'
      : status?.switching?.reason || '';
  el('operation').hidden = !running() && !el('operation').textContent;
}
function select(profile) { selected = profile; actionError = ''; tellError(''); render(); }
const controlsEnabled = () => connected && !submitting && !running() && !requestUnknown
  && status?.switching?.available !== false;

/* ----------------------------------------------------------------------------------- the header */
function renderHeader() {
  const active = configFor(status?.profile);
  el('machine').textContent = `${status?.host?.id || 'This machine'} / AI Runtime`;
  el('active-name').textContent = connected
    ? nameOf(active) || status?.profile || 'No configuration' : 'Status unavailable';
  const detail = active
    ? `${active.profile}${active.alias ? ` · ${defaultNameOf(active)}` : ''} · `
      + (active.exclusive ? `one model, ${ownsTextGpus()}` : `two residents, ${gpuList(active)} and the second pair`)
    : '';
  el('active-detail').textContent = connected ? detail : '';
  const health = el('health');
  health.textContent = !connected ? 'Disconnected' : running() ? 'Switching'
    : status?.ready ? 'Ready' : operation?.status === 'needs-attention' ? 'Needs attention'
      : status?.maintenance?.draining ? 'Maintenance' : 'Needs attention';
  health.className = `badge ${connected && status?.ready && !running() ? 'ready'
    : operation?.status === 'needs-attention' || status?.service?.serving === false ? 'bad' : 'pending'}`;
  // The first fact is the one read from across a room, so it is never left blank: a machine that is
  // not ready says what it is doing instead of saying nothing.
  fact('fact-health', connected, connected ? (status?.ready ? 'dot'
      : running() || status?.maintenance?.draining ? 'dot warn'
        : operation?.status === 'needs-attention' || status?.service?.serving === false ? 'dot bad' : 'dot warn')
    : 'dot mute',
    !connected ? 'Status unavailable' : status?.ready ? readyLine()
      : running() ? (operation.kind === 'restore' ? 'Restoring the last known good release'
        : 'Switching configurations')
      : status?.maintenance?.draining ? 'Paused for maintenance'
      : operation?.status === 'needs-attention' ? 'Recovery needs attention'
      : status?.service?.serving === false ? 'Nothing is serving' : 'Not ready');
  const counts = status?.maintenance;
  fact('fact-requests', connected && counts, `dot${counts && counts.active_requests ? '' : ' mute'}`,
    counts ? `${counts.active_requests} active · ${counts.queued_requests} queued` : '');
  fact('fact-publication', connected, publicationTone(), publicationLine());
  fact('fact-updated', connected, 'dot mute',
    status?.updated_at ? `Updated ${new Date(status.updated_at).toLocaleTimeString()}` : '');
}
function fact(id, visible, tone, text) {
  const element = el(id);
  element.hidden = !visible || !text;
  if (element.hidden) return;
  element.querySelector('.dot').className = `dot ${tone.replace('dot', '').trim()}`;
  element.querySelector('span').textContent = text;
}
const publicationTone = () => {
  const p = status?.publication;
  if (!p) return 'warn';
  if (p.error || p.retired_at || (p.sinks || []).some(s => s.error || !s.available)) return 'warn';
  return '';
};
// Where this machine's catalog lives, in one line. "Models running here" and "published there" are
// two different failures an operator should not need a shell to tell apart; the router's complaint
// about its own network is a reachability fact about the address, not this machine being unwell.
const publicationLine = () => {
  const p = status?.publication;
  const warn = [...new Set((p?.sinks || []).flatMap(s => s.warnings || []))]
    .map(w => w.startsWith('ROUTER_CANNOT_REACH_BACKEND:') ? 'router cannot reach these backends' : w);
  if (!connected) return 'Publication state unavailable';
  if (!p) return 'Catalog not published yet';
  if (p.error) return `Catalog publication failing: ${p.error}`;
  if (p.retired_at) return `Catalog retired ${new Date(p.retired_at).toLocaleString()}`;
  const sinks = (p.sinks || []).map(s => s.kind === 'file' ? 'the local router'
    : `${String(s.name).replace(/^\w+:\/\//, '')} revision ${String(s.revision || '').slice(0, 8)}`);
  return `Catalog published to ${sinks.join(' and ') || 'nothing yet'}${warn.length ? ` · ${warn.join(' · ')}` : ''}`;
};

/* --------------------------------------------------------------------- progress, restore, details */
// Prepare is the step a self-provisioning engine adds: the workspace build, done beside the serving
// model, before the drain. A switch that needs no build runs it as a no-op.
const phaseNames = {checking: 'Check', preparing: 'Prepare', draining: 'Drain', loading: 'Load', verifying: 'Verify'};
const phases = Object.keys(phaseNames);
function renderProgress() {
  const panel = el('operation');
  panel.hidden = !operation;
  if (!operation) return;
  const target = nameOf(configFor(operation.target_profile)) || operation.target_profile;
  // A restoration is a different promise from a switch, so it is titled as one: nothing was chosen,
  // and the point is that the previous configuration is serving again.
  const titles = operation.kind === 'restore'
    ? {running: `Restoring ${target}`, succeeded: `${target} is serving`, recovered: `${target} is serving`,
      failed: 'Restoration did not begin', 'needs-attention': 'Restoration needs attention'}
    : {running: operation.phase === 'restoring' ? `Restoring ${nameOf(configFor(operation.from_profile))}`
        : `Switching to ${target}`, succeeded: `${target} is ready`, failed: 'Switch did not complete',
      recovered: `Switch failed · ${nameOf(configFor(operation.from_profile))} restored`,
      'needs-attention': 'Recovery needs attention'};
  el('operation-title').textContent = titles[operation.status] || 'Result';
  const elapsed = Math.max(0, Math.floor(((operation.finished_at ? Date.parse(operation.finished_at)
    : Date.now()) - Date.parse(operation.started_at)) / 1000));
  el('elapsed').textContent = `${running() ? 'Elapsed' : 'Duration'} ${Math.floor(elapsed / 60)}m ${elapsed % 60}s`;
  const state = `${operation.status}:${operation.phase}`;
  const steps = el('steps');
  if (steps.dataset.state !== state) {
    steps.dataset.state = state; steps.replaceChildren();
    steps.hidden = !running() || operation.phase === 'restoring';
    const current = phases.indexOf(operation.phase);
    phases.forEach((phase, index) => {
      const done = operation.status === 'succeeded' || (running() && index < current);
      const now = running() && index === current;
      const step = node('li', '', `${done ? 'done' : ''} ${now ? 'now' : ''}${operation.status === 'failed' && index === current ? ' bad' : ''}`);
      const mark = node('span', done ? '✓' : '', 'm');
      mark.setAttribute('aria-hidden', 'true');
      add(step, mark, node('span', phaseNames[phase]));
      if (now) step.setAttribute('aria-current', 'step');
      add(step, node('span', done ? ' completed' : now ? ' in progress' : ' pending', 'sr-only'));
      add(steps, step);
    });
  }
  const message = operation.message || '';
  if (el('operation-message').textContent !== message) el('operation-message').textContent = message;
  el('admissions').textContent = !connected
    ? 'Connection lost. The switch may still be running; reconnecting without sending another request.'
    : admissionsPaused() ? 'New requests are paused for both models.'
    : operation.status === 'needs-attention'
      ? (operation.kind === 'restore'
        ? 'Nothing is serving yet. The runtime schedules the next attempt itself, and the button above runs it again.'
        : 'Check current runtime health and follow the documented recovery procedure.') : '';
}
function secondsUntil(timestamp) {
  const seconds = Math.round((Date.parse(timestamp) - Date.now()) / 1000);
  if (!Number.isFinite(seconds) || seconds <= 0) return '';
  return ` in ${seconds < 120 ? `${seconds} seconds` : `${Math.round(seconds / 60)} minutes`}`;
}
function renderRestoration() {
  const section = el('restoration');
  const service = status?.service || {};
  const restoring = running() && operation?.kind === 'restore';
  // A switch in flight also has nothing serving at this instant, and saying so here would compete
  // with the panel below that says what is being done about it. This card is for a machine that is
  // not serving and has no run in progress — the one where the button is the only lever.
  const visible = connected && (restoring || service.pending || (service.serving === false && !running()));
  section.hidden = !visible;
  if (!visible) return;
  const name = nameOf(configFor(service.profile || status?.profile));
  const record = status?.restoration || {};
  const attempts = record.attempts || 1;
  el('restore-state').textContent = restoring ? `Restoring ${name}. New requests stay paused until it is serving.`
    : service.pending ? `Nothing is serving. The runtime is retrying the restoration${secondsUntil(record.next_attempt_at)}, after ${attempts} attempt${attempts === 1 ? '' : 's'} — this machine does not need a reboot.`
      : `Nothing is serving and no restoration is pending. Run it now; nothing here needs a shell.`;
  const unavailable = status?.restoring && status.restoring.available === false ? status.restoring.reason : '';
  el('restore').disabled = submitting || !connected || restoring || Boolean(unavailable) || service.restorable === false;
  el('restore-reason').textContent = service.restorable === false
    ? 'No last-known-good release is recorded on this host, so there is nothing to restore to.'
    : unavailable || (restoring ? 'The restoration is already running.' : '');
}
function renderDetails() {
  const signature = JSON.stringify([status?.services, status?.deployed_revision, status?.source_deployment,
    status?.aliases, status?.revision]);
  if (signature === detailsSignature) return;
  detailsSignature = signature;
  const services = el('services');
  services.replaceChildren();
  for (const service of status?.services || []) {
    const article = add(node('article', '', 'service'), node('h3', service.name));
    const facts = node('dl');
    for (const [label, value] of [['Model', service.model], ['Context', context(service.context_tokens)],
      ['Container', service.container_name],
      ['Running since', service.started_at ? new Date(service.started_at).toLocaleString() : 'Unavailable'],
      ['Text GPUs', service.gpu_names?.join(' + ') || 'Names unavailable'],
      ['Vision', service.vision_gpu_id ? `${service.vision_gpu_name || 'Vision GPU'} · ${service.vision_device} · ${service.exclusive ? 'exclusive' : 'shared'}` : 'CPU'],
      ['Differences from this configuration', service.differences?.join(', ') || 'None']]) {
      add(facts, node('dt', label), node('dd', value || 'Unavailable'));
    }
    add(article, facts); add(services, article);
  }
  if (!status?.services?.length) add(services, node('p', 'No resident is loaded.', 'sub'));
  el('revision').textContent = status?.deployed_revision || 'Awaiting deployment';
  const note = status?.aliases?.note;
  el('diagnostics').hidden = !note;
  el('diagnostics').textContent = note || '';
  const deployment = status?.source_deployment;
  el('deploy-detail').textContent = deployment
    ? [`Source deployment: ${String(deployment.phase).replaceAll('-', ' ')}`,
      deployment.finished_at ? new Date(deployment.finished_at).toLocaleString() : 'In progress',
      deployment.revision?.slice(0, 12), deployment.error].filter(Boolean).join(' · ')
    : 'No source deployment recorded. Profile switches are reported above, separately from source releases.';
}
function render() {
  // One guard over the whole render: a page that has drawn half its facts and stopped is worse than
  // one that says it cannot read the runtime, because half a picture of a serving machine is what an
  // operator trusts and then acts on.
  try {
    renderHeader(); renderMap(); renderPreview(); renderRegistry();
    renderProgress(); renderRestoration(); renderDetails();
    if (connected) el('updated').textContent = `Status updated ${new Date(status.updated_at || Date.now()).toLocaleTimeString()}`;
  } catch (error) {
    tellError('The page could not be drawn from this status: ' + (error && error.message || error));
  }
}

/* --------------------------------------------------------------------------------- the plumbing */
let status = null, operation = null, selected = null, connected = false, submitting = false;
let mapSignature = '', registrySignature = '', detailsSignature = '', lastStatus = 0, timer;
let pending = null, restorePending = null, requestUnknown = false, polling = false, actionError = '';
let editing = null, aliasDraft = '', aliasError = '', aliasSaving = false;
function schedule(delay) { clearTimeout(timer); timer = setTimeout(tick, delay); }
try { pending = JSON.parse(sessionStorage.getItem('runtime-switch-request')); } catch (_) { /* may be disabled */ }
if (pending && typeof pending.request_id !== 'string') pending = null;
// A restore request is kept apart from a switch request: it names no profile and expects nothing to
// have been observed first, because it is the lever for a machine that is not serving.
try { restorePending = JSON.parse(sessionStorage.getItem('runtime-restore-request')); } catch (_) { /* as above */ }
if (restorePending && typeof restorePending.request_id !== 'string') restorePending = null;
function remember(key, request) {
  if (key === 'runtime-switch-request') pending = request; else restorePending = request;
  try {
    if (request) sessionStorage.setItem(key, JSON.stringify(request));
    else sessionStorage.removeItem(key);
  } catch (_) { /* The server receipt also survives a refresh and appears in status. */ }
}
const running = () => operation?.status === 'running';
const admissionsPaused = () => status?.maintenance?.draining || (running() && operation.phase !== 'checking');
function tellError(message) { el('error').hidden = !message; el('error').textContent = message || ''; }
function setOperation(next) {
  if (!next) return;
  if (pending?.request_id === next.id) { remember('runtime-switch-request', null); requestUnknown = false; }
  if (restorePending?.request_id === next.id) remember('runtime-restore-request', null);
  if (operation?.id === next.id && operation.updated_at > next.updated_at) return;
  if (operation && operation.id !== next.id && operation.started_at > next.started_at) return;
  operation = next;
}
async function getJSON(path, options = {}) {
  const controller = new AbortController();
  const deadline = setTimeout(() => controller.abort(), 12000);
  try {
    const response = await fetch(path, {...options, cache: 'no-store', signal: controller.signal});
    const data = await response.json();
    if (!response.ok) {
      const error = new Error(data.error || 'Runtime status unavailable');
      error.httpStatus = response.status;
      throw error;
    }
    return data;
  } finally { clearTimeout(deadline); }
}
function uuid() {
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 15) | 64; bytes[8] = (bytes[8] & 63) | 128;
  const hex = Array.from(bytes, b => b.toString(16).padStart(2, '0')).join('');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}
async function tick() {
  clearTimeout(timer);
  if (polling || submitting) { schedule(1000); return; }
  polling = true;
  try {
    if (!status || Date.now() - lastStatus >= 5000 || !connected) {
      const next = await getJSON('/api/status');
      if (!status || !next.updated_at || next.updated_at >= status.updated_at) status = next;
      lastStatus = Date.now(); connected = true;
      setOperation(next.operation);
      tellError(actionError || next.startup_error || next.error || next.configurations?.error);
    }
    const id = pending?.request_id || restorePending?.request_id || (running() ? operation.id : null);
    if (id) {
      try { setOperation(await getJSON(`/api/operations/${encodeURIComponent(id)}`)); }
      catch (error) {
        if (error.httpStatus !== 404) throw error;
        // An unrecorded request may be retried explicitly with the SAME ID. Never POST again
        // automatically after a lost response, a refresh, or a reconnect.
        requestUnknown = false;
        tellError(restorePending?.request_id === id
          ? 'No operation was recorded for that request. You can run the restoration again.'
          : 'No operation was recorded for that request. You can try the switch again.');
      }
    }
  } catch (error) {
    connected = false;
    tellError('Runtime status is unavailable. A switch may still be running; reconnecting without sending another request.');
  }
  render();
  // While a restoration is scheduled the page keeps watching at one-second cadence, because the
  // runtime may bring the service back between polls and that is worth seeing immediately.
  polling = false;
  schedule(running() || pending || restorePending || requestUnknown || status?.service?.pending ? 1000 : 5000);
}
// One action, no confirmation: it is the same restoration the runtime's supervisor performs, and the
// page is being read at a moment when the machine is not answering anyone.
el('restore').addEventListener('click', async () => {
  if (el('restore').disabled || submitting) return;
  submitting = true; actionError = ''; tellError(''); render(); clearTimeout(timer);
  const request = restorePending || {request_id: uuid()};
  remember('runtime-restore-request', request);
  try {
    const result = await getJSON('/api/restore-last-known-good', {method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-Runtime-CSRF': status.csrf_token},
      body: JSON.stringify(request)});
    setOperation(result.operation); requestUnknown = false;
  } catch (error) {
    if (error.httpStatus && error.httpStatus < 500) {
      remember('runtime-restore-request', null); requestUnknown = false; actionError = error.message; tellError(actionError);
    } else { connected = false; requestUnknown = true; tellError('Connection lost while starting the restoration. Checking whether it was accepted.'); }
  } finally { submitting = false; lastStatus = 0; render(); schedule(1000); }
});
/* A name is saved the way everything else on this page is sent: CSRF header, same body rules, and
   the same refusal to imagine what happened if the answer never arrives. A rename cannot drain or
   restart anything, so there is no receipt to survive — the next status either shows the name or it
   does not, and an error goes back into the editor that is still open. */
async function saveAlias(profile, alias) {
  if (aliasSaving) return;
  aliasSaving = true; aliasError = ''; render();
  try {
    await getJSON('/api/alias', {method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-Runtime-CSRF': status.csrf_token},
      body: JSON.stringify({profile, alias})});
    editing = null; aliasDraft = '';
    lastStatus = 0;   // Show the new name the moment the runtime has it, not on the next poll.
  } catch (error) {
    aliasError = error.httpStatus && error.httpStatus < 500
      ? error.message : 'Connection lost while saving the name. Nothing was renamed.';
  } finally {
    aliasSaving = false;
    await tick();
  }
}
el('switch-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (el('apply').disabled) return;
  const target = configFor(selected);
  // What happens to the second resident is stated from the target's own service IDs, not from the
  // word "exclusive": stopping a resident and stopping a service are different promises.
  const stopped = answers(target, 'nighttime') ? `${nameOf(target)} answers its requests in the meantime`
    : 'its requests will fail until you choose a configuration that loads it again';
  if (target?.exclusive && !configFor(status?.profile)?.exclusive && pending?.profile !== selected &&
      !confirm(`${nameOf(target)} uses ${ownsTextGpus()}, so the Everyday model will stop. ${stopped}. Continue?`)) return;
  submitting = true; actionError = ''; tellError(''); render(); clearTimeout(timer);
  const request = pending && pending.profile === selected && pending.expected_profile === status.profile &&
    pending.expected_revision === status.deployed_revision
    ? pending : {profile: selected, request_id: uuid(), expected_profile: status.profile,
      expected_revision: status.deployed_revision};
  remember('runtime-switch-request', request);
  try {
    const result = await getJSON('/api/profile-switch', {method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-Runtime-CSRF': status.csrf_token},
      body: JSON.stringify(request)});
    setOperation(result.operation); requestUnknown = false;
  } catch (error) {
    if (error.httpStatus && error.httpStatus < 500) {
      remember('runtime-switch-request', null); requestUnknown = false; actionError = error.message; tellError(actionError);
    } else { connected = false; requestUnknown = true; tellError('Connection lost while starting the switch. Checking whether it was accepted.'); }
  } finally { submitting = false; lastStatus = 0; render(); schedule(1000); }
});
tick();
