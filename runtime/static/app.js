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
/* The models a configuration loads: one, two, or as many as its cards hold. A configuration is a
   decision about which models run on which cards, and this page draws that list and nothing else.
   Nothing below it knows a word like "coding" or "everyday", because a model is not a role — there
   is no second half of a pair, no resident of a different kind, and no machine where a model that
   happens to be second is anything other than a model. */
const modelsOf = config => config?.models || [];
const modelFor = (profile, model) => modelsOf(configFor(profile)).find(m => m.model === model);
const indexFor = (profile, model) => {
  const at = modelsOf(configFor(profile)).findIndex(m => m.model === model);
  return at < 0 ? 0 : at;
};
/* What a model is called: the operator's alias when they saved one, and the default the naming rule
   produces otherwise. The server computes both; the page never assembles a name, because the router
   and the CLI publish the same string and must not be reading a fourth opinion. */
const nameOf = entry => entry?.display_name || entry?.model || entry?.profile || 'Unknown';
const defaultNameOf = entry => entry?.default_name || entry?.display_name || entry?.model
  || entry?.profile || '';
/* A configuration is not a thing a client asks for by name, so a row is headed by its id and the
   names underneath it belong to models. Where a sentence has to name a configuration anyway, it
   names the models it loads. */
const titleOf = config => modelsOf(config).map(nameOf).join(' + ') || nameOf(config);
/* The short handle a GPU tile carries: a service ID, which is what a client sends, what an operator
   types, and short enough to fit on a card. A tile that says `nighttime` is a tile about the model
   that answers `nighttime`. */
const handleOf = model => (model?.service_ids || [])[0] || model?.model || model?.profile || 'model';
const gpuList = entry => entry?.gpu_names?.length ? entry.gpu_names.join(' + ')
  : entry?.exclusive ? ownsTextGpus() : 'GPUs unavailable';
// A machine with one model is not "both models ready", and a page that says it has read the machine
// wrong: how many models a host runs is a fact about the configuration, not a fact about pages.
const readyLine = () => {
  const loaded = (status?.services || []).length;
  return !loaded ? 'Ready' : loaded === 1 ? 'One model ready' : `${loaded} models ready`;
};
// How a configuration that runs alone owns the machine, from the cards this host has: "all the text
// cards" is a different number on a two-card machine than on a four-card one, and a count of two
// reads better as a word than as a digit.
const ownsTextGpus = () => {
  const count = (inventory().text || []).length || status?.host?.text_gpus || 0;
  return !count ? 'every text GPU of this host' : count === 2 ? 'both text GPUs' : `all ${count} text GPUs`;
};
const context = tokens => Number.isFinite(tokens) ? `${tokens / 1024}K context` : 'Context unavailable';

/* ------------------------------------------------------------------ the machine, as a set of cards */
const inventory = () => status?.host?.gpu_inventory || {text: [], vision: null};
const cards = () => [...(inventory().text || []), ...(inventory().vision ? [inventory().vision] : [])];
/* Owners keyed by GPU UUID: which model has the card. Two facts of very different strength feed it —
   the running services say what is placed now by UUID, a registry entry says what a switch would
   place by the GPU group it names, which is the same order the host declares its cards in. A model
   that owns every card takes all of them, which is why ownership is computed per model and not per
   half of a pair. */
const ofModel = (profile, service) => modelFor(profile, service.model)
  || {model: service.model, display_name: service.name, service_ids: [], gpu_names: service.gpu_names};
const placementNow = () => {
  const owners = {};
  for (const service of status?.services || []) {
    const owner = {model: ofModel(status?.profile, service), index: indexFor(status?.profile, service.model), service};
    for (const id of service.gpu_ids || []) owners[id] = owner;
    if (service.vision_gpu_id) {
      owners[service.vision_gpu_id] = {...owner, vision: true, shared: service.vision_gpu_shared === true};
    }
  }
  return owners;
};
function placementOf(config) {
  const owners = {};
  modelsOf(config).forEach((model, index) => {
    for (const card of inventory().text) {
      if (model.exclusive || (model.gpu_group && card.group === model.gpu_group)) {
        owners[card.uuid] = {model, index};
      }
    }
    // The vision card belongs only to a model that would put something on it, and only on a host that
    // has one: what a profile declares is what it is for, what the host has is what it gets.
    if (model.vision_gpu && inventory().vision) owners[inventory().vision.uuid] = {model, index, vision: true};
  });
  return owners;
}
// Which of the running service IDs survive a switch: an ID the target names is answered by the model
// that names it, and an ID nothing names is the hole a switch leaves.
const answeredBy = (target, id) => modelsOf(target).find(m => (m.service_ids || []).includes(id));
const kindOf = owner => !owner ? 'free' : owner.vision ? 'vision' : `m${owner.index % 4}`;
const sameOwner = (a, b) => {
  const key = o => !o ? '' : `${o.vision ? 'vision' : 'model'}:${o.model?.model || ''}`;
  return key(a) === key(b);
};
function cardNode(card, owner, preview = false) {
  const tile = node('div', '', `gpu ${kindOf(owner)}${preview ? ' preview' : ''}`);
  add(tile, node('h3', card.name || 'GPU'), node('div', '', 'chips'));
  const chips = tile.querySelector('.chips');
  if (owner) {
    const tone = owner.vision ? 'vision' : `m${owner.index % 4}`;
    add(chips, node('span', handleOf(owner.model), `mini ${tone}`));
    if (owner.vision && owner.shared === false) add(chips, node('span', 'exclusive', 'mini vision'));
    if (owner.vision && owner.service?.vision_device) add(chips, node('span', owner.service.vision_device, 'mini vision'));
    if (owner.service && !owner.service.healthy) add(chips, node('span', 'unwell', 'mini off'));
    if (owner.service && admissionsPaused()) add(chips, node('span', 'paused', 'mini off'));
  } else add(chips, node('span', 'nothing', 'mini off'));
  // Whether the vision card is shared with a model that also decodes text is a fact about a running
  // service, and a preview has none, so a preview tile says only what a configuration puts there.
  const detail = owner?.vision
    ? 'image encoder' + (owner.shared === undefined ? '' : owner.shared ? ' · shared' : ' · exclusive')
    : owner ? (owner.model.exclusive ? 'every text card' : 'text cards') : 'free';
  add(tile, node('p', preview && !owner ? 'stays free' : detail, 'own'));
  return tile;
}
function healthOf(service) {
  if (!connected) return 'Status unavailable';
  if (!service.running) return 'Not running';
  if (!service.healthy) return 'Unavailable';
  return admissionsPaused() ? 'Loaded · new requests paused' : 'Ready';
}
/* One card per model a configuration loads, because a model is the unit everything else uses: the
   thing a client routes to, the thing a name belongs to, the thing a drain waits for. A machine that
   loads three models shows three of these, and none of them is a footnote to the others. */
function modelNode(model, {service, index = 0, state, detail, tag} = {}) {
  const card = node('article', '', `model m${index % 4}${state && state !== 'Ready' ? (service && service.healthy === false ? ' bad' : ' unwell') : ''}`);
  // The text gets a column of its own: `add` returns the parent it appended to, and a left hand
  // written as `add(card, node('div'))` is the card, which turns every line of a model into a
  // flex sibling and squeezes a model name to the width of the word beside it.
  const left = node('div');
  add(card, left);
  add(left,
    node('p', '', 'ids'),
    node('p', '', 'name'),
    node('p', '', 'alias'));
  card.querySelector('.ids').textContent = (model?.service_ids || []).join(' · ') || 'no service ID';
  card.querySelector('.name').textContent = model ? nameOf(model) : 'Nothing';
  card.querySelector('.alias').textContent = detail || '';
  add(card, node('span', state || 'Unavailable', 'state'));
  if (tag) add(card.querySelector('.name'), node('span', ' ' + tag, 'tag new'));
  return card;
}
const details = model => `${model.model} · ${context(model.context_tokens)} · ${gpuList(model)}`;
const legendItem = (cls, text) => {
  const item = node('span');
  return add(item, node('i', '', cls), node('span', text));
};
function renderLegend(models, vision) {
  const legend = el('legend');
  legend.replaceChildren();
  models.forEach((model, index) => add(legend, legendItem(`m${index % 4}`, `${handleOf(model)} — ${nameOf(model)}`)));
  if (vision) add(legend, legendItem('v', 'Vision'));
  add(legend, legendItem('d', 'Nothing placed'));
}

/* --------------------------------------------------------------------- rendering: the map itself */
function renderMap() {
  const loaded = el('models'), board = el('gpus');
  const services = status?.services || [];
  const signature = JSON.stringify([
    services.map(s => [s.model, s.name, s.context_tokens, s.gpu_ids, s.gpu_names, s.vision_gpu_id,
      s.vision_gpu_name, s.exclusive, s.healthy, s.running]),
    modelsOf(configFor(status?.profile)), status?.host?.gpu_inventory, status?.maintenance?.draining,
    status?.configurations?.active, connected,
    running() ? [operation.target_profile, operation.phase] : null]);
  if (signature === mapSignature) return;
  mapSignature = signature;
  loaded.replaceChildren(); board.replaceChildren();
  if (!connected) {
    add(loaded, node('p', 'Runtime status is unavailable. Nothing below is a claim about this machine.', 'sub'));
    return;
  }
  const config = configFor(status?.profile);
  const owners = placementNow();
  for (const service of services) {
    const model = ofModel(status.profile, service);
    add(loaded, modelNode(model, {
      index: indexFor(status.profile, service.model), service, state: healthOf(service),
      detail: details(model)}));
  }
  // A model of the live configuration with no service behind it is worth a card, because the promise
  // it makes — a service ID with nothing answering it — is the fact an operator needs when a client
  // is being refused. A model this configuration does not name is not drawn at all: it is not this
  // machine's business at all right now, whatever another configuration would do with it.
  for (const [index, model] of modelsOf(config).entries()) {
    if (services.some(s => s.model === model.model)) continue;
    add(loaded, modelNode(model, {index, state: 'Not loaded', detail: details(model)}));
  }
  if (!services.length && !modelsOf(config).length) {
    add(loaded, node('p', 'Nothing is loaded, and this machine has no configuration to read.', 'sub'));
  }
  for (const card of cards()) add(board, cardNode(card, owners[card.uuid]));
  renderLegend(modelsOf(config), Boolean(inventory().vision));
}

/* The models this machine is running that a target configuration does not name. A switch runs the
   models its configuration names and nothing else, so what falls out of that list is what stops. */
const sameModel = (a, b) => !!a && !!b && a.model === b.model;
const stopsOnSwitch = target => (status?.services || [])
  .filter(s => target && !modelsOf(target).some(m => sameModel(m, ofModel(status?.profile, s))));
/* A selected configuration that is not the live one: the same machine, redrawn as a promise, with
   the difference in ownership spelled out because that difference is the whole cost of the switch. */
function renderPreview() {
  const panel = el('preview');
  const target = configFor(selected);
  const show = connected && target && target.profile !== status?.profile && !running();
  panel.hidden = !show;
  if (!show) return;
  el('preview-name').textContent = titleOf(target);
  const before = placementNow(), after = placementOf(target);
  const listing = el('preview-models'), board = el('preview-gpus');
  listing.replaceChildren(); board.replaceChildren();
  modelsOf(target).forEach((model, index) => add(listing,
    modelNode(model, {index, state: 'Would load', detail: details(model)})));
  // A model running now that the target does not name is drawn in the same shape as the models it
  // would load, in the colour it has while it runs: what a switch stops is the part of the cost the
  // card layout cannot show, because the cards a stopping model frees are usually the cards a loading
  // model takes. What happens to its service IDs is said per ID, because an ID that another loaded
  // model also answers does not stop being answered.
  for (const service of stopsOnSwitch(target)) {
    const model = ofModel(status.profile, service);
    const ids = model.service_ids || [];
    const taken_by = ids.filter(id => answeredBy(target, id));
    const unanswered = ids.filter(id => !answeredBy(target, id));
    add(listing, modelNode(model, {index: indexFor(status.profile, service.model), state: 'Stops',
      detail: [unanswered.length ? `nothing would answer ${unanswered.join(', ')}` : '',
        taken_by.length ? `${taken_by.join(', ')} answered by a model this configuration names` : '']
        .filter(Boolean).join(' · ') || 'no service IDs'}));
  }
  // Which model ends up on which card, grouped by the model rather than by the card: an operator
  // deciding whether to switch is asking "what would that machine be doing", and a list of five cards
  // all going to the same model is one fact, not five.
  const taking = new Map();
  const freed = [];
  for (const card of cards()) {
    const was = before[card.uuid], now = after[card.uuid];
    add(board, cardNode(card, now, true));
    if (sameOwner(was, now)) continue;
    if (!now) { freed.push(card.name); continue; }
    const who = nameOf(now.model);
    if (!taking.has(who)) taking.set(who, []);
    taking.get(who).push(card.name);
  }
  const changes = [...taking].map(([who, list]) => `${who} would take ${list.join(', ')}`);
  if (freed.length) changes.push(`nothing would be placed on ${freed.join(', ')}`);
  const stopping = stopsOnSwitch(target).map(s => nameOf(ofModel(status.profile, s)));
  const note = [changes.length ? changes.join('; ') + '.' : 'Every card keeps the model it has.',
    stopping.length ? `${stopping.join(', ')} ${stopping.length === 1 ? 'stops' : 'stop'}: a switch runs the `
      + `models this configuration names and nothing else.` : '',
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
  radio.setAttribute('aria-label', `Preview ${titleOf(config)} on the GPU map`);
  radio.disabled = !controlsEnabled();
  radio.addEventListener('change', () => select(config.profile));
  add(row, radio);
  const body = node('div');
  // The row is headed by the configuration's id, not by a name, because a name belongs to a model and
  // a configuration is the decision about which models share which cards. Its id is what `Switch to
  // …` and a receipt in the log both use, so it is the honest heading for the thing being chosen.
  const title = add(node('p', '', 'title'), node('span', config.profile));
  add(body, title, node('p', modelsOf(config).map(m => m.model).join(' + ')
    || 'This configuration names no model', 'id mono'));
  // One line per model, each with the name it is published by, the cards it would run on, and its own
  // Rename. A configuration that loads two models can be given two names, and neither of them is a
  // spare for the other.
  const rows = modelsOf(config).map(model => [handleOf(model), `${nameOf(model)} · ${gpuList(model)}`, model]);
  if (config.variant_label) rows.push(['Variant', config.variant_label, null]);
  // A refusal is on the row a person is deciding not to click. What a switch would do — build a
  // workspace, reload a model — is a property of the selection, and the preview below the map
  // says it with the cards it applies to, where it can be read against the machines it names.
  if (config.reasons?.length) rows.push(['Refuses', config.reasons[0]]);
  for (const [label, value, model] of rows) {
    const line = add(node('p', '', 'row'), node('span', label), node('span', value));
    if (model) {
      // The name's own two controls are one column at the right of the line, not two loose elements:
      // the line wraps when a model's name and cards are long, and a control centred against a
      // three-line sentence reads as part of it.
      const controls = add(node('span', '', 'controls'));
      if (model.alias) add(controls, node('span', 'Renamed', 'pill'));
      const name = editingKey(config.profile, model.model);
      const rename = node('button', editing === name ? 'Close' : 'Rename', 'ghost tiny');
      rename.type = 'button';
      rename.addEventListener('click', () => {
        editing = editing === name ? null : name; aliasError = ''; aliasDraft = model.alias || ''; render();
      });
      add(controls, rename);
      add(line, controls);
      if (editing === name) add(body, editor(config, model));
    }
    add(body, line);
  }
  add(row, body);
  const end = node('div', '', 'end');
  const marker = node('span', '', `state${isActive || isSelected ? '' : ' muted'}`);
  marker.textContent = running() && config.profile === operation.target_profile ? 'Selected'
    : running() && ['loading', 'verifying', 'restoring'].includes(operation.phase) && config.profile === operation.from_profile
      ? 'Previous' : isActive && !isSelected ? 'Active' : isSelected && !isActive ? 'Preview'
      : isActive ? 'Active' : '';
  add(end, marker);
  add(row, end);
  return row;
}
// Which model's name is being edited: a configuration can have several, and an editor belongs to one.
const editingKey = (profile, model) => `${profile} ${model}`;
function editor(config, model) {
  const box = node('div', '', 'edit');
  const field = node('input', '', 'name');
  field.type = 'text'; field.value = aliasDraft;
  field.placeholder = defaultNameOf(model);
  field.setAttribute('aria-label', 'Name for ' + model.model);
  field.maxLength = 48;
  field.addEventListener('input', () => { aliasDraft = field.value; });
  const acts = node('div', '', 'acts');
  const save = node('button', aliasSaving ? 'Saving…' : 'Save', 'ghost');
  save.type = 'button'; save.disabled = aliasSaving;
  save.addEventListener('click', () => saveAlias(model.model, aliasDraft.trim()));
  const clear = node('button', 'Use default', 'ghost');
  clear.type = 'button'; clear.disabled = aliasSaving || !model.alias;
  clear.addEventListener('click', () => saveAlias(model.model, null));
  const cancel = node('button', 'Cancel', 'ghost');
  cancel.type = 'button';
  cancel.addEventListener('click', () => { editing = null; aliasError = ''; render(); });
  add(acts, save, clear, cancel);
  add(box, field,
    node('p', `Names ${model.model} wherever this machine speaks of it — the page, the CLI listing, and `
      + `the catalog it publishes to the router — in this configuration and in every other one that loads `
      + `it. Nothing is drained or reloaded to change a name. Saved on this machine only.`),
    acts);
  if (aliasError) add(box, node('p', aliasError, 'error'));
  return box;
}
function switchImpact(target) {
  if (!target) return 'Existing requests finish first. New requests pause during the switch.';
  // A switch runs the models the target names and nothing else, so the models that are running now
  // and are not in that list are the ones that stop — said by name, because "some of what is loaded
  // goes away" is not a cost estimate.
  const stopping = stopsOnSwitch(target).map(s => nameOf(ofModel(status?.profile, s)));
  const base = 'Existing requests finish first. New requests pause during the switch.'
    + (stopping.length ? ` ${stopping.join(', ')} ${stopping.length === 1 ? 'stops.' : 'stop.'}` : '');
  // What the selection will do, from the same read-only verdict the gate reports: a refusal is shown
  // instead of inviting a click that only fails, and a build is announced before it happens.
  if (target.reasons?.length) return `${base} ${target.reasons[0]}`;
  if (target.notes?.length) return `${base} ${target.notes.join(' ')}`;
  return base;
}
function renderRegistry() {
  const configs = configurations();
  const signature = JSON.stringify([configs, editing, aliasDraft, aliasError,
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
    : !connected ? 'Status unavailable' : active ? `${titleOf(configFor(selected))} is active`
      : `Switch to ${titleOf(configFor(selected))}`;
  el('switch-reason').textContent = requestUnknown
    ? 'Checking whether the request was accepted. It will not be sent again automatically.'
    : running() ? '' : !connected ? 'Reconnect to view the current runtime before switching.'
      : status?.switching?.reason || '';
  // `#operation` is not this function's to hide: it belongs to the progress of a change, and
  // renderProgress says when there is one. A second writer here used the panel's own leftover text
  // as its evidence, which is how a receipt three days old came back on every poll.
}
function select(profile) { selected = profile; actionError = ''; tellError(''); render(); }
const controlsEnabled = () => connected && !submitting && !running() && !requestUnknown
  && status?.switching?.available !== false;

/* ----------------------------------------------------------------------------------- the header */
function renderHeader() {
  const active = configFor(status?.profile);
  el('machine').textContent = `${status?.host?.id || 'This machine'} / AI Runtime`;
  el('active-name').textContent = connected
    ? titleOf(active) || status?.profile || 'No configuration' : 'Status unavailable';
  // One line that says what the machine is doing with its cards, in the machine's own terms: which
  // models, how many, and how much of the silicon they are using.
  const models = modelsOf(active);
  const whole = models.filter(m => m.exclusive).length;
  const detail = models.length
    ? `${active.profile} · ${models.length} model${models.length === 1 ? '' : 's'}`
      + (whole ? ` · one of them uses ${ownsTextGpus()}`
        : ` · ${models.map(m => gpuList(m)).filter((v, i, all) => all.indexOf(v) === i).join('; ')}`)
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
// A finished run is history, and history does not need room on a page about now. This box is for a
// change in flight and for an outcome that is not resolved yet; a run that ended well -- or ended
// badly and had the previous release restored under it, which is the same rest -- goes quiet five
// minutes after it stopped, which is long enough to have been read. Nothing is lost by that: the
// durable receipt is still one GET away at /api/operations/<id>, and the revision it ran is in the
// runtime details and in the catalog revision the router accepted. A machine whose last receipt was
// three days old used to spend its right-hand column on it forever.
const RECEIPT_GRACE_MS = 5 * 60 * 1000;
function receiptWorthRoom() {
  if (!operation) return false;
  if (['failed', 'needs-attention'].includes(operation.status)) return true;
  if (running()) return true;
  const ended = Date.parse(operation.finished_at || operation.started_at || '');
  return Number.isFinite(ended) && Date.now() - ended < RECEIPT_GRACE_MS;
}
function renderProgress() {
  const panel = el('operation');
  panel.hidden = !receiptWorthRoom();
  if (!operation) return;
  const target = titleOf(configFor(operation.target_profile)) || operation.target_profile;
  // A restoration is a different promise from a switch, so it is titled as one: nothing was chosen,
  // and the point is that the previous configuration is serving again.
  const titles = operation.kind === 'restore'
    ? {running: `Restoring ${target}`, succeeded: `${target} is serving`, recovered: `${target} is serving`,
      failed: 'Restoration did not begin', 'needs-attention': 'Restoration needs attention'}
    : {running: operation.phase === 'restoring' ? `Restoring ${titleOf(configFor(operation.from_profile))}`
        : `Switching to ${target}`, succeeded: `${target} is ready`, failed: 'Switch did not complete',
      recovered: `Switch failed · ${titleOf(configFor(operation.from_profile))} restored`,
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
  const name = titleOf(configFor(service.profile || status?.profile));
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
  if (!status?.services?.length) add(services, node('p', 'No model is loaded.', 'sub'));
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
async function saveAlias(model, alias) {
  if (aliasSaving) return;
  aliasSaving = true; aliasError = ''; render();
  try {
    await getJSON('/api/alias', {method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-Runtime-CSRF': status.csrf_token},
      body: JSON.stringify({model, alias})});
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
  // A switch runs the models its configuration names and nothing else, so a model that is answering
  // requests now and is not one of them stops when the switch does. That is the one part of the cost
  // the cards cannot show — the card it frees may be taken by a model that is loading — so it is said
  // by name and asked about before anything is drained.
  const stopping = stopsOnSwitch(target).map(s => ofModel(status?.profile, s));
  const named = stopping.map(model => nameOf(model));
  // The service IDs worth warning about are the ones no model in the target answers; an ID the new
  // configuration happens to answer as well is not lost, and saying it would be would teach an
  // operator to distrust the warning.
  const unanswered = [...new Set(stopping.flatMap(m => (m.service_ids || []).filter(id => !answeredBy(target, id))))];
  const lost = unanswered.length
    ? ` Nothing would answer ${unanswered.join(', ')} until a configuration that names them runs.`
    : ` Every service ID they answer is answered by a model this switch loads.`;
  if (named.length && pending?.profile !== selected &&
      !confirm(`${titleOf(target)} takes the cards ${named.join(', ')} ${named.length === 1 ? 'is' : 'are'} `
        + `on, so ${named.join(', ')} ${named.length === 1 ? 'stops' : 'stop'}.${lost} Continue?`)) return;
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
