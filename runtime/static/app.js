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
  /* A preview has no power state to show: a rung is what the driver is doing to this card right now,
     and a promise about next week's models does not change it. */
  if (!preview) add(tile, powerRow(card));
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
  radio.dataset.focus = 'config:' + config.profile;
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
  field.dataset.focus = 'alias:' + model.model;
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
  // `aliasDraft` is deliberately not in here. It changes on every keystroke, and the only thing
  // drawn from it is the box's own value — which the browser is already holding, with the caret in it.
  // Including it meant the next poll rebuilt the list out from under whoever was typing: the text
  // survived and the focus did not.
  const signature = JSON.stringify([configs, editing, aliasError,
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

/* ------------------------------------------------------ the power state of one physical card, on the page
   The three knobs a GPU can be asked to sit at — power limit, maximum SM clock, and the voltage/frequency
   offset that is the actual undervolt — are the host script's, and so is the rule that makes them safe:
   nothing on this page says a rung is applied until the driver has said it back. Everything below reads
   `status.power`, which is a *cache* with an age, because reading five cards means starting a container
   and the page polls every five seconds. Settings are volatile: a reboot returns a card to stock by
   itself, which is the recovery method and the reason the panel never claims to remember. */
const powerBlock = () => status?.power || {};
const powerCards = () => powerBlock().cards || {};
const powerCard = uuid => powerCards()[uuid] || null;
const powerWrites = () => powerBlock().writes || {};
/* The reading this card's panel is drawn from: what a write or a Re-read just brought back for this
   card, or the cached reading the status carries. The drawing, the boxes and the request are all
   worked out of this one function, so a field cannot show one moment's number and send another's. */
const powerRead = entry => (powerPatch && powerPatch.gpu === entry?.uuid ? powerPatch.read : null)
  || entry?.read || {};
const watts = value => value === null || value === undefined ? '—' : `${Math.round(value)} W`;
const mhz = value => value === null || value === undefined ? '—' : `${value} MHz`;
/* The three numbers a hand-written rung is made of, in the order the driver takes them and the order
   the drawer draws them. They are also the three names the API takes, so one list draws the fields,
   checks them and builds the request, instead of three mappings that can disagree. */
const handKeys = ['power', 'clock_max', 'offset'];
const handName = {power: 'the power limit in watts', clock_max: 'the maximum clock in MHz',
  offset: 'the V/F offset in MHz'};
const handShort = {power: 'power limit', clock_max: 'maximum clock', offset: 'V/F offset'};
/* What one hand-written box holds: the operator's own text once they have typed in that box — and the
   empty string counts, because clearing a box is how they say `not that` — and otherwise what this
   card reports right now. One function answers both `what do I draw in this box` and `what goes in
   the request`, because those two were worked out apart: the box showed the driver's number, the
   draft under it stayed empty, and an operator who changed the two numbers they cared about was
   refused for a third one they could plainly see filled in.
   `clock_max` deliberately starts empty. This driver reports no clock-lock field, so `read.clock_sm`
   is what the card is doing this instant and not what it is capped at — a vision card idling at its
   floor offered `210 MHz` as its own maximum clock, and an operator who trusted that prefill wrote a
   card that cannot boost. The placeholder and the hint carry the ceiling instead. */
function powerHand(key, read) {
  const typed = powerDraft[key];
  if (typed !== undefined) return typed;
  if (key === 'power') return read.power_limit;
  if (key === 'offset') return read.offset;
  return undefined;
}
// The same box as a finite number, or null for `this has nothing in it yet`.
function handValue(key, read) {
  const raw = powerHand(key, read);
  if (raw === undefined || raw === null || String(raw).trim() === '') return null;
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}
const handMissing = read => handKeys.filter(key => handValue(key, read) === null);
/* Why a hand-written write cannot go yet, worked out in one place: `powerSubmit` refuses in these
   words and the drawer's own line says them, so the page never tells the operator to fill in a box
   and then refuses over a different one. */
function powerHandRefusal(read) {
  const missing = handMissing(read);
  return missing.length ? `Fill in ${handName[missing[0]]}.` : '';
}
const handSpelled = value => `${value > 0 ? '+' : ''}${mhz(value)}`;
/* The drawer's line: the numbers that go to the driver if the button is pressed now, with anything
   missing named instead of drawn as a dash. It used to print the prefilled display values as though
   they were the values that would be written, which was the same over-claim as the boxes above it. */
function powerSummary(read) {
  const missing = handMissing(read);
  if (missing.length === handKeys.length) {
    return 'Values by hand — a trial rung this machine has not measured';
  }
  const said = handKeys.filter(key => !missing.includes(key)).map(key => key === 'power'
    ? watts(handValue(key, read))
    : key === 'offset' ? handSpelled(handValue(key, read)) : `${handValue(key, read)} MHz`);
  const gone = missing.length ? ` · no ${missing.map(key => handShort[key]).join(' and no ')} yet` : '';
  return `Values by hand — ${said.join(' · ')}${gone}`;
}
/* The one word on a tile. The numbers that prove it live in the panel, because a 150px tile holds a
   rung name or a watt figure, not a sentence an operator can trust. */
function powerWords(entry) {
  if (!entry) return {tone: '', dot: 'mute', text: 'no reading yet'};
  if (entry.error) return {tone: 'bad', dot: 'bad', text: 'power unreadable'};
  if (entry.state === 'unread') return {tone: '', dot: 'mute', text: 'no reading yet'};
  if (entry.state === 'rung') return {tone: 'rung', dot: '', text: entry.rung};
  if (entry.state === 'stock') return {tone: '', dot: 'mute', text: 'stock'};
  if (entry.state === 'custom') return {tone: 'custom', dot: 'warn', text: 'custom'};
  // The power limit read and the offset did not: never call that stock, which is a claim about both.
  return {tone: 'bad', dot: 'bad', text: 'incomplete'};
}
function powerRow(card) {
  const uuid = card.uuid, entry = powerCard(uuid), line = powerWords(entry);
  const row = node('div', '', 'pwrow'), text = node('p', '', `pw ${line.tone}`);
  add(text, node('i', '', `dot ${line.dot}`), node('b', line.text));
  // One word, and only ever one: a tile is 150px wide, and a line that wraps to four pushes the four
  // tiles beside it out of a row. That a card has no measured rungs is not a different state -- it is
  // the reason the panel offers fewer things to do -- and the panel says so.
  const button = node('button', powerOpen === uuid ? 'Close' : 'Power', 'ghost tiny');
  button.type = 'button';
  button.setAttribute('aria-label', `${powerOpen === uuid ? 'Close the power panel for' : 'Power state of'} ${card.name || 'this GPU'}`);
  button.addEventListener('click', () => {
    powerOpen = powerOpen === uuid ? null : uuid;
    powerError = ''; powerReply = ''; powerChoice = {}; powerDraft = {}; render();
  });
  add(row, text, button);
  return row;
}
/* Which of this card's rungs, if any, the panel should have picked for the operator: where the card
   actually is. A card at an unknown or hand-set state starts with nothing chosen, because the honest
   answer to "which rung?" is then "not one of these". */
const powerTarget = entry => powerChoice.mode ? powerChoice
  : {mode: entry?.state === 'rung' ? entry.rung : entry?.state === 'stock' ? 'revert' : null};
/* Why the write button cannot be used right now, in one answer. Its label, its `disabled` and what
   its click does all speak from the same choice, and the defect this closes was the button being
   marked disabled once — when the panel was drawn — and staying that way while the choice under it
   changed. An empty string means it can be pressed. */
function powerBlocked(entry, target) {
  if (!target.mode) return 'nothing is chosen yet';
  if (powerBusy) return 'a write is already running';
  if (powerWrites().available === false) return powerWrites().reason || 'this runtime is not writing';
  const unwell = powerUnwell(entry.uuid);
  if (unwell.length && !powerGate[entry.uuid]) {
    return `${unwell.map(service => service.name).join(', ')} is not healthy on this card`;
  }
  // A hand-written rung with an empty box in it is deliberately *not* blocked here. A button that
  // cannot be pressed says nothing; pressed, it names the box that is empty.
  return '';
}
const onCard = (service, uuid) => [service.vision_gpu_id, ...(service.gpu_ids || [])].includes(uuid);
const powerIsServing = uuid => (status?.services || []).some(service => service.healthy && onCard(service, uuid));
const powerUnwell = uuid => (status?.services || []).filter(service => !service.healthy && onCard(service, uuid));
/* The request a chosen action turns into, worked out in one place: the drawer's boxes, the drawer's
   summary line and this all read the same three numbers, which is the only way the page can promise
   that what an operator is looking at is what the driver is asked for. */
function powerBody(entry, target, read) {
  const body = {gpu: entry.uuid,
    action: target.mode === 'revert' ? 'revert' : target.mode === 'custom' ? 'custom' : 'apply'};
  if (target.mode && target.mode !== 'revert' && target.mode !== 'custom') body.rung = target.mode;
  if (body.action === 'custom') for (const key of handKeys) body[key] = handValue(key, read);
  return body;
}
function powerSubmit(uuid, entry) {
  const target = powerTarget(entry), read = powerRead(entry);
  if (!target.mode || powerBusy) return;
  if (target.mode === 'custom') {
    // Refused here in the same words the drawer's own line uses, and about a box that is visibly
    // empty — never about one the operator can see a number in, which is what the old split between
    // what a box displayed and what the draft held used to do.
    const refusal = powerHandRefusal(read);
    if (refusal) { powerError = refusal; render(); return; }
  }
  const body = powerBody(entry, target, read);
  body.override = !!powerGate[uuid];
  // Writing under a serving model is the one act here worth pausing on: a rung two notches too
  // aggressive does not fail, it produces strange turns hours later.
  if (body.action !== 'read' && powerIsServing(uuid) && !confirm(
      `${entry.name} is decoding for ${(status.services || []).filter(s => s.healthy
        && [s.vision_gpu_id, ...(s.gpu_ids || [])].includes(uuid)).map(s => s.name).join(', ')} now. `
      + `A rung that is too aggressive does not fail: it shows up later as strange turns or an Xid. `
      + `The way back is Revert to stock, or a reboot, which returns this card to stock by itself.\n\n`
      + `Write ${body.rung || body.action} to ${entry.name}?`)) return;
  powerBusy = true; powerError = ''; render();
  getJSON('/api/gpu-power', {method: 'POST',
    headers: {'Content-Type': 'application/json', 'X-Runtime-CSRF': status.csrf_token},
    body: JSON.stringify(body)})
    .then(reply => {
      powerPatch = {gpu: uuid, read: reply.read, at: reply.read_at};
      powerReply = reply.ok ? `Applied ${reply.to}; the driver agrees.` : '';
      if (!reply.ok) powerError = `Applied ${reply.to}, and the read-back failed: `
        + Object.entries(reply.readback || {}).filter(([, verdict]) => verdict === 'fail')
          .map(([knob]) => knob).join(', ') + ' did not read back. This card is now between states — '
        + 'Revert to stock is one click and needs no model to be healthy.';
      if (reply.rounded_offset) powerError = powerError || 'The offset is taken in 15 MHz steps; the '
        + 'driver was asked for the nearest one.';
    })
    .catch(error => { powerError = error.message || 'The card could not be written.'; })
    .finally(() => { powerBusy = false; lastStatus = 0; render(); schedule(1000); });
}
function powerFacts(entry, read) {
  const facts = node('dl', '', 'facts');
  const line = (label, value, note) => {
    const dt = node('dt', label), dd = node('dd');
    add(dd, node('span', value)); if (note) add(dd, node('em', ' ' + note));
    add(facts, dt, dd);
  };
  line('Power limit', watts(read.power_limit), `of this card's ${watts(read.power_default)} default, `
    + `range ${watts(read.power_min)}–${watts(read.power_max)}`);
  line('V/F offset', read.offset === null || read.offset === undefined ? 'not readable'
    : `${read.offset > 0 ? '+' : ''}${mhz(read.offset)}`, read.offset_error
    ? `NVML: ${read.offset_error}` : 'read back after every write; 15 MHz steps');
  line('Observed SM clock', mhz(read.clock_sm), `this driver reports no clock-lock field and no voltage, `
    + `so a cap is inferred from this number and nothing more. Ceiling the card offers: ${mhz(read.clock_max_sm)}`);
  line('Draw', watts(read.draw), `average ${watts(read.draw_average)} · ${read.temperature === undefined
    ? '—' : read.temperature} °C · p-state ${read.pstate || '—'}`
    + (read.power_capped_hours ? ` · power-capped ${read.power_capped_hours} h since boot` : ''));
  return facts;
}
/* "read from the driver N seconds ago" is the one fact in this panel that is always going stale, and
   it is the only one that is drawn from the clock rather than from the status. So it is refreshed on
   every poll, by itself, instead of rebuilding the panel underneath it — which is what used to close
   the operator's drawer and steal their caret. */
function powerAge(where, entry) {
  if (!where) {
    entry = powerOpen ? powerCard(powerOpen) : null;
    where = entry ? el('power').querySelector('.when .age') : null;
    if (!entry || !where) return;
  }
  where.textContent = powerPatch?.gpu === entry.uuid ? 'read from the driver just now'
    : entry.read_at
      ? `read from the driver ${Math.max(0, Math.round((Date.now() - Date.parse(entry.read_at)) / 1000))} seconds ago`
      : 'never read from the driver';
}

function renderPower() {
  const panel = el('power');
  const entry = powerOpen ? powerCard(powerOpen) : null;
  if (!entry) { panel.replaceChildren(); panel.hidden = true; powerSignature = ''; return; }
  /* Drawn from a signature, the way the map and the list are. This panel holds readings that arrive on
     their own slower cadence, and rebuilding it on every five-second poll closed the drawer the
     operator had just opened and took the caret out of a watts field mid-number. The clearing stays
     directly above the drawing, so a redraw replaces rather than appends and the panel cannot grow. */
  const signature = JSON.stringify([powerOpen, entry, powerWrites(), powerUnwell(powerOpen), powerPatch,
    powerChoice, powerDraft, powerError, powerReply, powerBusy, powerGate[powerOpen], powerCustom]);
  if (signature === powerSignature) { powerAge(); return; }
  powerSignature = signature;
  panel.replaceChildren();
  panel.hidden = false;
  const read = powerRead(entry);
  /* The bounds this card's driver reported, on the card. `status.power` has no `envelope` — there is
     one per card, because a 3090 and a 4080 do not share a floor — and the four places that read it
     off the block got `undefined`: the clock hint drew "locked from — up", the offset limit worked
     only because 150 is also the fallback, and the warning for a max clock at or below the floor,
     the one setting that crashes a card at rest, never fired at all. */
  const envelope = entry.envelope || {};
  const available = powerWrites(), target = powerTarget(entry);
  const refused = available.available === false;
  const head = node('div', '', 'head');
  const when = node('span', '', 'when'), stamp = node('i', '', 'dot mute');
  const age = node('span', '', 'age');
  const reread = node('button', 'Re-read', 'ghost tiny');
  reread.type = 'button';
  reread.addEventListener('click', () => {
    powerBusy = true; powerError = ''; render();
    getJSON('/api/gpu-power', {method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-Runtime-CSRF': status.csrf_token},
      body: JSON.stringify({gpu: entry.uuid, action: 'read'})})
      .then(reply => { powerPatch = {gpu: entry.uuid, read: reply.read, at: reply.read_at};
        if (!reply.ok) powerError = reply.error || 'This card could not be read.'; })
      .catch(error => { powerError = error.message || 'This card could not be read.'; })
      .finally(() => { powerBusy = false; render(); });
  });
  powerAge(age, entry);
  add(when, stamp, age, reread);
  add(head, node('h2', `Power state · ${entry.name || 'GPU'}`), when);
  add(panel, head);

  if (refused) add(panel, node('p', available.reason, 'gate'));
  if (entry.error) add(panel, node('p', entry.error, 'gate'));
  if (!Object.keys(read).length && !entry.error) add(panel,
    node('p', 'This card has not been read from the driver yet. The first reading arrives within '
      + `${powerBlock().cadence_seconds || 30} seconds, or now if you ask for it.`, 'note'));
  else add(panel, powerFacts(entry, read));

  const words = powerWords(entry);
  add(panel, node('p', '', 'note')).lastChild.textContent = entry.state === 'rung'
    ? `At ${entry.rung}${entry.rung === entry.default ? `, this card's default rung` : ''}. `
      + `Power limit and V/F offset both read back as that rung asks. A clock lock cannot be verified `
      + `on GeForce: ${read.clock_sm} MHz observed against ${rungValue(entry, entry.rung, 'clock_max')} MHz `
      + `asked is the whole evidence, which is why a rung is named from the two values that can be read.`
    : entry.state === 'stock' ? `Stock: the power limit is at this card's own default and the offset is 0. `
      + `Nothing is applied.`
    : entry.state === 'custom' ? `Between rungs: what the driver reports matches none of this card's rungs. `
      + `That is what a hand-set value or a half-landed write looks like from outside.`
    : `The driver gave back ${words.text.toLowerCase()}. Reading it again is the first move; the way back `
      + `from anything here is Revert to stock, and after that a reboot, which returns this card to stock `
      + `on its own.`;

  /* Which of the reasons a card has rungs, or has none, is the first thing an operator wants here:
     "measured for the model the driver reported", "overridden by this machine's settings" and "this
     model has never been through the ladder" are three different sentences with three different next
     moves, and a page that shows only the radio list leaves the last two indistinguishable. */
  if (entry.plan_source === 'model') add(panel, node('p', `Plan ${entry.plan}: nobody wrote this `
    + `card's UUID into this machine's settings, and it never had to be. The driver named the card and `
    + `this revision has that model's ladder. power_plans in the private settings is only for pointing `
    + `one card at other numbers, or at none.`, 'sub'));
  else if (entry.plan_source === 'off') add(panel, node('p', `This machine's settings offer this card `
    + `no plan, whatever its model has been measured at — the entry for its UUID in power_plans is what `
    + `says so. Revert and values by hand do not need a plan and still work.`, 'sub'));

  if (entry.note) add(panel, node('p', entry.note, 'sub'));

  const rungs = node('div', '', 'rungs');
  const rungRow = (name, values, why, tags, dashed) => {
    const row = node('label', '', `rung${target.mode === name ? ' now' : ''}${dashed ? ' stock' : ''}`);
    const radio = node('input'); radio.type = 'radio'; radio.name = 'power-rung';
    radio.dataset.focus = 'rung:' + name;
    radio.checked = target.mode === name;
    radio.disabled = refused || powerBusy;
    radio.addEventListener('change', () => { powerChoice = {mode: name}; powerError = ''; render(); });
    const box = node('div');
    const title = node('p', '', 'name');
    add(title, node('span', name === 'revert' ? 'Stock' : name));
    const marks = node('span', '', 'tags');
    for (const tag of tags) add(marks, node('span', tag, `tag ${tag === 'card default' ? 'def' : 'cur'}`));
    if (tags.length) add(title, marks);
    add(box, title, node('p', values, 'set'));
    if (why) add(box, node('p', why, 'why'));
    add(row, radio, box);
    return row;
  };
  for (const rung of entry.rungs || []) {
    add(rungs, rungRow(rung.name, `${watts(rung.power)} · max ${mhz(rung.clock_max)} · offset `
      + `${rung.offset > 0 ? '+' : ''}${mhz(rung.offset)}`,
      rung.name === entry.default && entry.measured ? entry.measured
        : rung.default ? 'One rung toward stock: what you apply when a serving engine has been unwell '
          + 'on anything lower.' : '',
      [rung.default ? 'card default' : '', target.mode === rung.name && entry.state === 'rung'
        && entry.rung === rung.name ? 'where it is now' : ''].filter(Boolean)));
  }
  add(rungs, rungRow('revert', `-rgc · offset 0 · power limit ${watts(read.power_default)}`,
    `Back to the driver's own numbers, live, with no restart of anything that is serving. This is the `
    + `action to take first when a card is misbehaving.`, ['revert'], true));
  if (!entry.rungs?.length) add(rungs, node('p', entry.plan_error
    // Operational state naming a plan source does not have: say which, in the words the runtime used.
    ? entry.plan_error : entry.plan
    ? `This plan has no rungs, so there is nothing to choose: it has been measured and found to have no `
      + `rung worth recommending. Revert and values by hand still work.`
    : entry.plan_source === 'off'
      // No plan because this machine says so, not because the model is unmeasured: different sentence.
      ? `Nothing is offered here by decision. This machine's settings give this card no plan, whatever `
        + `its model has been measured at, which leaves Revert and values by hand.`
      : `No measured rungs, so nothing to choose. No plan in source is for this model of card, and `
        + `${watts(read.power_limit)} / ${mhz(read.clock_max_sm)} / +75 is another card's measured result `
        + `that means nothing here. Measure it, or move it with numbers of your own.`, 'note'));
  add(panel, rungs);

  const custom = node('details', '', 'custom');
  // Open or closed is the operator's decision, not something the runtime gets to reset because a
  // reading arrived, so it is kept in module state and put back on every draw.
  custom.open = powerCustom;
  custom.addEventListener('toggle', () => {
    // A redraw hands a fresh `<details>` the open state it was drawn with, and the browser answers
    // that with a `toggle` of its own. Only a change that came from the operator is a decision.
    if (custom.open === powerCustom) return;
    powerCustom = custom.open;
    // Opening the drawer is somebody saying `by hand` — and on a card with no measured rungs it is the
    // only way in, so taking it at its word is what enables the write. A rung that is already chosen
    // stays chosen until they type, because opening a drawer to look at it is not an edit.
    if (custom.open && !powerTarget(entry).mode) powerChoice = {mode: 'custom'};
    render();
  });
  const summary = node('summary', powerSummary(read));
  summary.dataset.focus = 'power:custom';
  const fields = node('div', '', 'fields');
  const field = (key, label, hint, bounds) => {
    const box = node('label', '', target.mode === 'custom' && powerBad(key, read, envelope) ? 'bad' : '');
    const input = node('input');
    input.type = 'text'; input.value = powerHand(key, read) ?? '';
    input.placeholder = hint.placeholder;
    input.setAttribute('aria-label', label);
    // A stable name for the copy of this field that a redraw creates, so a poll landing while the
    // operator is typing can hand the caret and the selection back instead of dropping them.
    input.dataset.focus = 'power:' + key;
    input.addEventListener('input', () => {
      powerDraft = {...powerDraft, [key]: input.value};
      if (target.mode !== 'custom') powerChoice = {mode: 'custom'};
      /* Every other control on this panel ends by drawing itself, and this one has to as loudly: the
         button's label and its `disabled` are worked out from the choice, so a keystroke that changes
         the choice leaves a button that says one thing and refuses to be pressed. That was the defect
         — the drawer was the only control on the page that could not enable its own write, and the
         click landed on a button still marked disabled from the draw. The caret survives because the
         box names itself and `render()` hands focus and selection back. */
      render();
    });
    const bad = powerBad(key, read, envelope);
    add(box, node('span', label), input, node('span', bad || hint.hint, bad ? 'bad' : 'hint'));
    return box;
  };
  add(fields,
    field('power', 'Power limit (W)', {placeholder: `${watts(read.power_min)}–${watts(read.power_max)}`,
      hint: `this card reports ${watts(read.power_min)}–${watts(read.power_max)}`}),
    field('clock_max', 'Max SM clock (MHz)', {placeholder: read.clock_max_sm ? `up to ${read.clock_max_sm}`
        : '15 MHz bins',
      hint: `this one has no honest default; locked above ${mhz(envelope.clock_min)}, `
        + `this card's ceiling is ${mhz(read.clock_max_sm)}`}),
    field('offset', 'V/F offset (MHz)', {placeholder: '−150 … +150',
      hint: `15 MHz steps; ±${envelope.offset_max || 150} is what this driver has been held at`}));
  add(custom, summary, fields, node('p', `Hand-set values are the script's \`set\`: a trial rung, never offered `
    + `as though it had been measured. What the three boxes hold is what the driver is asked for, and a `
    + `box nobody typed in starts at what this card reports right now. The maximum clock starts empty `
    + `because this driver reports no clock lock: the clock you can read below is what the card is doing, `
    + `not what it is capped at. Applying them changes only this card, right now.`, 'sub'));
  add(panel, custom);

  add(panel, node('p', '', 'volat')).lastChild.append(node('b', `Nothing is installed in the boot path, by design. `),
    `These are volatile driver settings: a reboot returns this card to stock on its own, and that is the `
    + `way back. This page reads and writes them; it never re-applies one by itself. A rung belongs to the `
    + `card, so switching configuration leaves it exactly where it is.`);

  // The health gate, mirrored here from the same rule the runtime refuses by: a model on this card
  // that is not healthy means the write waits for an explicit "apply anyway". The page is not the
  // gate — it is the place that says so before you click rather than after.
  const unwell = powerUnwell(entry.uuid);
  const gate = powerGate[entry.uuid];
  if (unwell.length) {
    const notice = node('div', '', 'gate');
    const many = unwell.length > 1;
    const stopped = unwell.some(service => !service.running);
    // The sentence agrees with what it counts, because the count is the fact: one engine down is
    // "it is not running", two is "at least one of them", and an operator reading either is being
    // told how much of the machine is already hurting.
    add(notice, node('p', `${unwell.map(service => service.name).join(', ')} `
      + `${many ? 'are' : 'is'} not healthy on this card`
      + `${stopped ? (many ? ', and at least one of them is not running' : ', and it is not running')
        : ''}. Nothing has been changed and nothing will be. Say you know better, or fix the model `
      + `first.`));
    const label = node('label'), check = node('input');
    check.type = 'checkbox'; check.checked = !!gate;
    check.dataset.focus = 'gate:' + entry.uuid;
    check.addEventListener('change', () => { powerGate[entry.uuid] = check.checked; render(); });
    add(label, check, node('span', 'Apply anyway — this engine is not what I am worried about'));
    add(notice, label);
    add(panel, notice);
  }

  const acts = node('div', '', 'acts');
  const apply = node('button', powerBusy ? 'Writing…' : powerLabel(entry, target), '');
  apply.type = 'button';
  // One answer to `why is this not pressable`, recomputed on every draw — which is what a keystroke
  // now forces, so the button can no longer be sitting there marked disabled from a draw the choice
  // has moved past since.
  apply.disabled = !!powerBlocked(entry, target);
  if (apply.disabled) apply.title = powerBlocked(entry, target);
  apply.addEventListener('click', () => powerSubmit(entry.uuid, entry));
  const close = node('button', 'Close', 'ghost');
  close.type = 'button';
  close.addEventListener('click', () => { powerOpen = null; render(); });
  add(acts, apply, close);
  add(panel, acts);
  add(panel, node('p', powerError || powerReply || `Nothing is drained and nothing is paused: the write is three `
    + `driver calls and one read-back, a couple of seconds. It is refused while a switch, a restoration or a `
    + `source deployment has the machine.`, powerError ? 'gate' : 'sub'));

  const wrote = entry.last_write;
  if (wrote) add(panel, node('p', `Written from this page ${wrote.at}: ${wrote.from} → ${wrote.to}, `
    + `read-back ${wrote.ok ? 'OK' : 'FAILED'}. The last ${20} writes are kept on this machine, and none of `
    + `them survived a reboot.`, 'hist'));
}
const rungValue = (entry, name, key) => (entry.rungs || []).find(rung => rung.name === name)?.[key] ?? '—';
function powerLabel(entry, target) {
  // Nothing chosen yet — which is what a card at `custom`, at `incomplete` or with no rungs looks
  // like before the operator has said anything. It asked for `null` to be applied to the card.
  if (!target.mode) return 'Choose a rung, or write values by hand';
  if (target.mode === 'revert') return `Revert ${entry.name} to stock`;
  if (target.mode === 'custom') return `Write these values to ${entry.name}`;
  return (entry.state === 'rung' && entry.rung === target.mode ? 'Re-apply ' : 'Apply ')
    + `${target.mode} to ${entry.name}`;
}
/* Client-side because it is a hint and not a gate: the driver and runtime refuse whatever they are
   going to refuse, and a field that says so before the write costs nothing. */
function powerBad(key, read, envelope) {
  const raw = powerHand(key, read);
  if (raw === undefined || raw === '' || raw === null) return '';
  const value = Number(raw);
  if (!Number.isFinite(value)) return 'a number is expected';
  if (key === 'power' && read.power_max != null && value > read.power_max) return `this card's ceiling is ${watts(read.power_max)}`;
  if (key === 'power' && read.power_min != null && value < read.power_min) return `this card's floor is ${watts(read.power_min)}`;
  if (key === 'offset' && Math.abs(value) > (envelope?.offset_max || 150)) return `±${envelope?.offset_max || 150} MHz is what this driver has been held at`;
  if (key === 'offset' && Math.round(value / 15) * 15 !== value) return `the driver takes 15 MHz steps; this would be applied as ${Math.round(value / 15) * 15 > 0 ? '+' : ''}${Math.round(value / 15) * 15} MHz`;
  if (key === 'clock_max' && envelope?.clock_min && value <= envelope.clock_min) return `a max clock at or below the floor leaves the card no P-state to idle in`;
  return '';
}


/* A redraw is the runtime telling the page something new, and it must not cost the operator what they
   were in the middle of doing. Controls that can legitimately be holding a caret name themselves with
   `data-focus`; after a redraw the page looks for the copy that was just made, gives it the focus back
   and puts the caret where it was. Without this, a poll — or a driver reading landing on its own
   cadence — dropped the caret out of a half-typed model name or a half-typed wattage, at a moment the
   operator could not predict, which reads as the page taking the keyboard away. */
function keepFocus() {
  const active = document.activeElement;
  try {
    const name = active?.dataset?.focus;
    return name ? {name, start: active.selectionStart, end: active.selectionEnd} : null;
  } catch (_) { return null; }
}
function restoreFocus(at) {
  if (!at) return;
  const again = document.querySelector(`[data-focus="${at.name}"]`);
  if (!again) return;
  again.focus();
  if (typeof again.setSelectionRange !== 'function') return;
  try {
    if (typeof at.start === 'number' && again.selectionStart !== at.start) again.setSelectionRange(at.start, at.end);
  } catch (_) { /* this kind of control has no caret to put back */ }
}

function render() {
  const focus = keepFocus();
  renderPower();
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
  restoreFocus(focus);
}

/* --------------------------------------------------------------------------------- the plumbing */
let status = null, operation = null, selected = null, connected = false, submitting = false;
let mapSignature = '', registrySignature = '', detailsSignature = '', lastStatus = 0, timer;
let pending = null, restorePending = null, requestUnknown = false, polling = false, actionError = '';
let editing = null, aliasDraft = '', aliasError = '', aliasSaving = false;
/* The power panel is open on one card at a time, and what is chosen inside it is a draft rather than a
   fact: the fact is what the driver says, and that arrives with the next status. `powerGate` remembers
   an "apply anyway" tick so a re-render does not silently un-tick the thing that let the write through. */
let powerOpen = null, powerChoice = {}, powerDraft = {}, powerError = '', powerReply = '', powerBusy = false;
let powerPatch = null, powerGate = {};
/* The panel is drawn from a signature like every other region on this page, and "is the Values by
   hand drawer open" is part of what it draws. It lives here rather than only on the `<details>`
   element because the element is thrown away and rebuilt, and an operator who opened a drawer should
   not have it close under their hands because the runtime answered a poll. */
let powerCustom = false, powerSignature = '';
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
