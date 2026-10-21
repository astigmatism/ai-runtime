'use strict';
const el = id => document.getElementById(id);
const node = (tag, text = '', cls = '') => {
  const element = document.createElement(tag);
  element.textContent = text;
  if (cls) element.className = cls;
  return element;
};
// Names come from the configuration registry in /api/status, so the page never needs a code change
// when a configuration is added or renamed.
const configurations = () => status?.configurations?.selectable || [];
// The second resident a paired configuration loads. `everyday` is its role; `nighttime` is the
// service ID a client sends to reach it, and the two are not the same fact.
const everyday = profile => {
  const variants = status?.configurations?.always_included || [];
  return variants.find(n => n.profile === profile) || variants[0];
};
// Whether a configuration answers a service ID at all, whichever resident it points it at.
const answers = (config, id) => !!config && (config.service_ids || []).includes(id);
const configFor = profile => configurations().find(c => c.profile === profile);
const profileName = profile => configFor(profile)?.display_name || profile || 'Unknown configuration';
const gpuList = names => names?.length ? names.join(' + ') : 'GPUs unavailable';
// How a configuration that runs alone owns the machine. The count comes from this host's status,
// because "all the text cards" is a different number on a two-card host than on a four-card one, and
// a page that hard-codes the number the source was written against is a page that lies about a
// second machine.
const ownsTextGpus = () => {
  const count = status?.host?.text_gpus;
  return count ? `all ${count} text GPUs` : 'every text GPU of this host';
};
// Only added where a display name does not already say how many requests run at once.
const slotsText = (slots, name = '') => slots > 1 && !/request/i.test(name) ? ` · ${slots} requests at once` : '';
function configTitle(config) {
  // Two configurations of one backend have the same display name because it is the same model, so the
  // variant label is the only thing that tells an operator which one the radio is selecting.
  const own = config.variant_label ? ` (${config.variant_label})` : '';
  if (config.exclusive) return `${config.display_name}${own} · one model, ${ownsTextGpus()}`;
  const variant = everyday(config.everyday);
  const label = variant?.variant_label;
  return `${config.display_name} + Everyday${label ? ` (${label})` : ''}`;
}
function configRows(config) {
  const second = everyday(config.everyday);
  return [
    ['Coding', `${config.display_name}${slotsText(config.parallel_slots, config.display_name)} · ${gpuList(config.gpu_names)}`],
    ['Everyday', config.exclusive
      ? answers(config, 'nighttime') ? `Not loaded · ${config.display_name} answers its requests`
        : 'Not loaded while this configuration uses its GPUs'
      : second ? `${second.display_name} · ${gpuList(second.gpu_names)}` : 'Unavailable']];
}
function switchImpact(target) {
  const current = configFor(status?.profile);
  if (!target || !current) return 'Existing requests finish first. New requests pause during the switch.';
  const second = !target.exclusive && !current.exclusive ? 'The Everyday model stays loaded.'
    : target.exclusive && current.exclusive ? 'The Everyday model stays unloaded.'
    : target.exclusive ? 'The Everyday model will be stopped while this configuration runs.'
      : 'The Everyday model will be started again.';
  const base = `Existing requests finish first. New requests pause during the switch. ${second}`;
  // What the selection will do, from the same read-only verdict the gate reports: a refusal is
  // shown instead of inviting a click that only fails, and a build is announced before it happens.
  if (target.reasons?.length) return `${base} ${target.reasons[0]}`;
  if (target.notes?.length) return `${base} ${target.notes.join(' ')}`;
  return base;
}
const context = tokens => Number.isFinite(tokens) ? `${tokens / 1024}K context` : 'Context unavailable';
// Prepare is the step a self-provisioning engine adds: the workspace build, done beside the
// serving model, before the drain. A switch that needs no build runs it as a no-op.
const phaseNames = {checking: 'Check', preparing: 'Prepare', draining: 'Drain', loading: 'Load', verifying: 'Verify'};
const phases = Object.keys(phaseNames);
let status = null, operation = null, selected = null, connected = false, submitting = false;
let topologySignature = '', registrySignature = '', detailsSignature = '', lastStatus = 0, timer;
let pending = null, requestUnknown = false, polling = false, actionError = '';
function schedule(delay) { clearTimeout(timer); timer=setTimeout(tick,delay); }
try { pending = JSON.parse(sessionStorage.getItem('runtime-switch-request')); } catch (_) { /* Storage may be disabled. */ }
if (pending && typeof pending.request_id !== 'string') pending = null;
// A restore request is kept apart from a switch request: it names no profile and expects nothing
// to have been observed first, because it is the lever for a machine that is not serving.
let restorePending = null;
try { restorePending = JSON.parse(sessionStorage.getItem('runtime-restore-request')); } catch (_) { /* As above. */ }
if (restorePending && typeof restorePending.request_id !== 'string') restorePending = null;
function rememberRestore(request) {
  restorePending = request;
  try {
    if (request) sessionStorage.setItem('runtime-restore-request', JSON.stringify(request));
    else sessionStorage.removeItem('runtime-restore-request');
  } catch (_) { /* The server receipt also survives refresh and reconnect. */ }
}
function remember(request) {
  pending = request;
  try {
    if (request) sessionStorage.setItem('runtime-switch-request', JSON.stringify(request));
    else sessionStorage.removeItem('runtime-switch-request');
  } catch (_) { /* Server receipts also survive refresh and appear in status. */ }
}
function running() { return operation?.status === 'running'; }
function admissionsPaused() { return status?.maintenance?.draining || (running() && operation.phase !== 'checking'); }
function tellError(message) { el('error').hidden = !message; el('error').textContent = message || ''; }
function setOperation(next) {
  if (!next) return;
  if (pending?.request_id === next.id) { remember(null); requestUnknown = false; }
  if (restorePending?.request_id === next.id) rememberRestore(null);
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
  return `${hex.slice(0,8)}-${hex.slice(8,12)}-${hex.slice(12,16)}-${hex.slice(16,20)}-${hex.slice(20)}`;
}
function renderRegistry() {
  const configs = status?.configurations?.selectable || [];
  const signature = JSON.stringify([configs, status?.configurations?.always_included]);
  if (signature !== registrySignature) {
    registrySignature = signature;
    el('config-list').replaceChildren();
    for (const config of configs) {
      const label = node('label', '', 'profile-option');
      const input = node('input'); input.type = 'radio'; input.name = 'profile'; input.value = config.profile;
      input.addEventListener('change', () => { selected = input.value; actionError=''; tellError(''); updateControls(); });
      const text = node('span', '', 'profile-label');
      text.append(node('strong', configTitle(config)));
      for (const [role, detail] of configRows(config)) {
        const row = node('small', '', 'profile-row');
        row.append(node('span', role, 'profile-role'), node('span', detail));
        text.append(row);
      }
      const marker = node('span', '', 'profile-state'); marker.dataset.profile = config.profile;
      label.append(input, text, marker); el('config-list').append(label);
    }
  }
  if (running()) selected = operation.target_profile;
  if (!selected || !configs.some(c => c.profile === selected)) selected = pending?.profile || status?.profile || configs[0]?.profile;
  updateControls();
}
function updateControls() {
  const blocked = !connected || submitting || running() || requestUnknown || !status?.switching?.available;
  el('profile-options').disabled = blocked;
  for (const radio of el('config-list').querySelectorAll('input')) radio.checked = radio.value === selected;
  for (const marker of el('config-list').querySelectorAll('.profile-state')) {
    const previous = running() && marker.dataset.profile === operation.from_profile && ['loading', 'verifying', 'restoring'].includes(operation.phase);
    marker.textContent = previous ? 'Previous' : running() && marker.dataset.profile === operation.target_profile ? 'Selected' : marker.dataset.profile === status?.profile ? 'Active' : '';
    marker.classList.toggle('previous', previous);
  }
  const active = selected === status?.profile;
  el('impact').textContent = active ? '' : switchImpact(configFor(selected));
  el('apply').disabled = blocked || active || !selected;
  el('apply').textContent = submitting ? 'Starting switch…' : running() ? 'Switch in progress' : !connected ? 'Status unavailable' : active ? `${profileName(selected)} is active` : `Switch to ${profileName(selected)}`;
  el('switch-reason').textContent = requestUnknown ? 'Checking whether the request was accepted. It will not be sent again automatically.' : running() ? '' : !connected ? 'Reconnect to view the current runtime before switching.' : status?.switching?.reason || '';
}
function renderTopology() {
  const services = status?.services || [];
  const signature = JSON.stringify([services.map(s => [s.role,s.name,s.model,s.context_tokens,s.gpu_ids,s.gpu_names,s.vision_gpu_id,s.vision_gpu_name,s.exclusive,s.healthy]), status?.offline_roles, running() ? [operation.target_profile,operation.phase] : null, status?.maintenance?.draining, connected]);
  if (signature === topologySignature) return;
  topologySignature = signature;
  const graph = el('topology'); graph.replaceChildren();
  if (!services.length) { graph.append(node('p', 'GPU assignments are unavailable until runtime status is received.', 'secondary')); return; }
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); svg.classList.add('connections'); svg.setAttribute('aria-hidden', 'true'); graph.append(svg);
  const shared = services.find(s => s.vision_gpu_id);
  graph.classList.toggle('no-vision', !shared);
  services.forEach((service, index) => {
    const coding = service.role ? service.role === 'coding' : index === 0;
    const group = coding ? 'day' : 'night';
    const changing = coding && running() && ['loading','verifying','restoring'].includes(operation.phase);
    const bubble = node('article', '', `model-bubble ${group}${(!service.healthy || changing) ? ' unavailable' : ''}`); bubble.dataset.model = group;
    const slots = coding ? configFor(status.profile)?.parallel_slots : 1;
    const label = service.name || service.model || 'Model unavailable';
    bubble.append(node('h3', coding ? 'Coding' : 'Everyday'), node('p', `${label}${slotsText(slots, label)}`));
    if (changing) {
      bubble.querySelector('p').textContent = operation.phase === 'restoring' ? `Restoring ${profileName(operation.from_profile)}` : `Loading ${profileName(operation.target_profile)}`;
    }
    let health = !connected ? 'Status unavailable' : changing ? `${operation.phase === 'restoring' ? 'Restoring' : operation.phase === 'verifying' ? 'Verifying' : 'Loading'} · not ready` : service.healthy ? admissionsPaused() ? 'Loaded · new requests paused' : 'Ready' : 'Unavailable';
    if (!service.vision_gpu_id) health += ' · CPU vision';
    bubble.append(node('p', health, 'model-health')); graph.append(bubble);
    const set = node('div', '', `gpu-set ${group}`);
    (service.gpu_ids || []).forEach((id, i) => {
      const gpu = gpuNode(service.gpu_names?.[i] || `Text GPU ${i + 1}`, '', group); gpu.dataset.owner = group;
      set.append(gpu);
    });
    graph.append(set);
  });
  // A resident that is not loaded is only worth a bubble when nothing answers its service ID
  // either. A solo configuration that answers `nighttime` from its own model has the role offline
  // and no offline service, and saying "off" there would be a lie the page tells about a model
  // that is answering.
  const stopped = (status?.offline_services || []).find(s => (s.aliases || []).includes('nighttime'));
  if (status?.offline_roles?.includes('everyday') && stopped) {
    const offline = node('article', '', 'model-bubble night unavailable');
    offline.append(node('h3', 'Everyday'), node('p', `${stopped.display_name || 'The Everyday model'} is not loaded while ${profileName(status.profile)} uses its GPUs`), node('p', 'Not running', 'model-health'));
    graph.append(offline);
  }
  if (shared) graph.append(gpuNode(shared.vision_gpu_name || 'Vision GPU', shared.exclusive ? 'Vision + MTP draft' : 'Shared vision', 'shared vision-node'));
  drawConnections();
}
function gpuNode(name, role, cls) {
  const gpu = node('div', '', `gpu-node ${cls}`);
  gpu.append(node('div', '', 'gpu-dot'), node('p', name));
  if (role) gpu.append(node('p', role, 'gpu-role'));
  return gpu;
}
function drawConnections() {
  const graph = el('topology'), svg = graph.querySelector('.connections');
  if (!svg) return;
  const bounds = graph.getBoundingClientRect();
  svg.setAttribute('viewBox', `0 0 ${bounds.width} ${bounds.height}`); svg.replaceChildren();
  const compact = window.matchMedia('(max-width:600px)').matches;
  const rect = element => { const r = element.getBoundingClientRect(); return {x:r.left-bounds.left,y:r.top-bounds.top,width:r.width,height:r.height}; };
  function path(d, cls) { const p = document.createElementNS(svg.namespaceURI, 'path'); p.setAttribute('d', d); p.setAttribute('class', cls); svg.append(p); }
  if (!compact) {
    const dots = [...graph.querySelectorAll('.gpu-dot')].map(rect).sort((a,b) => a.x-b.x);
    if (dots.length) path(`M ${dots[0].x-15} ${dots[0].y+13} H ${dots.at(-1).x+59}`, 'rail');
  }
  for (const group of ['day','night']) {
    const bubble = graph.querySelector(`[data-model="${group}"]`);
    if (!bubble) continue;
    const source = rect(bubble);
    const dots = [...graph.querySelectorAll(`[data-owner="${group}"] .gpu-dot`)];
    dots.forEach((dot, i) => {
      const target = rect(dot), sx = source.x+source.width/2+(i ? 10 : -10), sy=source.y+source.height, tx=target.x+target.width/2, ty=target.y;
      const bend=(ty-sy)*.55;
      path(`M ${sx} ${sy} C ${sx} ${sy+bend}, ${tx} ${ty-bend}, ${tx} ${ty}`, `${group}${group==='day' && running() && ['loading','verifying','restoring'].includes(operation.phase) ? ' changing' : ''}`);
    });
    const vision = graph.querySelector('.vision-node .gpu-dot');
    if (vision) {
      const target=rect(vision), tx=target.x+target.width/2, ty=target.y;
      if (compact) {
        const right=group==='day', sx=right?source.x+source.width:source.x, sy=source.y+source.height/2, edge=right?bounds.width-5:5;
        path(`M ${sx} ${sy} C ${edge} ${sy}, ${edge} ${sy+12}, ${edge} ${sy+28} V ${ty-20} Q ${edge} ${ty+13}, ${edge+(right?-25:25)} ${ty+13} H ${tx+(right?22:-22)}`, `${group} shared`);
      } else {
        const sx=source.x+source.width*(group==='day'?.68:.32), sy=source.y+source.height;
        path(`M ${sx} ${sy} C ${sx} ${sy+38}, ${tx} ${ty-42}, ${tx} ${ty}`, `${group} shared`);
      }
    }
  }
}
new ResizeObserver(drawConnections).observe(el('topology'));
function renderDetails() {
  const signature = JSON.stringify([status?.services,status?.deployed_revision,status?.source_deployment]);
  if (signature === detailsSignature) return;
  detailsSignature = signature; el('services').replaceChildren();
  for (const service of status?.services || []) {
    const article = node('article', '', 'service-detail'), facts = node('dl');
    article.append(node('h3', service.name));
    for (const [label,value] of [['Model',service.model],['Context',context(service.context_tokens)],['Container',service.container_name],['Running since',service.started_at ? new Date(service.started_at).toLocaleString() : 'Unavailable'],['Text GPUs',service.gpu_names?.join(' + ') || 'Names unavailable'],['Vision',service.vision_gpu_id ? `${service.vision_gpu_name || 'Vision GPU'} · ${service.vision_device} · ${service.exclusive ? 'exclusive' : 'shared'}` : 'CPU'],['Configuration differences',service.differences?.join(', ') || 'None']]) {
      facts.append(node('dt',label),node('dd',value || 'Unavailable'));
    }
    article.append(facts); el('services').append(article);
  }
  el('revision').textContent = status?.deployed_revision || 'Awaiting deployment';
  const deployment = status?.source_deployment;
  el('deploy-state').textContent = deployment ? `Source deployment: ${deployment.phase.replaceAll('-',' ')}` : 'No source deployment recorded';
  el('deploy-detail').textContent = deployment ? [deployment.finished_at ? new Date(deployment.finished_at).toLocaleString() : 'In progress',deployment.revision?.slice(0,12),deployment.error].filter(Boolean).join(' · ') : 'Profile switches are reported above, separately from source releases.';
}
function renderProgress() {
  const panel = el('operation'); panel.hidden = !operation;
  if (!operation) return;
  panel.className = `operation ${operation.status}`;
  const target = profileName(operation.target_profile);
  // A restoration is a different promise from a switch, so it is titled as one: nothing was
  // chosen, and the point is that the previous configuration is serving again.
  const title = operation.kind === 'restore'
    ? {running: `Restoring ${target}`, succeeded: `${target} is serving`, recovered: `${target} is serving`, failed: 'Restoration did not begin', 'needs-attention': 'Restoration needs attention'}
    : {running: operation.phase === 'restoring' ? `Restoring ${profileName(operation.from_profile)}` : `Switching to ${target}`,succeeded:`${target} is ready`,failed:'Switch did not complete',recovered:`Switch failed · ${profileName(operation.from_profile)} restored`,'needs-attention':'Recovery needs attention'};
  el('operation-title').textContent = title[operation.status] || (operation.kind === 'restore' ? 'Restoration result' : 'Switch result');
  const elapsed = Math.max(0, Math.floor(((operation.finished_at ? Date.parse(operation.finished_at) : Date.now()) - Date.parse(operation.started_at))/1000));
  el('elapsed').textContent = `${running()?'Elapsed':'Duration'} ${Math.floor(elapsed/60)}m ${elapsed%60}s`;
  const stepKey = `${operation.status}:${operation.phase}`;
  if (el('steps').dataset.state !== stepKey) {
    el('steps').dataset.state=stepKey; el('steps').replaceChildren();
    el('steps').hidden=!running() || operation.phase==='restoring';
    phases.forEach((phase,index) => {
      const current=phases.indexOf(operation.phase);
      const step=node('li','',index<current?'done':index===current?'current':'');
      const mark=node('span',index<current?'✓':'','step-mark'); mark.setAttribute('aria-hidden','true');
      step.append(mark,node('span',phaseNames[phase]));
      if(index===current) step.setAttribute('aria-current','step');
      step.append(node('span',index<current?' completed':index===current?' in progress':' pending','sr-only'));
      el('steps').append(step);
    });
  }
  const message = operation.message || '';
  if (el('operation-message').textContent!==message) el('operation-message').textContent=message;
  el('admissions').textContent = !connected ? 'Connection lost. The switch may still be running; reconnecting to check its status.' : admissionsPaused() ? 'New requests are paused for both models.' : operation.status==='needs-attention' ? (operation.kind === 'restore' ? 'Nothing is serving yet. The runtime schedules the next attempt itself, and you can run the restoration again below.' : 'Check current runtime health and follow the documented recovery procedure.') : '';
}
// The one lever for a machine that is not serving. It runs the same restoration the runtime's own
// supervisor runs, so nobody has to reboot a host to get a model back.
function secondsUntil(timestamp) {
  const seconds = Math.round((Date.parse(timestamp) - Date.now()) / 1000);
  if (!Number.isFinite(seconds) || seconds <= 0) return '';
  return ` in ${seconds < 120 ? `${seconds} seconds` : `${Math.round(seconds / 60)} minutes`}`;
}
function renderRestoration() {
  const section = el('restoration');
  const service = status?.service || {};
  const restoring = running() && operation?.kind === 'restore';
  const visible = connected && (restoring || service.pending || service.serving === false);
  section.hidden = !visible;
  if (!visible) return;
  const name = profileName(service.profile || status?.profile);
  const record = status?.restoration || {};
  const attempts = record.attempts || 1;
  el('restore-state').textContent = restoring ? `Restoring ${name}. New requests stay paused until it is serving.`
    : service.pending ? `Nothing is serving. The runtime is retrying the restoration${secondsUntil(record.next_attempt_at)}, after ${attempts} attempt${attempts === 1 ? '' : 's'}.`
    : `Nothing is serving and no restoration is pending. Run it now; this machine does not need to be restarted.`;
  const unavailable = status?.restoring && status.restoring.available === false ? status.restoring.reason : '';
  el('restore').disabled = submitting || !connected || restoring || Boolean(unavailable) || service.restorable === false;
  el('restore-reason').textContent = service.restorable === false
    ? 'No last-known-good release is recorded on this host, so there is nothing to restore to.'
    : unavailable || (restoring ? 'The restoration is already running.' : '');
}
// Where this machine's catalog lives, in one line. "Models running here" and "published there" are
// two different failures and an operator should not need a shell to tell them apart; the router's
// complaint about its own network is shown as what it is — a reachability fact about the address,
// not this machine being unwell.
const publicationLine=()=>{
  const p=status?.publication, warn=[...new Set((p?.sinks||[]).flatMap(s=>s.warnings||[]))]
    .map(w=>w.startsWith('ROUTER_CANNOT_REACH_BACKEND:')?'router cannot reach these backends':w);
  if (!connected) return 'Publication state unavailable';
  if (!p) return 'Catalog not published yet';
  if (p.error) return `Catalog publication failing: ${p.error}`;
  if (p.retired_at) return `Catalog retired ${new Date(p.retired_at).toLocaleString()}`;
  const sinks=(p.sinks||[]).map(s=>s.kind==='file'?'the local router':`${String(s.name).replace(/^\w+:\/\//,'')} revision ${String(s.revision||'').slice(0,8)}`);
  return `Catalog published to ${sinks.join(' and ')}${warn.length?` · ${warn.join(' · ')}`:''}`;
};
function render() {
  const health=el('health');
  health.textContent=!connected?'Disconnected':running()?'Switching':status?.ready?'Ready':operation?.status==='needs-attention'?'Needs attention':status?.maintenance?.draining?'Maintenance':'Needs attention';
  health.className=`badge ${connected && status?.ready && !running()?'ready':'pending'}`;
  const counts = status?.maintenance;
  el('requests').textContent = connected && counts ? `${counts.active_requests} active · ${counts.queued_requests} queued` : 'Requests unavailable';
  el('publication').textContent = publicationLine();
  renderRegistry(); renderTopology(); renderProgress(); renderRestoration(); renderDetails();
  if (connected) el('updated').textContent=`Status updated ${new Date(status.updated_at || Date.now()).toLocaleTimeString()}`;
}
async function tick() {
  clearTimeout(timer);
  if (polling || submitting) { schedule(1000); return; }
  polling=true;
  try {
    if (!status || Date.now()-lastStatus>=5000 || !connected) {
      const next = await getJSON('/api/status');
      if (!status || !next.updated_at || next.updated_at >= status.updated_at) status=next;
      lastStatus=Date.now(); connected=true;
      setOperation(next.operation);
      tellError(actionError || next.startup_error || next.error || next.configurations?.error);
    }
    const id=pending?.request_id || restorePending?.request_id || (running()?operation.id:null);
    if (id) {
      try { setOperation(await getJSON(`/api/operations/${encodeURIComponent(id)}`)); }
      catch(error) {
        if (error.httpStatus!==404) throw error;
        // An unrecorded request may be retried explicitly with the SAME ID. Never
        // automatically POST after a lost response, refresh, or reconnect.
        requestUnknown=false;
        tellError(restorePending?.request_id===id ? 'No operation was recorded for that request. You can run the restoration again.' : 'No operation was recorded for that request. You can try the switch again.');
      }
    }
  } catch(error) {
    connected=false;
    tellError('Runtime status is unavailable. A switch may still be running; reconnecting without sending another request.');
  }
  render();
  // While a restoration is scheduled the page keeps watching at one-second cadence, because the
  // runtime may bring the service back between polls and that is worth seeing immediately.
  polling=false; schedule(running() || pending || restorePending || requestUnknown || status?.service?.pending ? 1000 : 5000);
}
// One action, no confirmation: it is the same restoration the runtime's supervisor performs, and
// the page is read at a moment when the machine is not answering anyone.
el('restore').addEventListener('click', async () => {
  if (el('restore').disabled || submitting) return;
  submitting=true; actionError=''; tellError(''); render(); clearTimeout(timer);
  // Remembered before it is sent, and a lost response is never reposted automatically — the same
  // rule the switch follows, so a retry cannot start a second restoration.
  const request = restorePending || {request_id:uuid()};
  rememberRestore(request);
  try {
    const result=await getJSON('/api/restore-last-known-good',{method:'POST',headers:{'Content-Type':'application/json','X-Runtime-CSRF':status.csrf_token},body:JSON.stringify(request)});
    setOperation(result.operation); requestUnknown=false;
  } catch(error) {
    if (error.httpStatus && error.httpStatus<500) { rememberRestore(null); requestUnknown=false; actionError=error.message; tellError(actionError); }
    else { connected=false; requestUnknown=true; tellError('Connection lost while starting the restoration. Checking whether it was accepted.'); }
  } finally { submitting=false; lastStatus=0; render(); schedule(1000); }
});
el('switch-form').addEventListener('submit',async event => {
  event.preventDefault();
  if (el('apply').disabled) return;
  const target = status?.configurations?.selectable?.find(c => c.profile === selected);
  // What happens to the second resident is stated from the target's own service IDs, not from the
  // word "exclusive": stopping a resident and stopping a service are different promises.
  const stopped = answers(target, 'nighttime') ? `${profileName(selected)} answers its requests in the meantime`
    : 'its requests will fail until you choose a configuration that loads it again';
  if (target?.exclusive && !configFor(status?.profile)?.exclusive && pending?.profile !== selected && !confirm(`${profileName(selected)} uses ${ownsTextGpus()}, so the Everyday model will stop. ${stopped}. Continue?`)) return;
  submitting=true; actionError=''; tellError(''); updateControls(); clearTimeout(timer);
  const request = pending && pending.profile===selected && pending.expected_profile===status.profile && pending.expected_revision===status.deployed_revision ? pending : {profile:selected,request_id:uuid(),expected_profile:status.profile,expected_revision:status.deployed_revision};
  remember(request);
  try {
    const result=await getJSON('/api/profile-switch',{method:'POST',headers:{'Content-Type':'application/json','X-Runtime-CSRF':status.csrf_token},body:JSON.stringify(request)});
    setOperation(result.operation); requestUnknown=false;
  } catch(error) {
    if (error.httpStatus && error.httpStatus<500) { remember(null); requestUnknown=false; actionError=error.message; tellError(actionError); }
    else { connected=false; requestUnknown=true; tellError('Connection lost while starting the switch. Checking whether it was accepted.'); }
  } finally { submitting=false; lastStatus=0; render(); schedule(1000); }
});
tick();
