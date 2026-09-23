'use strict';
const el = id => document.getElementById(id);
const node = (tag, text = '', cls = '') => {
  const element = document.createElement(tag);
  element.textContent = text;
  if (cls) element.className = cls;
  return element;
};
const profileName = profile => ({daytime: 'Daytime', 'daytime-27b': 'Daytime-27B', nighttime: 'Nighttime'})[profile] || profile || 'Unknown configuration';
const modelName = model => ({'qwen3.8-flash-next-ad4.27': 'Flash-Next', 'qwen3.8-27b-q8_0': '27B Q8', 'qwen3.8-27b-abliterated-q6_k': '27B Abliterated'})[model] || model || 'Model unavailable';
const context = tokens => Number.isFinite(tokens) ? `${tokens / 1024}K context` : 'Context unavailable';
const phaseNames = {checking: 'Check', draining: 'Drain', loading: 'Load', verifying: 'Verify'};
const phases = Object.keys(phaseNames);
let status = null, operation = null, selected = null, connected = false, submitting = false;
let topologySignature = '', registrySignature = '', detailsSignature = '', lastStatus = 0, timer;
let pending = null, requestUnknown = false, polling = false, actionError = '';
function schedule(delay) { clearTimeout(timer); timer=setTimeout(tick,delay); }
try { pending = JSON.parse(sessionStorage.getItem('runtime-switch-request')); } catch (_) { /* Storage may be disabled. */ }
if (pending && typeof pending.request_id !== 'string') pending = null;
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
  const signature = JSON.stringify(configs);
  if (signature !== registrySignature) {
    registrySignature = signature;
    el('config-list').replaceChildren();
    for (const config of configs) {
      const label = node('label', '', 'profile-option');
      const input = node('input'); input.type = 'radio'; input.name = 'profile'; input.value = config.profile;
      input.addEventListener('change', () => { selected = input.value; actionError=''; tellError(''); updateControls(); });
      const text = node('span', '', 'profile-label');
      text.append(node('strong', profileName(config.profile)), node('small', `${modelName(config.model)} · ${context(config.context_tokens)}`));
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
  el('apply').disabled = blocked || active || !selected;
  el('apply').textContent = submitting ? 'Starting switch…' : running() ? 'Switch in progress' : !connected ? 'Status unavailable' : active ? `${profileName(selected)} is active` : `Switch to ${profileName(selected)}`;
  el('switch-reason').textContent = requestUnknown ? 'Checking whether the request was accepted. It will not be sent again automatically.' : running() ? '' : !connected ? 'Reconnect to view the current runtime before switching.' : status?.switching?.reason || '';
}
function renderTopology() {
  const services = status?.services || [];
  const signature = JSON.stringify([services.map(s => [s.role,s.name,s.model,s.context_tokens,s.gpu_ids,s.gpu_names,s.vision_gpu_id,s.vision_gpu_name,s.healthy]), running() ? [operation.target_profile,operation.phase] : null, status?.maintenance?.draining, connected]);
  if (signature === topologySignature) return;
  topologySignature = signature;
  const graph = el('topology'); graph.replaceChildren();
  if (!services.length) { graph.append(node('p', 'GPU assignments are unavailable until runtime status is received.', 'secondary')); return; }
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); svg.classList.add('connections'); svg.setAttribute('aria-hidden', 'true'); graph.append(svg);
  const shared = services.find(s => s.vision_gpu_id);
  graph.classList.toggle('no-vision', !shared);
  services.forEach((service, index) => {
    const daytime = service.role ? service.role === 'coding' : index === 0;
    const group = daytime ? 'day' : 'night';
    const changing = daytime && running() && ['loading','verifying','restoring'].includes(operation.phase);
    const bubble = node('article', '', `model-bubble ${group}${(!service.healthy || changing) ? ' unavailable' : ''}`); bubble.dataset.model = group;
    const title = changing ? `${profileName(operation.from_profile)} → ${profileName(operation.target_profile)}` : daytime ? profileName(status.profile) : 'Nighttime';
    bubble.append(node('h3', title), node('p', `${modelName(service.model)} · ${context(service.context_tokens)}`));
    if (changing) {
      const target = status.configurations?.selectable?.find(c => c.profile === operation.target_profile);
      bubble.querySelector('p').textContent = operation.phase === 'restoring' ? 'Restoring the previous configuration' : `${modelName(target?.model)} · ${context(target?.context_tokens)}`;
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
  if (shared) graph.append(gpuNode(shared.vision_gpu_name || 'Vision GPU', 'Shared vision', 'shared vision-node'));
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
    for (const [label,value] of [['Model',service.model],['Context',context(service.context_tokens)],['Container',service.container_name],['Running since',service.started_at ? new Date(service.started_at).toLocaleString() : 'Unavailable'],['Text GPUs',service.gpu_names?.join(' + ') || 'Names unavailable'],['Vision',service.vision_gpu_id ? `${service.vision_gpu_name || 'Vision GPU'} · ${service.vision_device} · shared` : 'CPU'],['Configuration differences',service.differences?.join(', ') || 'None']]) {
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
  const title = {running: operation.phase === 'restoring' ? `Restoring ${profileName(operation.from_profile)}` : `Switching to ${target}`,succeeded:`${target} is ready`,failed:'Switch did not complete',recovered:`Switch failed · ${profileName(operation.from_profile)} restored`,'needs-attention':'Recovery needs attention'};
  el('operation-title').textContent = title[operation.status] || 'Switch result';
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
  el('admissions').textContent = !connected ? 'Connection lost. The switch may still be running; reconnecting to check its status.' : admissionsPaused() ? 'New requests are paused for both models.' : operation.status==='needs-attention' ? 'Check current runtime health and follow the documented recovery procedure.' : '';
}
function render() {
  const health=el('health');
  health.textContent=!connected?'Disconnected':running()?'Switching':status?.ready?'Ready':operation?.status==='needs-attention'?'Needs attention':status?.maintenance?.draining?'Maintenance':'Needs attention';
  health.className=`badge ${connected && status?.ready && !running()?'ready':'pending'}`;
  const counts = status?.maintenance;
  el('requests').textContent = connected && counts ? `${counts.active_requests} active · ${counts.queued_requests} queued` : 'Requests unavailable';
  renderRegistry(); renderTopology(); renderProgress(); renderDetails();
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
    const id=pending?.request_id || (running()?operation.id:null);
    if (id) {
      try { setOperation(await getJSON(`/api/operations/${encodeURIComponent(id)}`)); }
      catch(error) {
        if (error.httpStatus!==404) throw error;
        // An unrecorded request may be retried explicitly with the SAME ID. Never
        // automatically POST after a lost response, refresh, or reconnect.
        requestUnknown=false;
        tellError('No operation was recorded for that request. You can try the switch again.');
      }
    }
  } catch(error) {
    connected=false;
    tellError('Runtime status is unavailable. A switch may still be running; reconnecting without sending another request.');
  }
  render();
  polling=false; schedule(running() || pending || requestUnknown ? 1000 : 5000);
}
el('switch-form').addEventListener('submit',async event => {
  event.preventDefault();
  if (el('apply').disabled) return;
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
