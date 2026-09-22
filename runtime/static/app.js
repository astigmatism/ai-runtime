'use strict';
const el = id => document.getElementById(id);
function node(tag, text, cls) { const n = document.createElement(tag); n.textContent = text; if (cls) n.className = cls; return n; }
async function refresh() {
  try {
    const response = await fetch('/api/status', {cache: 'no-store'});
    if (!response.ok) throw new Error('Status service unavailable');
    const s = await response.json();
    el('health').textContent = s.ready ? 'Ready' : s.maintenance?.draining ? 'Maintenance' : 'Needs attention';
    el('health').className = 'badge ' + (s.ready ? 'ready' : 'pending');
    el('profile').textContent = s.profile || '—';
    el('revision').textContent = s.deployed_revision ? s.deployed_revision.slice(0, 12) : 'Awaiting migration';
    el('revision').title = s.deployed_revision || '';
    el('requests').textContent = s.maintenance ? `${s.maintenance.active_requests} active · ${s.maintenance.queued_requests} queued` : 'Unavailable';
    const error = s.startup_error || s.error;
    el('error').hidden = !error; el('error').textContent = error || '';
    el('services').replaceChildren();
    for (const service of s.services || []) {
      const card = node('article', '', 'model');
      const top = node('div', '', 'model-top');
      top.append(node('h2', service.name), node('span', service.healthy ? 'Healthy' : 'Unavailable', 'indicator'));
      card.append(top, node('p', service.model, 'model-id'));
      const facts = node('dl', '');
      for (const [label, value] of [['Context', `${(service.context_tokens / 1024).toLocaleString()}K tokens`], ['Text GPUs', service.gpu_names?.join(' + ') || service.gpu_ids.join(' + ')], ['Vision', service.vision_gpu_id ? `${service.vision_gpu_name || service.vision_gpu_id} · ${service.vision_device}${service.vision_gpu_shared ? ' · shared' : ''}` : 'CPU'], ['Container', service.container_name], ['Running since', service.started_at ? new Date(service.started_at).toLocaleString() : '—']]) facts.append(node('dt', label), node('dd', value));
      card.append(facts);
      if (service.differences.length) card.append(node('p', 'Configuration differences: ' + service.differences.join(', '), 'warning'));
      el('services').append(card);
    }
    const d = s.last_deployment;
    el('deploy-state').textContent = d ? d.phase.replaceAll('-', ' ') : 'No deployment recorded';
    el('deploy-detail').textContent = d ? [d.finished_at ? new Date(d.finished_at).toLocaleString() : 'In progress', d.revision?.slice(0, 12), d.error].filter(Boolean).join(' · ') : 'The first deployment will appear here.';
    el('updated').textContent = 'Updated ' + new Date().toLocaleTimeString();
  } catch (error) { el('health').textContent = 'Disconnected'; el('health').className = 'badge pending'; el('error').hidden = false; el('error').textContent = error.message; }
}
refresh(); setInterval(refresh, 5000);
