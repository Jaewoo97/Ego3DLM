// Semantic scene-awareness QA browser. Data from export_semantic_data.py.
const DATA = 'semantic/';
const TYPE_COLORS = {
  'object': '#3b82f6', 'place': '#10b981', 'color': '#f59e0b',
  'object nature': '#8b5cf6', 'number': '#14b8a6', 'other': '#6b7280',
};

let db = null;
const enabled = {};                       // type -> bool

const $scene = document.getElementById('scene');
const $types = document.getElementById('types');
const $grid  = document.getElementById('grid');
const $count = document.getElementById('count');
const $empty = document.getElementById('empty');

function chip(type) {
  const c = document.createElement('span');
  c.className = 'tchip';
  c.innerHTML = `<span class="dot" style="background:${TYPE_COLORS[type] || '#888'}"></span>${type}`;
  c.onclick = () => { enabled[type] = !enabled[type]; c.classList.toggle('off', !enabled[type]); render(); };
  return c;
}

function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>]/g, m => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[m]));
}

function render() {
  const scene = db.scenes.find(s => s.id === $scene.value);
  $grid.innerHTML = '';
  let nCards = 0, nQA = 0;
  for (const f of scene.frames) {
    const qa = f.qa.filter(q => enabled[q.type]);
    if (!qa.length) continue;
    nCards++; nQA += qa.length;
    const card = document.createElement('div');
    card.className = 'card';
    const img = document.createElement('img');
    img.loading = 'lazy';
    img.src = DATA + 'img/' + f.img;
    img.alt = 'egocentric frame';
    card.appendChild(img);
    for (const q of qa) {
      const el = document.createElement('div');
      el.className = 'qa';
      const color = TYPE_COLORS[q.type] || '#888';
      el.innerHTML =
        `<span class="badge" style="background:${color}">${esc(q.type)}</span>` +
        `<div class="q">${esc(q.q)}</div><div class="a">${esc(q.a)}</div>`;
      card.appendChild(el);
    }
    $grid.appendChild(card);
  }
  $empty.style.display = nCards ? 'none' : 'block';
  $count.textContent = `${nCards} frames · ${nQA} QA`;
}

(async function () {
  db = await (await fetch(DATA + 'semantic.json')).json();
  db.scenes.forEach(s => {
    const o = document.createElement('option');
    o.value = s.id; o.textContent = s.label; $scene.appendChild(o);
  });
  // types present in the data, in canonical order
  const present = new Set();
  db.scenes.forEach(s => s.frames.forEach(f => f.qa.forEach(q => present.add(q.type))));
  (db.types || [...present]).filter(t => present.has(t)).forEach(t => {
    enabled[t] = true; $types.appendChild(chip(t));
  });
  $scene.onchange = render;
  render();
})().catch(e => { $grid.innerHTML = '<p style="padding:20px;color:#a00">Failed to load QA: ' + e.message + '</p>'; });
