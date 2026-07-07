// Ego3DLM interactive 3D viewer — scene point cloud + GT/predicted skeletons.
// Data produced by export_viewer_data.py.  Rendering: three.js + OrbitControls.
import * as THREE from 'three';
import { OrbitControls } from '../js/vendor/OrbitControls.js';

const DATA = './data/';
const UP = new THREE.Vector3(0, 1, 0);
const BONE_R = 0.02, JOINT_R = 0.028, GT_KEY = 'gt';

// ── three basics ────────────────────────────────────────────────────────────
const stage = document.getElementById('stage');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(stage.clientWidth, stage.clientHeight);
stage.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf2f3f5);

const camera = new THREE.PerspectiveCamera(55, stage.clientWidth / stage.clientHeight, 0.05, 500);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.screenSpacePanning = true;

scene.add(new THREE.HemisphereLight(0xffffff, 0x5a5f66, 1.0));
const dir = new THREE.DirectionalLight(0xffffff, 1.6);
dir.position.set(3, 6, 4);
scene.add(dir);
const dir2 = new THREE.DirectionalLight(0xffffff, 0.5);
dir2.position.set(-4, 3, -2);
scene.add(dir2);

let grid = null;

// ── state ───────────────────────────────────────────────────────────────────
let meta = null;           // current sample meta json
let pcObj = null;          // THREE.Points
let skels = {};            // key -> {group, bones, joints, dummy, jdummy, color, lineFut, linePast}
let frame = 0, total = 1, playing = false, speed = 1;
let viewMode = 'forecast'; // 'forecast' | 'track'
const enabled = {};        // method key -> bool (GT always shown)
const clock = new THREE.Clock();
let acc = 0;

// ── skeleton construction ───────────────────────────────────────────────────
function makeSkeleton(colorHex, nJoints, nBones) {
  const color = new THREE.Color(colorHex);
  const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0.0 });
  mat.emissive = color.clone().multiplyScalar(0.18);
  const cyl = new THREE.CylinderGeometry(1, 1, 1, 10);
  const sph = new THREE.SphereGeometry(1, 12, 10);
  const bones = new THREE.InstancedMesh(cyl, mat, nBones);
  const joints = new THREE.InstancedMesh(sph, mat, nJoints);
  bones.frustumCulled = false; joints.frustumCulled = false;
  const group = new THREE.Group();
  group.add(bones); group.add(joints);
  scene.add(group);
  return { group, bones, joints, color, dummy: new THREE.Object3D(),
           lineFut: null, linePast: null };
}

function poseSkeleton(sk, joints, bonePairs) {
  const d = sk.dummy;
  for (let j = 0; j < joints.length; j++) {
    d.position.set(joints[j][0], joints[j][1], joints[j][2]);
    d.quaternion.identity();
    d.scale.setScalar(JOINT_R);
    d.updateMatrix();
    sk.joints.setMatrixAt(j, d.matrix);
  }
  const a = new THREE.Vector3(), b = new THREE.Vector3(), dirv = new THREE.Vector3(), mid = new THREE.Vector3();
  for (let i = 0; i < bonePairs.length; i++) {
    a.fromArray(joints[bonePairs[i][0]]);
    b.fromArray(joints[bonePairs[i][1]]);
    dirv.subVectors(b, a);
    const len = dirv.length() || 1e-6;
    mid.addVectors(a, b).multiplyScalar(0.5);
    d.position.copy(mid);
    d.quaternion.setFromUnitVectors(UP, dirv.normalize());
    d.scale.set(BONE_R, len, BONE_R);
    d.updateMatrix();
    sk.bones.setMatrixAt(i, d.matrix);
  }
  sk.joints.instanceMatrix.needsUpdate = true;
  sk.bones.instanceMatrix.needsUpdate = true;
}

function rootLine(seq, colorHex, opacity) {
  const pos = new Float32Array(seq.length * 3);
  for (let t = 0; t < seq.length; t++) {
    pos[t * 3] = seq[t][0][0]; pos[t * 3 + 1] = seq[t][0][1]; pos[t * 3 + 2] = seq[t][0][2];
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const m = new THREE.LineBasicMaterial({ color: new THREE.Color(colorHex), transparent: true, opacity });
  const line = new THREE.Line(g, m);
  line.frustumCulled = false;
  scene.add(line);
  return line;
}

// ── data loading ────────────────────────────────────────────────────────────
async function loadIndex() {
  const idx = await (await fetch(DATA + 'index.json')).json();
  const sel = document.getElementById('sample');
  idx.samples.forEach(s => {
    const o = document.createElement('option');
    o.value = s.id; o.textContent = s.label; sel.appendChild(o);
  });
  return idx.samples[0].id;
}

function disposeSample() {
  if (pcObj) { scene.remove(pcObj); pcObj.geometry.dispose(); pcObj.material.dispose(); pcObj = null; }
  Object.values(skels).forEach(sk => {
    scene.remove(sk.group);
    sk.bones.geometry.dispose(); sk.bones.material.dispose();
    sk.joints.geometry.dispose();
    [sk.lineFut, sk.linePast].forEach(l => { if (l) { scene.remove(l); l.geometry.dispose(); l.material.dispose(); } });
  });
  skels = {};
  if (grid) { scene.remove(grid); grid.geometry.dispose(); grid.material.dispose(); grid = null; }
}

async function loadSample(id) {
  document.getElementById('loading').classList.remove('hidden');
  const m = await (await fetch(DATA + id + '.json')).json();
  const buf = await (await fetch(DATA + m.pc_file)).arrayBuffer();
  disposeSample();
  meta = m;

  // point cloud, coloured by height
  const pos = new Float32Array(buf);
  const n = pos.length / 3;
  let ymin = Infinity, ymax = -Infinity;
  for (let i = 0; i < n; i++) { const y = pos[i * 3 + 1]; if (y < ymin) ymin = y; if (y > ymax) ymax = y; }
  const col = new Float32Array(pos.length);
  const lo = new THREE.Color(0x6b7480), hi = new THREE.Color(0xcfd4da), tmp = new THREE.Color();
  for (let i = 0; i < n; i++) {
    const a = (pos[i * 3 + 1] - ymin) / (ymax - ymin + 1e-6);
    tmp.copy(lo).lerp(hi, a);
    col[i * 3] = tmp.r; col[i * 3 + 1] = tmp.g; col[i * 3 + 2] = tmp.b;
  }
  const pg = new THREE.BufferGeometry();
  pg.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  pg.setAttribute('color', new THREE.BufferAttribute(col, 3));
  const pmat = new THREE.PointsMaterial({ size: parseFloat(document.getElementById('psize').value),
                                          vertexColors: true, sizeAttenuation: true });
  pcObj = new THREE.Points(pg, pmat);
  pcObj.frustumCulled = false;
  scene.add(pcObj);

  // skeletons: GT + each method
  const specs = [[GT_KEY, m.gt.color, m.gt]];
  for (const key of Object.keys(m.methods)) specs.push([key, m.methods[key].color, m.methods[key]]);
  for (const [key, color, entry] of specs) {
    const sk = makeSkeleton(color, m.n_joints, m.bones.length);
    if (key === GT_KEY) {
      sk.linePast = rootLine(m.gt.past, color, 0.35);
      sk.lineFut = rootLine(m.gt.future, color, 0.6);
    } else {
      sk.linePast = rootLine(entry.pred_past, color, 0.5);
      sk.lineFut = rootLine(entry.pred_future, color, 0.75);
    }
    skels[key] = sk;
  }

  // faint floor grid at ~feet level
  const span = Math.max(...['0', '2'].map(i => m.motion_max[+i] - m.motion_min[+i]), 6) + 6;
  grid = new THREE.GridHelper(Math.ceil(span), Math.ceil(span), 0xc4cad2, 0xd8dde3);
  grid.position.y = m.motion_min[1] - 0.05;
  grid.material.transparent = true; grid.material.opacity = 0.35;
  scene.add(grid);

  // framing
  total = (viewMode === 'track') ? m.n_past : (m.n_past + m.n_future);
  frame = 0;
  document.getElementById('timeline').max = String(total - 1);
  document.getElementById('s-label').textContent = m.label;
  buildMethodChips();
  frameCamera();
  setFrame(0);
  document.getElementById('loading').classList.add('hidden');
}

function frameCamera() {
  const mn = meta.motion_min, mx = meta.motion_max;
  const c = new THREE.Vector3((mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2);
  const size = Math.max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2], 1.5);
  const dist = Math.max(size * 2.4, 3.0);
  controls.target.copy(c);
  camera.position.copy(c).add(new THREE.Vector3(0.75, 0.55, 1).normalize().multiplyScalar(dist));
  camera.updateProjectionMatrix();
  controls.update();
}

// ── per-frame pose update ───────────────────────────────────────────────────
function setFrame(t) {
  frame = Math.max(0, Math.min(total - 1, Math.round(t)));
  const np = meta.n_past, bones = meta.bones;
  const inFuture = (viewMode === 'forecast') && (frame >= np);
  const fidx = frame - np;

  // GT always visible
  const gt = skels[GT_KEY];
  const gtJoints = (viewMode === 'track' || frame < np) ? meta.gt.past[Math.min(frame, np - 1)]
                                                        : meta.gt.future[fidx];
  poseSkeleton(gt, gtJoints, bones);
  gt.group.visible = true;
  gt.linePast.visible = true;
  gt.lineFut.visible = (viewMode === 'forecast');

  for (const key of Object.keys(meta.methods)) {
    const sk = skels[key], on = !!enabled[key];
    const m = meta.methods[key];
    let show = false, joints = null;
    if (on) {
      if (viewMode === 'track') { show = true; joints = m.pred_past[Math.min(frame, np - 1)]; }
      else if (inFuture) { show = true; joints = m.pred_future[fidx]; }
    }
    sk.group.visible = show;
    if (show) poseSkeleton(sk, joints, bones);
    sk.lineFut.visible = on && viewMode === 'forecast';
    sk.linePast.visible = on && viewMode === 'track';
  }

  document.getElementById('timeline').value = String(frame);
  document.getElementById('framelab').textContent = `frame ${frame + 1} / ${total}`;
  const phase = (viewMode === 'track') ? 'tracking past'
              : (frame < np ? 'given past' : 'predicting future');
  document.getElementById('phase').textContent = phase;
  updateCot();
}

function updateCot() {
  // show the leading enabled method's spatial reasoning (fallback: ours)
  const order = ['ours_withGRPO', ...Object.keys(meta.methods)];
  const key = order.find(k => enabled[k]) || 'ours_withGRPO';
  const cot = (meta.methods[key] && meta.methods[key].cot) || '';
  const el = document.getElementById('s-cot');
  el.innerHTML = cot ? `<b>${meta.methods[key].label} — spatial reasoning:</b> ${cot}` : '';
}

// ── method chips ────────────────────────────────────────────────────────────
function buildMethodChips() {
  const wrap = document.getElementById('methods');
  wrap.innerHTML = '';
  // GT chip (always on, non-toggle look)
  wrap.appendChild(chip(GT_KEY, meta.gt.label, meta.gt.color, null, true));
  for (const key of Object.keys(meta.methods)) {
    if (!(key in enabled)) enabled[key] = (key === 'ours_withGRPO');
    wrap.appendChild(chip(key, meta.methods[key].label, meta.methods[key].color, meta.methods[key].ade, false));
  }
}
function chip(key, label, color, ade, isGT) {
  const c = document.createElement('span');
  c.className = 'chip' + (isGT ? '' : (enabled[key] ? '' : ' off'));
  c.innerHTML = `<span class="sw" style="background:${color}"></span>${label}` +
                (ade != null ? ` <span class="ade">${ade.toFixed(2)}</span>` : '');
  if (!isGT) c.onclick = () => { enabled[key] = !enabled[key]; c.classList.toggle('off', !enabled[key]); setFrame(frame); };
  return c;
}

// ── loop ────────────────────────────────────────────────────────────────────
function animate() {
  requestAnimationFrame(animate);
  const dt = clock.getDelta();
  if (playing && meta) {
    acc += dt * meta.fps * speed;
    if (acc >= 1) { const step = Math.floor(acc); acc -= step; setFrame((frame + step) % total); }
  }
  controls.update();
  renderer.render(scene, camera);
}

// ── UI wiring ───────────────────────────────────────────────────────────────
function setPlaying(p) {
  playing = p;
  document.getElementById('play').textContent = p ? '❚❚ Pause' : '▶ Play';
}
document.getElementById('play').onclick = () => setPlaying(!playing);
document.getElementById('timeline').oninput = e => { setPlaying(false); setFrame(+e.target.value); };
document.getElementById('speed').onchange = e => { speed = parseFloat(e.target.value); };
document.getElementById('sample').onchange = e => { setPlaying(false); loadSample(e.target.value); };
document.getElementById('view').onchange = e => {
  viewMode = e.target.value;
  total = (viewMode === 'track') ? meta.n_past : (meta.n_past + meta.n_future);
  document.getElementById('timeline').max = String(total - 1);
  setPlaying(false); setFrame(0);
};
document.getElementById('psize').oninput = e => { if (pcObj) pcObj.material.size = parseFloat(e.target.value); };
document.getElementById('reset').onclick = () => frameCamera();
document.getElementById('full').onclick = () => {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.getElementById('app').requestFullscreen?.();
};
addEventListener('resize', () => {
  camera.aspect = stage.clientWidth / stage.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(stage.clientWidth, stage.clientHeight);
});

// ── go ──────────────────────────────────────────────────────────────────────
(async function () {
  const first = await loadIndex();
  await loadSample(first);
  setPlaying(true);
  animate();
})().catch(err => {
  document.getElementById('loading').textContent = 'Failed to load viewer: ' + err.message;
  console.error(err);
});
