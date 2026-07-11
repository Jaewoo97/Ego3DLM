// Ego3DLM interactive 3D viewer — scene point cloud + GT/predicted skeletons.
// Data produced by export_viewer_data.py.  Rendering: three.js + OrbitControls.
import * as THREE from 'three';
import { OrbitControls } from '../js/vendor/OrbitControls.js';

const DATA = './data/';
const UP = new THREE.Vector3(0, 1, 0);
const BONE_R = 0.02, JOINT_R = 0.028, GT_KEY = 'gt';
const TP_JOINTS = [6, 10, 14];       // head, right hand, left hand — the 3-point tracking input
const egoVid = document.getElementById('ego-vid');
const egoWrap = document.getElementById('ego');
const params = new URLSearchParams(location.search);
const LAYER = params.get('layer');   // cloud | tp | pose — isolate one layer for panel capture
const CAM = params.get('cam');       // 'head' — render the scene from the wearer's viewpoint
const HB = parseFloat(params.get('back') || '0.35');   // camera set-back behind the eyes (m)
const HR = parseFloat(params.get('rise') || '0.12');   // camera lift above the eyes (m)

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
let tpMesh = null;             // 3-point tracking markers (input)
let tpTrails = null;           // 3-point tracking trajectories (head + both hands)
let headCam = false;           // drive the camera from the wearer's head pose
let egoReady = false;
const tpDummy = new THREE.Object3D();

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

// Persistent faded "ghost" poses that visualise the observed-past motion — a set
// of sampled skeletons that stay visible so past and future are seen together.
function buildGhostTrail(poses, bonePairs, colorHex) {
  const base = new THREE.Color(colorHex), white = new THREE.Color(0xffffff);
  const nb = bonePairs.length;
  const mat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.62, depthWrite: false });
  const mesh = new THREE.InstancedMesh(new THREE.CylinderGeometry(1, 1, 1, 8), mat, poses.length * nb);
  mesh.frustumCulled = false; mesh.renderOrder = 1;
  const d = new THREE.Object3D(), a = new THREE.Vector3(), b = new THREE.Vector3(), dir = new THREE.Vector3(), mid = new THREE.Vector3();
  let inst = 0;
  for (let p = 0; p < poses.length; p++) {
    const age = poses.length > 1 ? p / (poses.length - 1) : 1;   // 0 oldest -> 1 newest
    const col = base.clone().lerp(white, 0.5 * (1 - age));        // older -> fainter
    const j = poses[p];
    for (let i = 0; i < nb; i++) {
      a.fromArray(j[bonePairs[i][0]]); b.fromArray(j[bonePairs[i][1]]);
      dir.subVectors(b, a); const len = dir.length() || 1e-6; mid.addVectors(a, b).multiplyScalar(0.5);
      d.position.copy(mid); d.quaternion.setFromUnitVectors(UP, dir.normalize());
      d.scale.set(BONE_R * 0.7, len, BONE_R * 0.7); d.updateMatrix();
      mesh.setMatrixAt(inst, d.matrix); mesh.setColorAt(inst, col); inst++;
    }
  }
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  scene.add(mesh);
  return mesh;
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
    if (sk.pastGhost) { scene.remove(sk.pastGhost); sk.pastGhost.geometry.dispose(); sk.pastGhost.material.dispose(); }
    [sk.lineFut, sk.linePast].forEach(l => { if (l) { scene.remove(l); l.geometry.dispose(); l.material.dispose(); } });
  });
  skels = {};
  if (grid) { scene.remove(grid); grid.geometry.dispose(); grid.material.dispose(); grid = null; }
  if (tpMesh) { scene.remove(tpMesh); tpMesh.geometry.dispose(); tpMesh.material.dispose(); tpMesh = null; }
  if (tpTrails) { tpTrails.forEach(l => { scene.remove(l); l.geometry.dispose(); l.material.dispose(); }); tpTrails = null; }
}

async function loadSample(id) {
  document.getElementById('loading').classList.remove('hidden');
  const m = await (await fetch(DATA + id + '.json')).json();
  const buf = await (await fetch(DATA + m.pc_file)).arrayBuffer();
  disposeSample();
  meta = m;
  // continuous forecast sequence: observed past -> eased bridge -> future
  meta._gtFwd = [...m.gt.past, ...(m.gt.bridge || []), ...m.gt.future];
  meta._futStart = m.n_past + (m.n_bridge || 0);

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
      sk.lineFut = rootLine(meta._gtFwd, color, 0.55);   // full continuous path (past->future)
      const K = Math.min(9, m.gt.past.length);
      const gi = Array.from({ length: K }, (_, i) => Math.round(i * (m.gt.past.length - 1) / (K - 1)));
      sk.pastGhost = buildGhostTrail(gi.map(i => m.gt.past[i]), m.bones, color);
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

  // 3-point tracking markers (input): head + both hands
  tpMesh = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 16, 12),
    new THREE.MeshStandardMaterial({ color: 0xf59e0b }), TP_JOINTS.length);
  tpMesh.material.emissive = new THREE.Color(0xf59e0b).multiplyScalar(0.35);
  tpMesh.frustumCulled = false; tpMesh.renderOrder = 3; scene.add(tpMesh);

  // 3-point tracking trails: head + both hands trajectories over the past, drawn
  // progressively (setDrawRange grows with the frame) to emphasise the input.
  tpTrails = TP_JOINTS.map((j) => {
    const g = new THREE.BufferGeometry();
    const pos = new Float32Array(m.n_past * 3);
    for (let t = 0; t < m.n_past; t++) {
      const jt = m.gt.past[t][j];
      pos[t * 3] = jt[0]; pos[t * 3 + 1] = jt[1]; pos[t * 3 + 2] = jt[2];
    }
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setDrawRange(0, 1);
    const line = new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0xf59e0b, transparent: true, opacity: 0.9 }));
    line.frustumCulled = false; line.renderOrder = 2; line.visible = false; scene.add(line);
    return line;
  });

  // egocentric RGB (input) — per-sample clip, synced to the observed past; hidden if absent
  egoReady = false; egoWrap.classList.add('hidden');
  egoVid.onloadeddata = () => { egoReady = true; egoWrap.classList.remove('hidden'); egoVid.pause(); };
  egoVid.onerror = () => { egoReady = false; egoWrap.classList.add('hidden'); };
  egoVid.src = 'ego/' + id + '.mp4';

  // framing — forecasting plays past -> bridge -> future as one continuous sequence
  total = (viewMode === 'track') ? m.n_past : meta._gtFwd.length;
  frame = 0;
  headCam = (CAM === 'head' && Array.isArray(meta.head_cam) && meta.head_cam.length > 0);
  if (headCam) {                          // egocentric viewpoint for the scene panel
    camera.fov = parseFloat(params.get('fov') || '80');
    camera.updateProjectionMatrix();
    controls.enabled = false;
  }
  document.getElementById('timeline').max = String(total - 1);
  document.getElementById('s-label').textContent = m.label;
  document.getElementById('s-gt').innerHTML = m.gt_text ? '<b>Ground-truth motion:</b> ' + m.gt_text : '';
  buildMethodChips();
  frameCamera();
  setFrame(0);
  document.getElementById('loading').classList.add('hidden');
}

function frameCamera() {
  if (headCam) return;                    // head-follow camera owns the view
  const mn = meta.motion_min, mx = meta.motion_max;
  const c = new THREE.Vector3((mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2);
  const size = Math.max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2], 1.5);
  const dist = Math.max(size * 2.4, 3.0);
  controls.target.copy(c);
  camera.position.copy(c).add(new THREE.Vector3(0.75, 0.55, 1).normalize().multiplyScalar(dist));
  camera.updateProjectionMatrix();
  controls.update();
}

// Egocentric head camera: look from just behind/above the wearer's eyes along the
// (smoothed, horizontal) head facing, so the scene panel matches the ego video.
const _eye = new THREE.Vector3(), _fwd = new THREE.Vector3(), _tgt = new THREE.Vector3();
function applyHeadCam(fr) {
  const hc = meta.head_cam; if (!hc || !hc.length) return;
  const i = Math.max(0, Math.min(hc.length - 1, fr));
  const p = hc[i].p, f = hc[i].f;
  _fwd.set(f[0], f[1], f[2]).normalize();
  _tgt.set(p[0], p[1], p[2]).addScaledVector(_fwd, 3.0);   // look ahead along facing
  _eye.set(p[0], p[1], p[2]).addScaledVector(_fwd, -HB);   // set back behind the eyes
  _eye.y += HR;                                            // and slightly above
  camera.up.set(0, 1, 0);
  camera.position.copy(_eye);
  camera.lookAt(_tgt);
}

// ── per-frame pose update ───────────────────────────────────────────────────
// Forecasting animates ONLY the future segment (every method starts at the same
// future timepoint); the observed past is shown as trajectory context, so there
// is no jump across the past->future segment gap. Tracking animates the past.
// Each skeleton is hidden once its own sequence ends — predictions are never
// padded to a common length.
function applyLayer() {   // isolate one layer for teaser-panel capture (?layer=cloud|tp|pose)
  const cloud = LAYER === 'cloud', tp = LAYER === 'tp', pose = LAYER === 'pose';
  if (pcObj) pcObj.visible = cloud;
  if (grid) grid.visible = false;
  if (tpMesh) tpMesh.visible = tp && tpMesh.visible;
  if (tpTrails) tpTrails.forEach(l => l.visible = tp && l.visible);   // trails ride with the tp layer
  for (const key of Object.keys(skels)) {
    const sk = skels[key];
    // tp layer shows the GT pose alongside the tracked points; pose layer shows GT + ours
    sk.group.visible = ((pose && (key === GT_KEY || key === 'ours_withGRPO')) || (tp && key === GT_KEY)) && sk.group.visible;
    if (sk.lineFut) sk.lineFut.visible = false;
    if (sk.linePast) sk.linePast.visible = false;
    if (sk.pastGhost) sk.pastGhost.visible = false;
  }
}

function setFrame(t) {
  frame = Math.max(0, Math.min(total - 1, Math.round(t)));
  const bones = meta.bones;
  const forecast = (viewMode === 'forecast');

  const gt = skels[GT_KEY];
  if (forecast) {                        // GT animates past -> bridge -> future
    poseSkeleton(gt, meta._gtFwd[frame], bones);
    gt.group.visible = true;
  } else {                               // tracking: observed past only
    const gvis = frame < meta.gt.past.length;
    gt.group.visible = gvis;
    if (gvis) poseSkeleton(gt, meta.gt.past[frame], bones);
  }
  gt.linePast.visible = !forecast;       // track: observed-past path
  gt.lineFut.visible = forecast;         // forecast: full continuous path
  if (gt.pastGhost) gt.pastGhost.visible = forecast;   // persistent past-motion ghosts

  const futStart = meta._futStart;
  for (const key of Object.keys(meta.methods)) {
    const sk = skels[key], on = !!enabled[key];
    let show = false, joints = null;
    if (on && forecast && frame >= futStart) {    // predictions overlay on the future only
      const idx = frame - futStart, seq = meta.methods[key].pred_future;
      if (idx < seq.length) { show = true; joints = seq[idx]; }   // no padding
    } else if (on && !forecast) {                 // tracking: pred_past
      const seq = meta.methods[key].pred_past;
      if (frame < seq.length) { show = true; joints = seq[frame]; }
    }
    sk.group.visible = show;
    if (show) poseSkeleton(sk, joints, bones);
    sk.lineFut.visible = on && forecast;
    sk.linePast.visible = on && !forecast;
  }

  // input visualisations: 3-point markers + egocentric RGB during the observed past
  const nPast = meta.n_past;
  const inPast = forecast ? (frame < nPast) : (frame < meta.gt.past.length);
  if (tpMesh) {
    tpMesh.visible = inPast;
    if (inPast) {
      const jp = forecast ? meta._gtFwd[frame] : meta.gt.past[frame];
      for (let i = 0; i < TP_JOINTS.length; i++) {
        const j = jp[TP_JOINTS[i]];
        tpDummy.position.set(j[0], j[1], j[2]); tpDummy.quaternion.identity();
        tpDummy.scale.setScalar(0.05); tpDummy.updateMatrix(); tpMesh.setMatrixAt(i, tpDummy.matrix);
      }
      tpMesh.instanceMatrix.needsUpdate = true;
    }
  }
  if (tpTrails) {                          // grow the head/hand trails up to the current past frame
    const kk = Math.max(1, Math.min(meta.n_past, frame + 1));
    for (const ln of tpTrails) { ln.visible = inPast; ln.geometry.setDrawRange(0, kk); }
  }
  if (egoReady) {
    const ct = Math.min(inPast ? frame : nPast - 1, nPast - 1) / meta.fps;
    if (Math.abs(egoVid.currentTime - ct) > 0.02) { try { egoVid.currentTime = ct; } catch (e) {} }
  }

  document.getElementById('timeline').value = String(frame);
  document.getElementById('framelab').textContent = `frame ${frame + 1} / ${total}`;
  document.getElementById('phase').textContent =
    forecast ? (frame < meta._futStart ? 'observed past · input' : 'predicting future · output') : 'tracking past · input';
  if (headCam) applyHeadCam(frame);
  if (LAYER) applyLayer();
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
  if (!headCam) controls.update();        // head-follow camera sets the view itself
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
  total = (viewMode === 'track') ? meta.n_past : meta._gtFwd.length;
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
