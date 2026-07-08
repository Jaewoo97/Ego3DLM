// Spatial scene-awareness viewer: point cloud + tracked person + directional
// clearance wedges (front/left/right), coloured LOW/MID/HIGH, best dir highlighted.
import * as THREE from 'three';
import { OrbitControls } from '../js/vendor/OrbitControls.js';

const DATA = 'spatial/';
const UP = new THREE.Vector3(0, 1, 0);
const BONE_R = 0.02, JOINT_R = 0.028;
const HALF = THREE.MathUtils.degToRad(60), SEG = 22;   // ±60° sensing cones (front/left/right overlap ~30°)
const CAT_COL = [0xef4444, 0xf59e0b, 0x22c55e];      // LOW / MID / HIGH
const CAT_TXT = ['Low', 'Mid', 'High'];
const DIR_TXT = ['Front', 'Left', 'Right'];

const stage = document.getElementById('stage');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(stage.clientWidth, stage.clientHeight);
stage.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf2f3f5);
const camera = new THREE.PerspectiveCamera(55, stage.clientWidth / stage.clientHeight, 0.05, 500);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.08; controls.screenSpacePanning = true;
scene.add(new THREE.HemisphereLight(0xffffff, 0x5a5f66, 1.0));
const dl = new THREE.DirectionalLight(0xffffff, 1.5); dl.position.set(3, 6, 4); scene.add(dl);

let meta = null, pcObj = null, grid = null;
let boneMesh = null, jointMesh = null, wedges = [], arrow = null;
let frame = 0, total = 1, playing = false, speed = 1;
const clock = new THREE.Clock(); let acc = 0;
const dummy = new THREE.Object3D();

// ── wedge geometry (flat triangle fan, updated per frame) ───────────────────
function makeWedge() {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(SEG * 3 * 3), 3));
  const m = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.34, side: THREE.DoubleSide, depthWrite: false });
  const mesh = new THREE.Mesh(g, m); mesh.renderOrder = 2; scene.add(mesh); return mesh;
}
function updateWedge(mesh, gx, gy, gz, dx, dz, radius, colorHex, opacity) {
  const th0 = Math.atan2(dz, dx), pos = mesh.geometry.attributes.position.array;
  let i = 0;
  for (let s = 0; s < SEG; s++) {
    const a0 = th0 - HALF + (2 * HALF) * (s / SEG);
    const a1 = th0 - HALF + (2 * HALF) * ((s + 1) / SEG);
    pos[i++] = gx; pos[i++] = gy; pos[i++] = gz;
    pos[i++] = gx + radius * Math.cos(a0); pos[i++] = gy; pos[i++] = gz + radius * Math.sin(a0);
    pos[i++] = gx + radius * Math.cos(a1); pos[i++] = gy; pos[i++] = gz + radius * Math.sin(a1);
  }
  mesh.geometry.attributes.position.needsUpdate = true;
  mesh.geometry.computeBoundingSphere();
  mesh.material.color.setHex(colorHex); mesh.material.opacity = opacity;
}

// ── skeleton ────────────────────────────────────────────────────────────────
function buildSkeleton(nJoints, nBones) {
  const col = new THREE.Color(0x2b6cb0);
  const bmat = new THREE.MeshStandardMaterial({ color: col, roughness: 0.55 });
  bmat.emissive = col.clone().multiplyScalar(0.15);
  boneMesh = new THREE.InstancedMesh(new THREE.CylinderGeometry(1, 1, 1, 10), bmat, nBones);
  jointMesh = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 12, 10), bmat, nJoints);
  boneMesh.frustumCulled = jointMesh.frustumCulled = false;
  scene.add(boneMesh); scene.add(jointMesh);
}
function poseSkeleton(joints, bones) {
  for (let j = 0; j < joints.length; j++) {
    dummy.position.set(joints[j][0], joints[j][1], joints[j][2]);
    dummy.quaternion.identity(); dummy.scale.setScalar(JOINT_R); dummy.updateMatrix();
    jointMesh.setMatrixAt(j, dummy.matrix);
  }
  const a = new THREE.Vector3(), b = new THREE.Vector3(), d = new THREE.Vector3(), mid = new THREE.Vector3();
  for (let i = 0; i < bones.length; i++) {
    a.fromArray(joints[bones[i][0]]); b.fromArray(joints[bones[i][1]]);
    d.subVectors(b, a); const len = d.length() || 1e-6; mid.addVectors(a, b).multiplyScalar(0.5);
    dummy.position.copy(mid); dummy.quaternion.setFromUnitVectors(UP, d.normalize());
    dummy.scale.set(BONE_R, len, BONE_R); dummy.updateMatrix();
    boneMesh.setMatrixAt(i, dummy.matrix);
  }
  jointMesh.instanceMatrix.needsUpdate = true; boneMesh.instanceMatrix.needsUpdate = true;
}

// ── load ────────────────────────────────────────────────────────────────────
function disposeSample() {
  [pcObj, grid, boneMesh, jointMesh, arrow, ...wedges].forEach(o => {
    if (!o) return; scene.remove(o); o.geometry?.dispose(); o.material?.dispose();
  });
  pcObj = grid = boneMesh = jointMesh = arrow = null; wedges = [];
}
async function loadSample(id) {
  document.getElementById('loading').classList.remove('hidden');
  const m = await (await fetch(DATA + id + '.json')).json();
  const buf = await (await fetch(DATA + m.pc_file)).arrayBuffer();
  disposeSample(); meta = m;

  const pos = new Float32Array(buf), n = pos.length / 3;
  let ymin = Infinity, ymax = -Infinity;
  for (let i = 0; i < n; i++) { const y = pos[i * 3 + 1]; if (y < ymin) ymin = y; if (y > ymax) ymax = y; }
  const col = new Float32Array(pos.length), lo = new THREE.Color(0x6b7480), hi = new THREE.Color(0xcfd4da), tmp = new THREE.Color();
  for (let i = 0; i < n; i++) {
    tmp.copy(lo).lerp(hi, (pos[i * 3 + 1] - ymin) / (ymax - ymin + 1e-6));
    col[i * 3] = tmp.r; col[i * 3 + 1] = tmp.g; col[i * 3 + 2] = tmp.b;
  }
  const pg = new THREE.BufferGeometry();
  pg.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  pg.setAttribute('color', new THREE.BufferAttribute(col, 3));
  pcObj = new THREE.Points(pg, new THREE.PointsMaterial({ size: 0.02, vertexColors: true, sizeAttenuation: true }));
  pcObj.frustumCulled = false; scene.add(pcObj);

  buildSkeleton(m.n_joints, m.bones.length);
  wedges = [makeWedge(), makeWedge(), makeWedge()];
  const ag = new THREE.ConeGeometry(0.09, 0.28, 12);
  arrow = new THREE.Mesh(ag, new THREE.MeshBasicMaterial({ color: 0x14315e }));
  arrow.renderOrder = 3; scene.add(arrow);

  const span = Math.max(...[0, 2].map(i => m.motion_max[i] - m.motion_min[i]), 6) + 6;
  grid = new THREE.GridHelper(Math.ceil(span), Math.ceil(span), 0xc4cad2, 0xd8dde3);
  grid.position.y = m.floor_y - 0.02; grid.material.transparent = true; grid.material.opacity = 0.3; scene.add(grid);

  total = m.n; frame = 0;
  document.getElementById('timeline').max = String(total - 1);
  frameCamera(); setFrame(0);
  document.getElementById('loading').classList.add('hidden');
}
function frameCamera() {
  const mn = meta.motion_min, mx = meta.motion_max;
  const c = new THREE.Vector3((mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2);
  const size = Math.max(mx[0] - mn[0], mx[2] - mn[2], 2) + 4;   // include wedge reach
  controls.target.copy(c);
  camera.position.copy(c).add(new THREE.Vector3(0.4, 0.9, 1).normalize().multiplyScalar(size * 1.15));
  camera.updateProjectionMatrix(); controls.update();
}

// ── per frame ───────────────────────────────────────────────────────────────
function setFrame(t) {
  frame = Math.max(0, Math.min(total - 1, Math.round(t)));
  poseSkeleton(meta.pose[frame], meta.bones);
  const f = meta.frames[frame], g = f.ground, gy = g[1] + 0.02;
  const dirs = [[f.fwd[0], f.fwd[1]], [-f.right[0], -f.right[1]], [f.right[0], f.right[1]]];
  for (let k = 0; k < 3; k++) {
    const isBest = f.best === k;
    updateWedge(wedges[k], g[0], gy, g[2], dirs[k][0], dirs[k][1],
                Math.max(f.free[k], 0.25), CAT_COL[f.cat[k]], isBest ? 0.62 : 0.28);
    const row = document.querySelector(`.row[data-dir="${k}"]`);
    row.querySelector('.dot').style.background = '#' + CAT_COL[f.cat[k]].toString(16).padStart(6, '0');
    row.querySelector('.lvl').textContent = CAT_TXT[f.cat[k]];
    row.querySelector('.m').textContent = f.free[k].toFixed(1) + ' m';
    row.classList.toggle('best', isBest);
  }
  // best-dir arrow at tip of best wedge
  const bd = dirs[f.best], br = Math.max(f.free[f.best], 0.25);
  const bx = g[0] + bd[0] * br, bz = g[2] + bd[1] * br;
  arrow.position.set(bx, gy + 0.15, bz);
  arrow.quaternion.setFromUnitVectors(UP, new THREE.Vector3(bd[0], 0, bd[1]).normalize());
  document.getElementById('best').innerHTML = 'Best direction: <b>' + DIR_TXT[f.best] + '</b>';
  document.getElementById('timeline').value = String(frame);
  document.getElementById('framelab').textContent = `frame ${frame + 1} / ${total}`;
}

function animate() {
  requestAnimationFrame(animate);
  const dt = clock.getDelta();
  if (playing && meta) { acc += dt * meta.fps * speed; if (acc >= 1) { const s = Math.floor(acc); acc -= s; setFrame((frame + s) % total); } }
  controls.update(); renderer.render(scene, camera);
}

function setPlaying(p) { playing = p; document.getElementById('play').textContent = p ? '❚❚ Pause' : '▶ Play'; }
document.getElementById('play').onclick = () => setPlaying(!playing);
document.getElementById('timeline').oninput = e => { setPlaying(false); setFrame(+e.target.value); };
document.getElementById('speed').onchange = e => { speed = parseFloat(e.target.value); };
document.getElementById('sample').onchange = e => { setPlaying(false); loadSample(e.target.value); };
document.getElementById('reset').onclick = () => frameCamera();
document.getElementById('full').onclick = () => { if (document.fullscreenElement) document.exitFullscreen(); else document.getElementById('app').requestFullscreen?.(); };
addEventListener('resize', () => {
  camera.aspect = stage.clientWidth / stage.clientHeight; camera.updateProjectionMatrix();
  renderer.setSize(stage.clientWidth, stage.clientHeight);
});

(async function () {
  const idx = await (await fetch(DATA + 'index.json')).json();
  const sel = document.getElementById('sample');
  idx.samples.forEach(s => { const o = document.createElement('option'); o.value = s.id; o.textContent = s.label; sel.appendChild(o); });
  await loadSample(idx.samples[0].id);
  setPlaying(true); animate();
})().catch(err => { document.getElementById('loading').textContent = 'Failed to load: ' + err.message; console.error(err); });
