#!/usr/bin/env python3
"""Export spatial (obstacle) scene-awareness clips for the interactive viewer.

Clearance is measured with parse_obstacle.py's own rule and thresholds, but on
the point cloud this viewer actually displays rather than the clean voxelized
scene the published labels were computed on. That keeps the visualization
self-consistent: the colour, the wedge length and the highlighted points all
come from one measurement, so an empty direction reads HIGH/green and a red
wedge always has the obstacle that caused it highlighted at its tip. The
official labels are still read, to report per-scene agreement. Cones are
oriented by the true head facing (from three_points), and samples are chosen so
the person actually walks with a sweeping best-direction.

Per frame: ground point, head-facing forward (horizontal, viewer frame),
category [front,left,right] in {0:LOW,1:MID,2:HIGH}, best in {0:F,1:L,2:R,3:BACK},
and free = measured clearance in metres for [front,left,right].
Outputs into ./spatial:  index.json, <id>.json, <id>.pc.bin (float32 Y-up).
"""
import os, re, json
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE       = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT  = os.path.abspath(os.path.join(HERE, '..', 'ECCV2026', 'qual', 'data'))
TP_DIR     = os.path.abspath(os.path.join(HERE, '..', 'ECCV2026', 'nymeria_egolm_full_v6_2', 'three_points'))
OBST_DIR   = '/mnt/jaewoo4tb/yujinbae/EgoVLM-I/obstacle_labels'
OUT        = os.path.join(HERE, 'spatial')
FPS        = 10

SKELETON_PAIRS = [
    [0,1],[2,0],[3,2],[4,3],[5,4],[6,5],[7,4],[8,7],[9,8],[10,9],[11,4],
    [12,11],[13,12],[14,13],[15,1],[16,15],[17,16],[18,17],[19,1],[20,19],[21,20],[22,21],
]
# Clearance is measured on the cloud we DISPLAY (see free_dirs below), so the
# colour, the wedge length and the highlighted points all come from ONE
# measurement: an empty direction really reads HIGH/green, and a red wedge always
# has the obstacle that caused it highlighted at its tip.
# Scenes are then chosen for FIDELITY to the published labels (how often the
# recomputed level matches the official one), subject to actually showing some
# variety -- a clip where every direction is LOW for all 49 frames demonstrates
# nothing. The numbers below are printed by this script.
# Order matters: the viewer opens on samples[0].
SAMPLES = [
    ('across',    'Walking across a room',        '20230817_s0_brittney_powell_act3_1t2she/0048.pt'),  # 019083  68% agree, 22/117/8 -- only clip with HIGH/green
    ('furniture', 'Walking past furniture',       '20230803_s1_jennifer_sexton_act3_y5o5bu/0157.pt'),  # 014423  91% agree, 42/105/0
    ('bedhall',   'Bedroom into a hallway',       '20230809_s1_laura_smith_act1_iarj4m/0056.pt'),      # 016199  86% agree, 136/11/0
    ('corridor',  'Walking down a corridor',      '20230829_s1_angel_roberts_act2_zv48bm/0044.pt'),    # 024767  57% agree, 110/37/0, 7.2 m walk
]
# Dropped -- removed on request, all-LOW for the whole clip (nothing to illustrate),
# or lowest agreement:
#   kitchen  samantha_lester_act0/0066 removed on request (was 65% | 122/25/0)
#   exitbed  william_wilson_act3/0168  97% agree but 147/0/0   enterbed robert_howard_act4/0084  93% | 147/0/0
#   hallway  alison_riddle_act3/0001   93% | 144/3/0           doorway  kyle_parker_act2/0175    91% | 145/2/0
#   shift    jeffery_bryant_act3/0043  54% agree               navigate alison_riddle_act1/0117  51% agree
#   turning  evelyn_moody_act0/0026    46% agree

R_UP  = np.array([[1,0,0],[0,0,1],[0,-1,0]], float)   # world +Z up -> viewer +Y up
HEAD_I = 6
LVL = {'LOW': 0, 'MID': 1, 'HIGH': 2}
BEST = {'FRONT': 0, 'LEFT': 1, 'RIGHT': 2, 'BACK': 3}


def _np(x): return x.numpy() if torch.is_tensor(x) else np.asarray(x)
def _sq(x): x=_np(x); return x[0] if x.ndim==4 else x
def r4(a):  return np.round(a, 4).tolist()


def kabsch(X, Y):
    cx, cy = X.mean(0), Y.mean(0)
    H = (X - cx).T @ (Y - cy)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cy - R @ cx


def read_official(data_idx):
    """Per-frame (cat_front, cat_left, cat_right, best) from the obstacle label file."""
    txt = open(os.path.join(OBST_DIR, f'{data_idx}.txt')).read()
    out = []
    for fr in re.split(r'\[Frame \d+\]', txt)[1:]:
        m = [re.search(rf'FREE_{k}\s*=\s*(\w+)', fr) for k in ('FRONT', 'LEFT', 'RIGHT')]
        b = re.search(r'BEST_DIR\s*=\s*(\w+)', fr)
        if all(m) and b:
            out.append(([LVL[x.group(1)] for x in m], BEST[b.group(1)]))
    return out


# ── clearance measured on the cloud we actually display ───────────────────────
# parse_obstacle.py's own rule (cone ±60°, |Δheight| < 1 m around the head, 3D
# distance to the nearest point, LOW < 1 m <= MID < 3 m <= HIGH, and max_dist
# when a cone is empty), applied to the exported point cloud instead of the
# clean voxelised scene the labels were computed on. Same definition, same
# thresholds, but now the colour, the wedge length and the highlighted points
# all come from one measurement — so an empty direction really does read HIGH
# (green) and a red wedge always has the obstacle that caused it highlighted.
MAX_DIST, LOW_TH, HIGH_TH, HEIGHT_MARGIN = 5.0, 1.0, 3.0, 1.0
COS_CONE = np.cos(np.radians(60.0))


def free_dirs(pcv, head, fwd_xz):
    """Free distance for FRONT/LEFT/RIGHT/BACK (viewer frame, +Y up)."""
    fx, fz = fwd_xz
    rx, rz = -fz, fx
    order = [(fx, fz), (-rx, -rz), (rx, rz), (-fx, -fz)]      # F, L, R, BACK
    out = [MAX_DIST] * 4
    near = pcv[np.abs(pcv[:, 1] - head[1]) < HEIGHT_MARGIN]
    if len(near):
        rel = near - head
        dist = np.linalg.norm(rel, axis=1)
        m = (dist > 0.05) & (dist < MAX_DIST)
        rel, dist = rel[m], dist[m]
        if len(dist):
            rh = rel.copy(); rh[:, 1] = 0.0
            rn = rh / (np.linalg.norm(rh, axis=1, keepdims=True) + 1e-8)
            for i, (dx, dz) in enumerate(order):
                sel = (rn @ np.array([dx, 0.0, dz])) > COS_CONE
                if sel.any():
                    out[i] = float(dist[sel].min())
    return out


def cat_free(d):
    return 0 if d < LOW_TH else (1 if d < HIGH_TH else 2)


# Reconstruction noise leaves thousands of tiny floating specks (e.g. the corridor
# clip has ~3.2k connected components smaller than 5 points). Since clearance is
# the distance to the NEAREST point, a single 4-5 point speck was enough to make a
# whole direction read LOW/red. Link points within CLUSTER_R and drop any connected
# component smaller than MIN_CLUSTER, so both the measurement and what is drawn use
# real surfaces.
# The radius is deliberately generous: a speck lying against real geometry is
# absorbed into it (and is harmless anyway, since the surface sets the same
# distance), while a speck floating alone in an otherwise empty cone -- the case
# that produced a spurious red/orange -- stays its own small component and is
# dropped. A tighter radius (0.12) instead chewed into genuinely sparse scenes:
# agreement with the published labels fell from 74% to 64%, whereas this setting
# keeps 95-99% of the points and 69%.
CLUSTER_R, MIN_CLUSTER = 0.30, 20


def drop_speckles(pcv):
    if len(pcv) < MIN_CLUSTER:
        return pcv, 0
    tree = cKDTree(pcv)
    pairs = tree.query_pairs(CLUSTER_R, output_type='ndarray')
    n = len(pcv)
    if len(pairs) == 0:
        return pcv[:0], n
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    keep = np.bincount(lab)[lab] >= MIN_CLUSTER
    return pcv[keep], int((~keep).sum())


def main():
    os.makedirs(OUT, exist_ok=True)
    index = []
    for sid, label, rel in SAMPLES:
        data_idx = str(torch.load(os.path.join(DATA_ROOT, 'ours_withGRPO', rel),
                                  map_location='cpu', weights_only=False).get('data_idx'))
        d = torch.load(os.path.join(DATA_ROOT, 'ours_noGRPO', rel), map_location='cpu', weights_only=False)
        gp = _sq(d['gt_past']).astype(float)
        pc = _sq(np.asarray(d['pc'], float))
        center = gp[:, 0].mean(0)
        Wp = lambda P: (P - center) @ R_UP.T          # positions
        Wd = lambda D: D @ R_UP.T                      # directions
        gpv, pcv = Wp(gp), Wp(pc)
        floor_y = float(np.percentile(gpv[:, :, 1], 2))

        # head facing from three_points, aligned to this sample's world frame
        tp = np.load(os.path.join(TP_DIR, f'{data_idx}.npy'))    # (T,3,4,4)
        T = min(tp.shape[0], gp.shape[0])
        R_tp, _ = kabsch(tp[:T, 0, :3, 3], gp[:T, HEAD_I])       # head positions align
        fwd_world = (tp[:T, 0, :3, 2] @ R_tp.T)                  # head Z-axis = facing
        # resolve sign so facing points along travel
        vel = np.diff(gp[:T, HEAD_I], axis=0)
        if np.sum(fwd_world[:-1, :2] * vel[:, :2]) < 0:
            fwd_world = -fwd_world
        fwd_v = Wd(fwd_world); fwd_v[:, 1] = 0
        fwd_v /= (np.linalg.norm(fwd_v, axis=1, keepdims=True) + 1e-9)

        official = read_official(data_idx)
        T = min(T, len(official), gpv.shape[0])

        # keep only points within the height of the human (floor .. just above the head),
        # dropping the ceiling / high clutter so the navigable obstacles are what's shown.
        # Filter BEFORE measuring clearance so the numbers describe what is on screen.
        head_top = float(gpv[:, HEAD_I, 1].max()) + 0.25
        pcv = pcv[(pcv[:, 1] >= floor_y - 0.15) & (pcv[:, 1] <= head_top)]
        n_before = len(pcv)
        pcv, n_dropped = drop_speckles(pcv)   # isolated specks must not set the clearance

        frames = []
        agree = 0
        for t in range(T):
            head = gpv[t, HEAD_I]
            fd = free_dirs(pcv, head, (float(fwd_v[t, 0]), float(fwd_v[t, 2])))
            cat = [cat_free(fd[0]), cat_free(fd[1]), cat_free(fd[2])]
            best = int(np.argmax(fd))                     # F/L/R/BACK, same order as BEST
            ocat, _ = official[t]
            agree += sum(int(cat[k] == ocat[k]) for k in range(3))
            frames.append(dict(
                ground=r4([float(head[0]), floor_y, float(head[2])]),
                fwd=r4([float(fwd_v[t, 0]), float(fwd_v[t, 2])]),
                cat=cat, best=best, free=r4(fd[:3]),
            ))
        pcf = pcv.astype(np.float32)
        pcf.tofile(os.path.join(OUT, f'{sid}.pc.bin'))
        allj = gpv[:T].reshape(-1, 3)
        meta = dict(id=sid, label=label, fps=FPS, n=T, n_joints=int(gpv.shape[1]),
                    bones=SKELETON_PAIRS, pose=r4(gpv[:T]),
                    pc_file=f'{sid}.pc.bin', pc_count=int(pcf.shape[0]), floor_y=round(floor_y, 4),
                    motion_min=r4(allj.min(0)), motion_max=r4(allj.max(0)), frames=frames)
        json.dump(meta, open(os.path.join(OUT, f'{sid}.json'), 'w'), separators=(',', ':'))
        bd = {k: sum(1 for f in frames if f['best'] == v) for k, v in BEST.items()}
        lv = [sum(1 for f in frames for c in f['cat'] if c == l) for l in (0, 1, 2)]
        index.append(dict(id=sid, label=label, file=f'{sid}.json'))
        print(f'[{sid}] {label}: {T} frames, disp={np.linalg.norm(gpv[T-1,0]-gpv[0,0]):.1f}m, '
              f'LOW/MID/HIGH={lv[0]}/{lv[1]}/{lv[2]}, agree-with-official={agree/(3*T)*100:.0f}%, '
              f'speckles-dropped={n_dropped}/{n_before}, best-dir {bd}')

    json.dump(dict(samples=index), open(os.path.join(OUT, 'index.json'), 'w'), indent=1)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
