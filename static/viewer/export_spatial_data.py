#!/usr/bin/env python3
"""Export spatial (obstacle) scene-awareness clips for the interactive viewer.

Accuracy fix: instead of recomputing clearance on the noisy 50k cloud, use the
OFFICIAL obstacle labels (parse_obstacle.py output, computed on the clean
voxelized scene) for the per-frame LOW/MID/HIGH clearance and BEST_DIR, and
orient the cones by the true head facing (from three_points). Samples are chosen
so the person actually walks (global motion) with a sweeping best-direction.

Per frame: ground point, head-facing forward (horizontal, viewer frame),
category [front,left,right] in {0:LOW,1:MID,2:HIGH}, best in {0:F,1:L,2:R,3:BACK}.
Outputs into ./spatial:  index.json, <id>.json, <id>.pc.bin (float32 Y-up).
"""
import os, re, json
import numpy as np
import torch

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
# walking clips with a sweeping best-direction (distinct scenes)
# Mostly WALKING scenes (global motion, disp > 1.6 m) with clearance variety, so
# you watch the clearance/obstacles change as the person navigates. Diverse
# recordings; brittney/0048 also has an open (HIGH) direction.
# The LOW/MID/HIGH colour comes from the official labels (clean voxelised scene)
# while the highlighted points are the nearest cluster in the displayed, noisy 50k
# cloud, so the two can disagree — giving highlights that are "far but red" or
# "close but orange". Scenes are therefore screened for VISUAL CONSISTENCY: per
# frame and direction, measure the nearest highlighted distance the way spatial.js
# does and count violations (LOW but >1.2 m, MID but <0.5 m, HIGH but <1.0 m), and
# require the median LOW distance to be genuinely closer than the median MID.
# Kept: violations <=7.5% AND a positive LOW->MID margin.
SAMPLES = [
    ('furniture', 'Walking past furniture',       '20230803_s1_jennifer_sexton_act3_y5o5bu/0157.pt'),  # 014423  0.7% viol, margin +0.42
    ('enterbed',  'Entering a bedroom',           '20230803_s0_robert_howard_act4_e29s94/0084.pt'),    # 013788  2.0% viol, margin +0.08
    ('exitbed',   'Exiting a bedroom',            '20230821_s1_william_wilson_act3_gnf0bz/0168.pt'),   # 020842  3.4% viol, margin +0.01
    ('bedhall',   'Bedroom into a hallway',       '20230809_s1_laura_smith_act1_iarj4m/0056.pt'),      # 016199  6.1% viol, margin +0.05
    ('corridor',  'Walking down a corridor',      '20230829_s1_angel_roberts_act2_zv48bm/0044.pt'),    # 024767  7.5% viol, margin +0.19
]
# Dropped for inconsistent colouring (violation % | LOW->MID margin):
#   navigate  alison_riddle_act1/0117   33.3% | -0.06   shift    jeffery_bryant_act3/0043  22.4% | +0.26
#   kitchen   samantha_lester_act0/0066 21.1% | -0.21   across   brittney_powell_act3/0048 15.0% | +1.29
#   turning   evelyn_moody_act0/0026     8.8% | -0.11   doorway  kyle_parker_act2/0175      5.4% | -0.03
#   hallway   alison_riddle_act3/0001    3.4% | -0.03
# NB: `across` was the only clip with a HIGH/green direction (only 3 clips in the
# whole dataset have one, all in brittney_powell), but 14% of its highlights were
# close-yet-orange/green, so the remaining set is LOW/MID only.

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

        frames = []
        for t in range(T):
            head = gpv[t, HEAD_I]
            cat, best = official[t]
            frames.append(dict(
                ground=r4([float(head[0]), floor_y, float(head[2])]),
                fwd=r4([float(fwd_v[t, 0]), float(fwd_v[t, 2])]),
                cat=cat, best=best,
            ))
        # keep only points within the height of the human (floor .. just above the head),
        # dropping the ceiling / high clutter so the navigable obstacles are what's shown
        head_top = float(gpv[:, HEAD_I, 1].max()) + 0.25
        pcv = pcv[(pcv[:, 1] >= floor_y - 0.15) & (pcv[:, 1] <= head_top)]
        pcf = pcv.astype(np.float32)
        pcf.tofile(os.path.join(OUT, f'{sid}.pc.bin'))
        allj = gpv[:T].reshape(-1, 3)
        meta = dict(id=sid, label=label, fps=FPS, n=T, n_joints=int(gpv.shape[1]),
                    bones=SKELETON_PAIRS, pose=r4(gpv[:T]),
                    pc_file=f'{sid}.pc.bin', pc_count=int(pcf.shape[0]), floor_y=round(floor_y, 4),
                    motion_min=r4(allj.min(0)), motion_max=r4(allj.max(0)), frames=frames)
        json.dump(meta, open(os.path.join(OUT, f'{sid}.json'), 'w'), separators=(',', ':'))
        bd = {k: sum(1 for f in frames if f['best'] == v) for k, v in BEST.items()}
        index.append(dict(id=sid, label=label, file=f'{sid}.json'))
        print(f'[{sid}] {label}: {T} frames, disp={np.linalg.norm(gpv[T-1,0]-gpv[0,0]):.1f}m, best-dir {bd}')

    json.dump(dict(samples=index), open(os.path.join(OUT, 'index.json'), 'w'), indent=1)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
