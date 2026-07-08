#!/usr/bin/env python3
"""Export spatial (obstacle) scene-awareness clips for the interactive sector
viewer in the 'Spatial-Semantic Scene Awareness Data' section.

Reproduces parse_obstacle.py's directional cone-casting ON the same 50k-point
cloud + skeleton we render (so the wedges match the shown scene): per frame,
free distance in the FRONT / LEFT / RIGHT cones (±60deg, points within ±1 m of
head height), categorised LOW<1 m / MID 1-3 m / HIGH>=3 m, plus BEST_DIR.

Samples are the *tracked past* segment of clips whose obstacle labels show varied
clearance (people navigating open space, not boxed in).

Outputs into ./spatial:  index.json, <id>.json, <id>.pc.bin (float32 Y-up).
"""
import os, json
import numpy as np
import torch

HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.abspath(os.path.join(HERE, '..', 'ECCV2026', 'qual', 'data'))
OUT       = os.path.join(HERE, 'spatial')
FPS       = 10

SKELETON_PAIRS = [
    [0,1],[2,0],[3,2],[4,3],[5,4],[6,5],[7,4],[8,7],[9,8],[10,9],[11,4],
    [12,11],[13,12],[14,13],[15,1],[16,15],[17,16],[18,17],[19,1],[20,19],[21,20],[22,21],
]
# clips with varied clearance in the tracked-past window
SAMPLES = [
    ('hallway', 'Hallway — open path ahead',      '20230829_s1_angel_roberts_act2_zv48bm/0040.pt'),
    ('choose',  'Living room — choosing a path',   '20230817_s0_brittney_powell_act3_1t2she/0044.pt'),
    ('doorway', 'Living room — toward the doorway', '20230817_s0_brittney_powell_act3_1t2she/0048.pt'),
]

R_UP  = np.array([[1,0,0],[0,0,1],[0,-1,0]], float)   # world +Z up -> viewer +Y up
UP    = np.array([0,1,0.])
CONE  = np.cos(np.radians(60.0))
MAXD  = 5.0
HEAD_I, RSH_I, LSH_I = 6, 7, 11


def _np(x): return x.numpy() if torch.is_tensor(x) else np.asarray(x)
def _sq(x): x=_np(x); return x[0] if x.ndim==4 else x
def cat(x):  return 0 if x < 1.0 else (1 if x < 3.0 else 2)     # LOW / MID / HIGH


def clearance(pc, head, fwd):
    fwd = fwd.copy(); fwd[1] = 0; fwd /= np.linalg.norm(fwd) + 1e-9
    right = np.cross(fwd, UP); right /= np.linalg.norm(right) + 1e-9
    band = np.abs(pc[:, 1] - head[1]) < 1.0                     # ±1 m around head height
    rel = pc[band] - head; rel[:, 1] = 0.0
    dist = np.linalg.norm(rel, axis=1)
    keep = (dist > 0.3) & (dist < MAXD)
    ru, dd = rel[keep] / dist[keep, None], dist[keep]
    dirs = {'front': fwd, 'left': -right, 'right': right}
    free = {}
    for k, v in dirs.items():
        inc = (ru @ v) > CONE
        free[k] = float(dd[inc].min()) if inc.any() else MAXD
    best = max(free, key=free.get)
    return fwd, right, free, best


def r4(a): return np.round(a, 4).tolist()


def main():
    os.makedirs(OUT, exist_ok=True)
    index = []
    for sid, label, rel in SAMPLES:
        d = torch.load(os.path.join(DATA_ROOT, 'ours_noGRPO', rel),
                       map_location='cpu', weights_only=False)
        gp = _sq(d['gt_past']).astype(float)          # tracked-past segment
        pc = _sq(np.asarray(d['pc'], float))
        center = gp[:, 0].mean(0)
        W = lambda P: (P - center) @ R_UP.T
        gp, pc = W(gp), W(pc)
        floor_y = float(np.percentile(gp[:, :, 1], 2))            # ~feet level

        frames = []
        for t in range(gp.shape[0]):
            p = gp[t]; head = p[HEAD_I]
            fwd = np.cross(p[LSH_I] - p[RSH_I], UP)
            fwd, right, free, best = clearance(pc, head, fwd)
            ground = [float(head[0]), floor_y, float(head[2])]
            frames.append(dict(
                ground=r4(ground),
                fwd=r4([fwd[0], fwd[2]]), right=r4([right[0], right[2]]),  # horizontal (x,z)
                free=[round(free['front'],3), round(free['left'],3), round(free['right'],3)],
                cat=[cat(free['front']), cat(free['left']), cat(free['right'])],
                best={'front':0,'left':1,'right':2}[best],
            ))
        pcf = pc.astype(np.float32)
        pcf.tofile(os.path.join(OUT, f'{sid}.pc.bin'))
        allj = gp.reshape(-1, 3)
        meta = dict(id=sid, label=label, fps=FPS, n=int(gp.shape[0]),
                    n_joints=int(gp.shape[1]), bones=SKELETON_PAIRS,
                    pose=r4(gp), pc_file=f'{sid}.pc.bin', pc_count=int(pcf.shape[0]),
                    floor_y=round(floor_y,4),
                    motion_min=r4(allj.min(0)), motion_max=r4(allj.max(0)),
                    frames=frames)
        json.dump(meta, open(os.path.join(OUT, f'{sid}.json'), 'w'), separators=(',', ':'))
        hi = sum(1 for f in frames for c in f['cat'] if c == 2)
        index.append(dict(id=sid, label=label, file=f'{sid}.json'))
        print(f'[{sid}] {label}: {gp.shape[0]} frames, HIGH-cells={hi}, '
              f'json={os.path.getsize(os.path.join(OUT, sid+".json"))//1024}KB')

    json.dump(dict(samples=index), open(os.path.join(OUT, 'index.json'), 'w'), indent=1)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
