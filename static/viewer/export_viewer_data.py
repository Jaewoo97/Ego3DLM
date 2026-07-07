#!/usr/bin/env python3
"""Export qual/ .pt samples to web assets for the interactive Three.js viewer.

For each sample we load all methods, Kabsch-align every method's GT+prediction
onto the reference method's (ours_noGRPO) world frame (identical to
qual_viz_compare2.py), borrow the reference's 50k-point scene cloud, rotate the
world so up (+Z) -> +Y, and recenter on the person's trajectory.

Outputs (into ./data):
  index.json              list of samples
  <id>.json               skeletons (GT + per method), bones, colors, cot text
  <id>.pc.bin             float32 [N,3] point cloud (Y-up, centered)
"""
import os, json, struct
import numpy as np
import torch

HERE       = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT  = os.path.abspath(os.path.join(HERE, '..', 'ECCV2026', 'qual', 'data'))
OUT_DIR    = os.path.join(HERE, 'data')
REFERENCE  = 'ours_noGRPO'          # holds the real point cloud
FPS        = 10

# method key -> (label, hex color).  GT handled separately.
METHODS = [
    ('ours_withGRPO', 'Ours (Ego3DLM)',  '#33b864'),
    ('ours_noGRPO',   'Ours (no GRPO)',  '#3b82f6'),
    ('ours_no3d',     'Ours (no 3D)',    '#a855f7'),
    ('fiction',       'FICTION',         '#f59e0b'),
    ('uniegomotion',  'UniEgoMotion',    '#ec4899'),
    ('egolm',         'EgoLM',           '#ef4444'),
]
GT_COLOR = '#8b9198'

# 22 bones over the 23-joint kinematic tree (from qual/utils.py skeleton_pairs)
SKELETON_PAIRS = [
    [0,1],[2,0],[3,2],[4,3],[5,4],[6,5],[7,4],[8,7],[9,8],[10,9],[11,4],
    [12,11],[13,12],[14,13],[15,1],[16,15],[17,16],[18,17],[19,1],[20,19],[21,20],[22,21],
]

# samples to publish: (id, human label, method-relative path)
SAMPLES = [
    ('kitchen',  'Kitchen — walking & reaching a chair',        '20230822_s0_kyle_parker_act2_y3l7lv/0098.pt'),
    ('bedroom',  'Bedroom — walking out toward the door',       '20230815_s0_samantha_lester_act0_513kae/0068.pt'),
    ('tracking', 'Living room — turning / tracking',            '20230817_s0_brittney_powell_act3_1t2she/0041.pt'),
]

# world (+Z up) -> viewer (+Y up):  (x,y,z) -> (x, z, -y)
R_UP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)


def _np(x):
    return x.numpy() if torch.is_tensor(x) else np.asarray(x)


def _sq(x):
    return x[0] if x.ndim == 4 else x


def kabsch(X, Y):
    cx, cy = X.mean(0), Y.mean(0)
    H = (X - cx).T @ (Y - cy)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cy - R @ cx


def apply_rt(J, R, t):
    return (J.reshape(-1, 3) @ R.T + t).reshape(J.shape)


def load_method(folder, rel):
    d = torch.load(os.path.join(DATA_ROOT, folder, rel),
                   map_location='cpu', weights_only=False)
    gp, gf, pp = _sq(_np(d['gt_past'])), _sq(_np(d['gt_future'])), _sq(_np(d['pred_past']))
    preds = d['pred_future']
    if isinstance(preds, list):
        modes = [_sq(_np(p)) for p in preds]
    else:
        arr = _np(preds)
        if arr.ndim == 4 and arr.shape[0] > 1 and arr.shape[1] == gf.shape[0]:
            modes = [arr[i] for i in range(arr.shape[0])]   # (M,T,J,3) -> M modes
        else:
            modes = [_sq(arr)]
    T = min(gf.shape[0], min(m.shape[0] for m in modes))
    ades = [np.mean(np.linalg.norm(m[:T] - gf[:T], axis=-1)) for m in modes]
    best = int(np.argmin(ades))
    pc = d.get('pc')
    pc = _sq(_np(pc)) if not (pc is None or isinstance(pc, str)) else None
    if pc is not None and (pc.ndim != 2 or pc.shape[1] != 3):
        pc = None
    cot = d.get('cot')
    cot = cot[best] if isinstance(cot, list) and cot else (cot if isinstance(cot, str) else '')
    return dict(gp=gp, gf=gf, pp=pp, modes=modes, best=best, pc=pc,
                ade=float(ades[best]), cot=cot)


def align(info, ref):
    Tp = min(info['gp'].shape[0], ref['gp'].shape[0])
    Tf = min(info['gf'].shape[0], ref['gf'].shape[0])
    X = np.concatenate([info['gp'][:Tp].reshape(-1, 3), info['gf'][:Tf].reshape(-1, 3)])
    Y = np.concatenate([ref['gp'][:Tp].reshape(-1, 3),  ref['gf'][:Tf].reshape(-1, 3)])
    R, t = kabsch(X, Y)
    return R, t


def r3(a):  # round -> nested python lists (mm precision)
    return np.round(a, 4).tolist()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    index = []
    for sid, label, rel in SAMPLES:
        per = {m: load_method(m, rel) for m, _, _ in METHODS}
        ref = per[REFERENCE]
        if ref['pc'] is None:
            raise RuntimeError(f'reference {REFERENCE} has no point cloud for {rel}')

        # world transform: center on person's trajectory, rotate +Z -> +Y
        center = np.concatenate([ref['gp'][:, 0], ref['gf'][:, 0]]).mean(0)

        def W(P):  # to viewer frame
            return apply_rt(P - center, R_UP, np.zeros(3))

        # GT (reference frame already)
        gt_past, gt_future = W(ref['gp']), W(ref['gf'])

        methods_out = {}
        for m, mlabel, color in METHODS:
            info = per[m]
            R, t = align(info, ref)
            pp = W(apply_rt(info['pp'], R, t))
            pf = W(apply_rt(info['modes'][info['best']], R, t))
            entry = dict(label=mlabel, color=color, ade=round(info['ade'], 3),
                         pred_past=r3(pp), pred_future=r3(pf), cot=info['cot'])
            if m == 'ours_withGRPO':
                entry['pred_future_modes'] = [r3(W(apply_rt(mm, R, t))) for mm in info['modes']]
            methods_out[m] = entry

        # point cloud -> viewer frame, float32 bin
        pc = W(ref['pc']).astype(np.float32)
        with open(os.path.join(OUT_DIR, f'{sid}.pc.bin'), 'wb') as fh:
            fh.write(pc.tobytes())

        allg = np.concatenate([gt_past.reshape(-1, 3), gt_future.reshape(-1, 3)])
        meta = dict(
            id=sid, label=label,
            fps=FPS, n_past=int(gt_past.shape[0]), n_future=int(gt_future.shape[0]),
            n_joints=int(gt_past.shape[1]), bones=SKELETON_PAIRS,
            gt=dict(label='Ground truth', color=GT_COLOR,
                    past=r3(gt_past), future=r3(gt_future)),
            methods=methods_out,
            pc_file=f'{sid}.pc.bin', pc_count=int(pc.shape[0]),
            person_center=[0, 0, 0],
            focus=r3(np.concatenate([gt_past[:, 0], gt_future[:, 0]]).mean(0)),
            motion_min=r3(allg.min(0)), motion_max=r3(allg.max(0)),
        )
        with open(os.path.join(OUT_DIR, f'{sid}.json'), 'w') as fh:
            json.dump(meta, fh, separators=(',', ':'))
        index.append(dict(id=sid, label=label, file=f'{sid}.json',
                          pc_kb=round(os.path.getsize(os.path.join(OUT_DIR, f'{sid}.pc.bin')) / 1024)))
        print(f'[{sid}] pc={pc.shape[0]} json={os.path.getsize(os.path.join(OUT_DIR, sid+".json"))//1024}KB '
              f'best-ADE ours={methods_out["ours_withGRPO"]["ade"]}')

    with open(os.path.join(OUT_DIR, 'index.json'), 'w') as fh:
        json.dump(dict(samples=index, methods=[m for m, _, _ in METHODS]), fh, indent=1)
    print('wrote', OUT_DIR)


if __name__ == '__main__':
    main()
