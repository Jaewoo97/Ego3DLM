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
# Baselines are trimmed to EgoLM only (per request); the other rows are our own
# variants/ablations. FICTION and UniEgoMotion .pt dumps still exist on disk but
# are intentionally not published to the viewer.
METHODS = [
    ('ours_withGRPO', 'Ours (Ego3DLM)',  '#33b864'),
    ('ours_noGRPO',   'Ours (no GRPO)',  '#3b82f6'),
    ('ours_no3d',     'Ours (no 3D)',    '#a855f7'),
    ('egolm',         'EgoLM',           '#ef4444'),
]
GT_COLOR = '#8b9198'

# 22 bones over the 23-joint kinematic tree (from qual/utils.py skeleton_pairs)
SKELETON_PAIRS = [
    [0,1],[2,0],[3,2],[4,3],[5,4],[6,5],[7,4],[8,7],[9,8],[10,9],[11,4],
    [12,11],[13,12],[14,13],[15,1],[16,15],[17,16],[18,17],[19,1],[20,19],[21,20],[22,21],
]

# samples to publish: (id, human label, method-relative path)
# Chosen so ours_withGRPO has the lowest future ADE across all 6 methods AND the
# GT motion is natural (near-upright body, feet plant rather than glide) — samples
# with a large torso lean look weird, so they're excluded.
SAMPLES = [
    ('bedroom', 'Bedroom → hallway (paper figure)',       '20230821_s1_william_wilson_act3_gnf0bz/0050.pt'),
    ('kitchen', 'Living area — talking then walking',     '20230822_s0_kyle_parker_act2_y3l7lv/0062.pt'),
    ('living',  'Living room — walking and turning',       '20230817_s0_brittney_powell_act3_1t2she/0050.pt'),
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
                ade=float(ades[best]), cot=cot, data_idx=str(d.get('data_idx')))


def align_seg(seg_info, seg_ref):
    """Kabsch aligning ONE segment (past OR future) to the reference's segment.
    Some methods (e.g. ours_withGRPO) canonicalise each segment separately — both
    gt_past[0] and gt_future[0] sit at the origin — so past and future must be
    aligned on their own. A single past+future transform can't fit two
    independently-centered segments and would misplace the prediction (~1.7 m)."""
    T = min(seg_info.shape[0], seg_ref.shape[0])
    return kabsch(seg_info[:T].reshape(-1, 3), seg_ref[:T].reshape(-1, 3))


def r3(a):  # round -> nested python lists (mm precision)
    return np.round(a, 4).tolist()


TEXTS_DIR = os.path.abspath(os.path.join(HERE, '..', 'ECCV2026', 'nymeria_egolm_full_v6_2', 'texts'))
TP_DIR    = os.path.abspath(os.path.join(HERE, '..', 'ECCV2026', 'nymeria_egolm_full_v6_2', 'three_points'))
HEAD_I    = 6                       # head joint (= TP_JOINTS[0] in the viewer)


def gt_narration(data_idx):
    """Ground-truth motion narration for one segment (texts/<data_idx>.txt).
    The dataset annotates EACH segment with its own full sentence; the past and
    the future are consecutive segments, so each has its own complete narration
    (they must NOT be treated as two halves of one sentence)."""
    p = os.path.join(TEXTS_DIR, f'{data_idx}.txt')
    if data_idx and os.path.exists(p):
        return open(p, encoding='utf-8').read().split('#')[0].strip()
    return ''


def future_data_idx(ref_gf, past_idx):
    """data_idx of the FUTURE segment = the next segment (past_idx+1), verified by
    Kabsch-matching gt_future's head to that segment's three-point head."""
    try:
        cand = f'{int(past_idx) + 1:06d}'
    except (TypeError, ValueError):
        return None
    p = os.path.join(TP_DIR, f'{cand}.npy')
    if not os.path.exists(p):
        return None
    tp = np.load(p)[:, 0, :3, 3]
    T = min(tp.shape[0], ref_gf.shape[0])
    R, t = kabsch(tp[:T], ref_gf[:T, HEAD_I])
    resid = np.linalg.norm((tp[:T] @ R.T + t) - ref_gf[:T, HEAD_I], axis=1).mean()
    return cand if resid < 0.1 else None


def head_forward(ref_gp, data_idx):
    """Per-past-frame egocentric camera optical axis (in the reference raw world
    frame) from the three-point head pose. The head's local **X-axis (column 0)**
    is the camera-forward / look direction — it is ~horizontal and tracks travel
    (cos≈0.93); column 2 is the head's up axis (points up), and column 1 is
    lateral. Kabsch-aligned to the GT head trajectory, sign-resolved by travel so
    it points forward, lightly smoothed so the camera doesn't jitter. The natural
    ~10° downward pitch is kept so the scene view matches what the ego camera sees."""
    p = os.path.join(TP_DIR, f'{data_idx}.npy')
    if not (data_idx and os.path.exists(p)):
        return None
    tp = np.load(p)                              # (T,3,4,4): point 0 = head
    T = min(tp.shape[0], ref_gp.shape[0])
    R_tp, _ = kabsch(tp[:T, 0, :3, 3], ref_gp[:T, HEAD_I])
    fwd = tp[:T, 0, :3, 0] @ R_tp.T              # head X-axis = camera optical axis
    vel = np.diff(ref_gp[:T, HEAD_I], axis=0)
    if np.sum(fwd[:-1, :2] * vel[:, :2]) < 0:    # point it along travel
        fwd = -fwd
    k = 5                                         # box-smooth to steady the camera
    pad = np.pad(fwd, ((k // 2, k // 2), (0, 0)), mode='edge')
    return np.stack([pad[i:i + k].mean(0) for i in range(T)])


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

        # The past & future are separate narrated segments with a real spatial
        # seam; bridge it with a few eased frames so past->future playback is
        # spatiotemporally continuous (natural walking speed, no teleport).
        A, Bf = gt_past[-1], gt_future[0]
        step = float(np.median(np.linalg.norm(np.diff(gt_future[:, 0], axis=0), axis=1))) + 1e-6
        n_bridge = int(np.clip(round(np.linalg.norm(Bf[0] - A[0]) / step), 2, 6))
        ss = lambda x: x * x * (3 - 2 * x)                        # smoothstep ease
        gt_bridge = np.stack([A * (1 - ss((i + 1) / (n_bridge + 1))) +
                              Bf * ss((i + 1) / (n_bridge + 1)) for i in range(n_bridge)])

        # egocentric head camera: per-past-frame eye position (GT head, viewer
        # frame) + facing (three-point head Z-axis -> viewer frame). Lets the
        # scene panel be rendered from the wearer's viewpoint, so it lines up
        # with the egocentric video.
        fwd_raw = head_forward(ref['gp'], per['ours_withGRPO']['data_idx'])
        head_cam = None
        if fwd_raw is not None:
            Tc = min(fwd_raw.shape[0], gt_past.shape[0])
            fwd_v = fwd_raw @ R_UP.T                       # optical axis -> viewer frame
            fwd_v /= (np.linalg.norm(fwd_v, axis=1, keepdims=True) + 1e-9)  # (pitch kept)
            head_cam = [dict(p=r3(gt_past[t, HEAD_I]),
                             f=r3(fwd_v[t])) for t in range(Tc)]

        methods_out = {}
        for m, mlabel, color in METHODS:
            info = per[m]
            Rp, tp = align_seg(info['gp'], ref['gp'])   # align past segment
            Rf, tf = align_seg(info['gf'], ref['gf'])   # align future segment on its own
            pp = W(apply_rt(info['pp'], Rp, tp))
            pf = W(apply_rt(info['modes'][info['best']], Rf, tf))
            entry = dict(label=mlabel, color=color, ade=round(info['ade'], 3),
                         pred_past=r3(pp), pred_future=r3(pf), cot=info['cot'])
            if m == 'ours_withGRPO':
                entry['pred_future_modes'] = [r3(W(apply_rt(mm, Rf, tf))) for mm in info['modes']]
            methods_out[m] = entry

        # point cloud -> viewer frame, float32 bin
        pc = W(ref['pc']).astype(np.float32)
        with open(os.path.join(OUT_DIR, f'{sid}.pc.bin'), 'wb') as fh:
            fh.write(pc.tobytes())

        allg = np.concatenate([gt_past.reshape(-1, 3), gt_future.reshape(-1, 3)])
        meta = dict(
            id=sid, label=label,
            fps=FPS, n_past=int(gt_past.shape[0]), n_future=int(gt_future.shape[0]),
            n_bridge=int(n_bridge),
            n_joints=int(gt_past.shape[1]), bones=SKELETON_PAIRS,
            gt=dict(label='Ground truth', color=GT_COLOR,
                    past=r3(gt_past), bridge=r3(gt_bridge), future=r3(gt_future)),
            gt_text=gt_narration(per['ours_withGRPO']['data_idx']),
            gt_text_future=gt_narration(future_data_idx(ref['gf'], per['ours_withGRPO']['data_idx'])),
            head_cam=head_cam,
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
