#!/usr/bin/env python3
"""Export semantic scene-awareness QA (+ ego frames) to web assets for the
interactive QA browser in the 'Spatial-Semantic Scene Awareness Data' section.

Source: /ssd4tb/egovlm/3D_qa/qa_dataset.final.filtered.total.json  (ScanQA-style
per-frame QA over Nymeria ego video), images under
/ssd4tb/egovlm/parsed_data_vqa/<scene>/rgb_img_undistort/frame_XXXX.jpg.

Outputs into ./semantic:
  semantic.json           scenes -> frames -> QA items
  img/<sid>/frame_*.jpg   compressed ego frames (video poster / fallback)
  vid/<sid>/frame_*.mp4   short ego VIDEO clip centred on each QA frame
"""
import os, json, collections, subprocess, shutil
from PIL import Image

HERE     = os.path.dirname(os.path.abspath(__file__))
QA_JSON  = '/ssd4tb/egovlm/3D_qa/qa_dataset.final.filtered.total.json'
IMROOT   = '/ssd4tb/egovlm/parsed_data_vqa'
OUT      = os.path.join(HERE, 'semantic')
IMG_OUT  = os.path.join(OUT, 'img')
VID_OUT  = os.path.join(OUT, 'vid')
FRAMES_PER_SCENE = 12          # richest frames per scene
IMG_SIZE = 448
IMG_Q    = 82
# short looping clip around each QA frame, so the QA is shown against live ego
# video rather than a single still
# The vqa frames are sampled ~1 s apart (measured: their consecutive-frame motion
# exceeds even a 3 s gap in true 10 fps footage), so playing them at 10 fps ran
# ~10x too fast. Play them slowly instead.
VID_HALF = 7                   # frames each side of the QA frame (15 frames, ~5 s)
VID_FPS  = 3
VID_SIZE = 448
VID_CRF  = 27

# 6 diverse scenes (distinct wearers). id -> (scene_name, label)
SCENES = [
    ('barbara',  '20230928_s1_barbara_sandoval_act4_pr1raw', 'Barbara — kitchen'),
    ('samuel',   '20230929_s0_samuel_campos_act4_5kbgah',    'Samuel — living room'),
    ('grace',    '20230928_s0_grace_randolph_act3_090k3i',   'Grace — kitchen'),
    ('alan',     '20230929_s1_alan_burns_act2_hap6xx',       'Alan — dining area'),
    ('zachary',  '20230927_s0_zachary_price_act1_5uyac6',    'Zachary — bedroom'),
    ('megan',    '20230926_s1_megan_mejia_act4_qfzkks',      'Megan — kitchen'),
]


def encode_clip(scene_dir, center_idx, n_avail, dst):
    """Encode a short mp4 centred on `center_idx` from the scene's jpg sequence."""
    start = max(0, center_idx - VID_HALF)
    count = min(VID_HALF * 2 + 1, n_avail - start)
    if count < 4:
        return False
    cmd = ['ffmpeg', '-y', '-loglevel', 'error',
           '-start_number', str(start), '-framerate', str(VID_FPS),
           '-i', os.path.join(scene_dir, 'frame_%04d.jpg'),
           '-frames:v', str(count),
           '-vf', f'scale={VID_SIZE}:{VID_SIZE}',
           '-c:v', 'libx264', '-crf', str(VID_CRF), '-preset', 'veryfast',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', dst]
    return subprocess.run(cmd).returncode == 0 and os.path.exists(dst)


def frame_name(item):
    return os.path.basename(item['image'])


def src_image(scene_name, fname):
    return os.path.join(IMROOT, scene_name, 'rgb_img_undistort', fname)


def main():
    os.makedirs(IMG_OUT, exist_ok=True)
    os.makedirs(VID_OUT, exist_ok=True)
    data = json.load(open(QA_JSON))
    by_scene = collections.defaultdict(lambda: collections.defaultdict(list))
    for x in data:
        by_scene[x['scene_name']][frame_name(x)].append(x)

    types = ['object', 'place', 'color', 'object nature', 'number', 'other']
    out_scenes = []
    for sid, scene_name, label in SCENES:
        frames = by_scene.get(scene_name)
        if not frames:
            print(f'[skip] {scene_name}: no QA'); continue
        # richest frames first
        ranked = sorted(frames.items(), key=lambda kv: -len(kv[1]))[:FRAMES_PER_SCENE]
        ranked.sort(key=lambda kv: kv[0])   # chronological for display
        os.makedirs(os.path.join(IMG_OUT, sid), exist_ok=True)
        os.makedirs(os.path.join(VID_OUT, sid), exist_ok=True)
        scene_dir = os.path.join(IMROOT, scene_name, 'rgb_img_undistort')
        n_avail = len([f for f in os.listdir(scene_dir) if f.endswith('.jpg')])
        frames_out = []
        n_vid = 0
        for fname, items in ranked:
            src = src_image(scene_name, fname)
            if not os.path.exists(src):
                continue
            im = Image.open(src).convert('RGB')
            if max(im.size) > IMG_SIZE:
                im = im.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            im.save(os.path.join(IMG_OUT, sid, fname), quality=IMG_Q, optimize=True)
            qa = [dict(type=it['type'], q=it['question'], a=it['answer'],
                       obj=it.get('objects', []), rel=it.get('relation'))
                  for it in items]
            # stable order: by type then question
            qa.sort(key=lambda q: (types.index(q['type']) if q['type'] in types else 9, q['q']))
            entry = dict(img=f'{sid}/{fname}', qa=qa)
            # short ego video clip centred on this QA frame
            stem = os.path.splitext(fname)[0]
            vdst = os.path.join(VID_OUT, sid, stem + '.mp4')
            if encode_clip(scene_dir, int(stem.split('_')[-1]), n_avail, vdst):
                entry['vid'] = f'{sid}/{stem}.mp4'
                n_vid += 1
            frames_out.append(entry)
        n_qa = sum(len(f['qa']) for f in frames_out)
        out_scenes.append(dict(id=sid, label=label, frames=frames_out))
        print(f'[{sid}] {label}: {len(frames_out)} frames ({n_vid} clips), {n_qa} QA')

    json.dump(dict(scenes=out_scenes, types=types),
              open(os.path.join(OUT, 'semantic.json'), 'w'), separators=(',', ':'))
    # report size
    def dirsize(d):
        return sum(os.path.getsize(os.path.join(dp, f))
                   for dp, _, fs in os.walk(d) for f in fs)
    print(f'wrote {OUT}  images={dirsize(IMG_OUT)//1024}KB  video={dirsize(VID_OUT)//1024}KB')


if __name__ == '__main__':
    main()
