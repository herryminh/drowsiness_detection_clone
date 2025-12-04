from pathlib import Path
import sys, cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.landmarks import FaceLandmarker, MOUTH_OUTER, MOUTH_INNER, bbox_from_points

SRC  = ROOT / 'dataset'
OUT  = ROOT / 'data_crops' / 'mouth'
IMG_SIZE = 256

def ensure_dirs():
    for split in ['train','test']:
        for cls in ['no_yawn','yawn']:
            (OUT/split/cls).mkdir(parents=True, exist_ok=True)

def process_split(split, lmk):
    for cls in ['no_yawn','yawn']:
        for p in (SRC/split/cls).glob('*.*'):
            img = cv2.imread(str(p))
            if img is None: continue
            pts = lmk.detect(img)
            if pts is None: continue
            mouth_pts = np.vstack([pts[MOUTH_OUTER], pts[MOUTH_INNER]])
            x1,y1,x2,y2 = bbox_from_points(mouth_pts, scale=1.3, img_shape=img.shape)
            crop = img[y1:y2, x1:x2]
            if crop.size==0: continue
            crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(str(OUT/split/cls/p.name), crop)

if __name__=='__main__':
    ensure_dirs()
    lmk = FaceLandmarker(static=True, max_faces=1)
    process_split('train', lmk)
    process_split('test', lmk)
    print('Mouth crops (landmark-based) saved to', OUT)



 