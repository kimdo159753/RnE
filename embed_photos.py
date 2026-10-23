#!/usr/bin/env python3
"""공고 사진 임베딩 (사전학습 모델 고정, 학습 안 함) → emb_cache/photo_<모델>[_<태그>].npz

  python embed_photos.py --dir photos_all/crop                       # 3개 모델 모두
  python embed_photos.py --dir photo_clean_all/text_nobg --tag tn    # 처리한 사진으로
  python embed_photos.py --dir photos_all/crop --models dinov2 --limit 500   # 시험

모델 (CLAUDE.md 계획 4)
  dinov2    facebook/dinov2-small CLS (384)   ViT, 털·모양 세부
  clip      open_clip ViT-B-32 openai (512)   ViT, "귀여운·아파 보이는" 같은 느낌
  convnext  timm convnext_tiny (768)           CNN, 비교 기준
저장: x(float16, L2 정규화), ids(desertionNo 문자열). PCA는 compare_photo.py가 학습 구간에서만 맞춘다.
"""
import argparse
import os
import time

import numpy as np


def build(name, device):
    """(모델, PIL → 텐서 변환, 텐서 배치 → 특징) 반환"""
    import torch
    if name == "dinov2":
        from torchvision import transforms as T
        from transformers import AutoModel
        model = AutoModel.from_pretrained("facebook/dinov2-small").eval().to(device)
        # transformers의 dinov2 전처리와 같음(짧은 변 256 bicubic → 가운데 224, ImageNet 정규화)
        tf = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return tf, lambda x: model(pixel_values=x).pooler_output
    if name == "clip":
        import open_clip
        model, _, tf = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        model = model.eval().to(device)
        return tf, model.encode_image
    if name == "convnext":
        import timm
        model = timm.create_model("convnext_tiny", pretrained=True, num_classes=0).eval().to(device)
        tf = timm.data.create_transform(**timm.data.resolve_data_config({}, model=model))
        return tf, model
    raise ValueError(name)


class Photos:
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="photos_all/crop")
    ap.add_argument("--models", nargs="+", default=["dinov2", "clip", "convnext"])
    ap.add_argument("--tag", default="", help="출력 파일 이름 뒤에 붙일 태그(처리한 사진 구분용)")
    ap.add_argument("--out-dir", default="emb_cache")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="시험용: 앞에서 n장만")
    a = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ids = sorted(f[:-4] for f in os.listdir(a.dir) if f.endswith(".jpg"))
    if a.limit:
        ids = ids[:a.limit]
    paths = [os.path.join(a.dir, d + ".jpg") for d in ids]
    os.makedirs(a.out_dir, exist_ok=True)
    print(f"사진 {len(ids):,}장 ({a.dir}), 장치 {device}", flush=True)

    for name in a.models:
        out = os.path.join(a.out_dir, f"photo_{name}{'_' + a.tag if a.tag else ''}.npz")
        if os.path.exists(out) and not a.limit:
            print(f"[{name}] 이미 있음 → 건너뜀 ({out})", flush=True)
            continue
        tf, feat = build(name, device)
        dl = DataLoader(Photos(paths, tf), batch_size=a.batch, num_workers=a.workers, pin_memory=device == "cuda")
        xs, t0 = [], time.time()
        for i, x in enumerate(dl, 1):
            with torch.autocast(device, enabled=device == "cuda"):
                xs.append(feat(x.to(device, non_blocking=True)).float().cpu().numpy())
            if i % 100 == 0:
                done = min(i * a.batch, len(ids))
                print(f"  [{name}] {done:,}/{len(ids):,} ({done / (time.time() - t0):.0f}장/초)", flush=True)
        x = np.concatenate(xs)
        x = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16)
        np.savez(out, x=x, ids=np.array(ids, dtype=str))
        print(f"[{name}] {x.shape} 저장 {out} ({time.time() - t0:.0f}초)", flush=True)
        del feat
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
