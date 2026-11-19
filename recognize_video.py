"""
手順3：実際の動画を入れて認識結果を見る（eval_transformer.py と同じ使い方）

  - 動画全体での予測（学習と同じ条件）
  - 窓をずらしながらの予測（--window フレームずつ、--stride ずつずらす）
を動画に重ねて表示し、<動画名>_<モデル名>_labeled.mp4 として results/recognized/ に保存する。

使い方:
    python recognize_video.py --model results/models/cnn_base+A.pt 動画.mp4
    python recognize_video.py --model results/models/transformer_base.pt フォルダ/
    python recognize_video.py --model ... 動画.mp4 --window 30 --stride 5 --no-show
"""
import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

from common import CLASS_NAMES, RESULT_DIR, VIDEO_EXTS
from extract_landmarks import extract_video
from features import compute_features
from models import build_model


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["model"], ckpt["in_dim"], ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def run(model, seq, device):
    x = torch.from_numpy(seq[None].astype(np.float32)).to(device)
    m = torch.ones(1, len(seq), dtype=torch.bool, device=device)
    with torch.no_grad():
        return torch.softmax(model(x, m), 1)[0].cpu().numpy()


def draw(frame, lines, probs):
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (20, 45 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5)
        cv2.putText(frame, text, (20, 45 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
    if probs is not None:  # 窓の予測確率を棒グラフで表示
        y0 = 45 + 40 * len(lines)
        for c, p in enumerate(probs):
            y = y0 + 26 * c
            cv2.rectangle(frame, (20, y - 16), (20 + int(200 * p), y + 2), (0, 200, 255), -1)
            cv2.putText(frame, f"{CLASS_NAMES[c]} {p:.2f}", (230, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def process(video, model, ckpt, device, args):
    print(f"\n動画: {video}")
    lm, w, h, fps = extract_video(video, args.flip)
    if len(lm) == 0:
        print("  フレームが読めませんでした")
        return
    rate = (~np.isnan(lm).any(axis=(1, 2))).mean()
    seq = compute_features(lm, w, h, ckpt["features"], fps)
    seq = (seq - ckpt["mean"]) / ckpt["std"]

    whole = run(model, seq, device)
    print(f"  手の検出率 {rate:.0%}, {len(lm)} フレーム")
    print(f"  動画全体の予測: {CLASS_NAMES[int(whole.argmax())]}  "
          f"（{', '.join(f'{c}:{p:.2f}' for c, p in enumerate(whole))}）")

    # 窓をずらしながら予測し、各フレームには「そのフレームを中心に近く含む窓」の結果を割り当てる
    frame_probs = [None] * len(seq)
    if args.window and len(seq) >= args.window:
        starts = list(range(0, len(seq) - args.window + 1, args.stride))
        win_probs = [run(model, seq[s:s + args.window], device) for s in starts]
        centers = np.array(starts) + args.window // 2
        for t in range(len(seq)):
            frame_probs[t] = win_probs[int(np.abs(centers - t).argmin())]
        timeline = [int(p.argmax()) for p in win_probs]
        print(f"  窓ごとの予測（{args.window}フレーム, {args.stride}ずつ）: {timeline}")
        print(f"  最も多いクラス: {CLASS_NAMES[Counter(timeline).most_common(1)[0][0]]}")

    out_dir = RESULT_DIR / "recognized"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(video).stem}_{ckpt['model']}_{ckpt['features']}_labeled.mp4"
    cap = cv2.VideoCapture(str(video))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    t = 0
    while True:
        ok, frame = cap.read()
        if not ok or t >= len(seq):
            break
        if args.flip:
            frame = cv2.flip(frame, 1)
        lines = [(f"Video: {CLASS_NAMES[int(whole.argmax())]} ({whole.max():.2f})", (0, 255, 0))]
        if frame_probs[t] is not None:
            lines.append((f"Window: {CLASS_NAMES[int(frame_probs[t].argmax())]}", (0, 200, 255)))
        if np.isnan(lm[t]).any():
            lines.append(("no hand", (0, 0, 255)))
        draw(frame, lines, frame_probs[t])
        writer.write(frame)
        if not args.no_show:
            cv2.imshow("recognize", frame)
            if cv2.waitKey(int(1000 / fps)) == 27:  # Esc で次の動画へ
                break
        t += 1
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"  保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="動画を入れて動作モデルの認識結果を見る")
    parser.add_argument("inputs", nargs="+", help="動画ファイル、または動画の入ったフォルダ")
    parser.add_argument("--model", required=True, help="train_compare.py --final で保存した .pt")
    parser.add_argument("--window", type=int, default=30, help="窓の長さ（0 で窓ごとの予測をしない）")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--flip", action="store_true", help="左右反転してから検出する（学習時と合わせる）")
    parser.add_argument("--no-show", action="store_true", help="画面に表示せず保存だけする")
    args = parser.parse_args()

    device = torch.device("cpu")
    model, ckpt = load_model(args.model, device)
    print(f"モデル: {ckpt['model']} / 特徴量: {ckpt['features']}（交差検証の正解率 {ckpt['cv_accuracy']:.3f}）")
    videos = []
    for p in map(Path, args.inputs):
        if p.is_dir():
            videos += sorted(v for v in p.rglob("*") if v.suffix.lower() in VIDEO_EXTS and not v.name.startswith("."))
        elif p.is_file():
            videos.append(p)
        else:
            print(f"見つかりません: {p}")
    for v in videos:
        process(v, model, ckpt, device, args)


if __name__ == "__main__":
    main()
