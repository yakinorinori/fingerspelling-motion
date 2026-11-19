"""
手順1：動画から MediaPipe Hands のランドマークを取り出して保存する（時間がかかるのは最初の1回だけ）

入力:  Fingerspelling/Video/transformer訓練/手の動き 5クラス分類/クラスN/かな/動画
出力:  動作モデル/cache/landmarks/クラスN/かな/動画名.npz   （ランドマーク配列）
       動作モデル/cache/index.csv                          （動画の一覧とラベル）

特徴量はこの保存結果から計算するので、特徴量を変えても MediaPipe をやり直す必要はない。

使い方:
    python extract_landmarks.py
    python extract_landmarks.py --data "/Users/.../手の動き 5クラス分類"
    python extract_landmarks.py --redo      # 保存済みもやり直す
"""
import argparse
import csv

import cv2
import numpy as np

from common import (CACHE_DIR, DEFAULT_DATA_DIR, INDEX_CSV, LANDMARK_DIR, VIDEO_EXTS,
                    class_label, resolve_dir)


def extract_video(video_path, flip=False):
    """動画 → (T,21,3) のランドマーク（手が無いフレームは NaN）、幅、高さ、fps"""
    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        raise SystemExit("この mediapipe には mp.solutions がありません。既存のコードと同じバージョンの環境で実行してください")
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    # 既存の Feature_static_move.py / eval_transformer.py と同じ設定
    with mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                  min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if flip:
                frame = cv2.flip(frame, 1)
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark
                frames.append([[p.x, p.y, p.z] for p in lm])
            else:
                frames.append(np.full((21, 3), np.nan).tolist())
    cap.release()
    lm = np.asarray(frames, dtype=np.float32).reshape(-1, 21, 3)
    return lm, width, height, float(fps)


def main():
    parser = argparse.ArgumentParser(description="動画から手のランドマークを取り出して保存する")
    parser.add_argument("--data", default=str(DEFAULT_DATA_DIR), help="手の動き 5クラス分類 フォルダ")
    parser.add_argument("--redo", action="store_true", help="保存済みの動画もやり直す")
    parser.add_argument("--flip", action="store_true", help="左右反転してから検出する")
    args = parser.parse_args()

    data_dir = resolve_dir(args.data)
    if data_dir is None:
        raise SystemExit(f"データのフォルダが見つかりません: {args.data}（--data で指定できます）")

    rows = []
    for class_dir in sorted(d for d in data_dir.iterdir() if d.is_dir()):
        label = class_label(class_dir.name)
        if label is None:
            print(f"[注意] クラス番号が読めないフォルダを飛ばします: {class_dir.name}")
            continue
        videos = sorted(p for p in class_dir.rglob("*")
                        if p.suffix.lower() in VIDEO_EXTS and not p.name.startswith("."))
        print(f"{class_dir.name}（ラベル {label}）: {len(videos)} 本")
        for v in videos:
            rel = v.relative_to(data_dir)
            kana = rel.parts[1] if len(rel.parts) > 2 else ""
            out = LANDMARK_DIR / rel.with_suffix(".npz")
            if args.redo or not out.exists():
                lm, w, h, fps = extract_video(v, args.flip)
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out, lm=lm, width=w, height=h, fps=fps)
                rate = (~np.isnan(lm).any(axis=(1, 2))).mean() if len(lm) else 0
                print(f"  {rel}  {len(lm)} フレーム, 手の検出率 {rate:.0%}")
            rows.append({"npz": str(out.relative_to(CACHE_DIR)), "video": str(rel),
                         "label": label, "kana": kana})

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(INDEX_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["npz", "video", "label", "kana"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} 本の一覧を保存しました: {INDEX_CSV}")


if __name__ == "__main__":
    main()
