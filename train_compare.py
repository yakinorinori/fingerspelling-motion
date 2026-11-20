"""
手順2：モデル2種類（1D-CNN・Transformer）× 特徴量2種類（base・base+A）を交差検証で比較する

  - 全組み合わせで同じ分割（fold）を使う
  - 各 fold では、学習データの一部を検証用に取り分けて最良エポックを選び、テスト用 fold で評価する
  - 標準化（平均0・分散1）は学習データだけで計算する

出力（動作モデル/results/）:
  summary_folds.csv        fold ごとの正解率・マクロF1
  summary.csv              組み合わせごとの平均±標準偏差、パラメータ数、学習時間
  confusion_<model>_<features>.png   混同行列（全 fold の合計）
  models/<model>_<features>.pt       --final を付けたとき：全データで学習したモデル（recognize_video.py で使う）

使い方:
    python train_compare.py                       # 4つの組み合わせを5分割交差検証
    python train_compare.py --final               # 比較のあと、全データで学習したモデルも保存
    python train_compare.py --group-regex "^(ASL_[A-E]|ubmoji)"   # ファイル名から人などのグループを取り出して分割
"""
import argparse
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Mac(MPS)で未対応の処理は CPU で計算
import csv
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch import nn

from common import CACHE_DIR, NUM_CLASSES, RESULT_DIR, load_index
from features import FEATURE_SETS, base_features, compute_features, motion_a_features
from models import MODELS, build_model, count_params


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(feature_sets):
    """全動画について、特徴量セットごとの (T,D) 配列を作る"""
    rows = load_index()
    feats = {fs: [] for fs in feature_sets}
    keep = []
    for r in rows:
        d = np.load(CACHE_DIR / r["npz"])
        lm = d["lm"]
        if len(lm) == 0 or np.isnan(lm).any(axis=(1, 2)).all():
            print(f"[注意] 手が1フレームも検出できなかったので除外: {r['video']}")
            continue
        base = base_features(lm)  # 既存125次元は1回だけ計算して使い回す
        for fs in feature_sets:
            if fs == "base":
                feats[fs].append(base)
            elif fs == "base+A":
                feats[fs].append(np.concatenate([base, motion_a_features(lm, int(d["width"]), int(d["height"]))], 1))
            else:
                feats[fs].append(compute_features(lm, int(d["width"]), int(d["height"]), fs, float(d["fps"])))
        if len(keep) % 500 == 0:
            print(f"  特徴量を計算中… {len(keep)}/{len(rows)}")
        keep.append(r)
    labels = np.array([r["label"] for r in keep])
    return keep, feats, labels


def make_groups(rows, regex):
    if not regex:
        return np.arange(len(rows))  # 動画ごと
    pat = re.compile(regex)
    groups = []
    for r in rows:
        m = pat.search(Path(r["video"]).name)
        groups.append(m.group(1) if m and m.groups() else (m.group(0) if m else "other"))
    return np.array(groups)


class Scaler:
    def fit(self, seqs):
        x = np.concatenate(seqs)
        self.mean, self.std = x.mean(0), x.std(0) + 1e-6
        return self

    def __call__(self, seq):
        return ((seq - self.mean) / self.std).astype(np.float32)


def batches(seqs, labels, idx, batch_size, shuffle):
    idx = np.array(idx)
    if shuffle:
        idx = np.random.permutation(idx)
    for i in range(0, len(idx), batch_size):
        b = idx[i:i + batch_size]
        T = max(len(seqs[j]) for j in b)
        x = np.zeros((len(b), T, seqs[b[0]].shape[1]), dtype=np.float32)
        mask = np.zeros((len(b), T), dtype=bool)
        for k, j in enumerate(b):
            x[k, :len(seqs[j])] = seqs[j]
            mask[k, :len(seqs[j])] = True
        yield torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(labels[b])


def predict(model, seqs, idx, batch_size, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for x, m, _ in batches(seqs, np.zeros(len(seqs), dtype=np.int64), idx, batch_size, False):
            probs.append(torch.softmax(model(x.to(device), m.to(device)), 1).cpu().numpy())
    return np.concatenate(probs) if probs else np.zeros((0, NUM_CLASSES))


def train_one(model_name, raw_seqs, labels, train_idx, val_idx, args, device):
    """学習して、検証データで最良だったエポックの重みを返す"""
    scaler = Scaler().fit([raw_seqs[i] for i in train_idx])
    seqs = [scaler(s) for s in raw_seqs]
    set_seed(args.seed)
    model = build_model(model_name, seqs[0].shape[1], NUM_CLASSES).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # クラスの数の偏りを損失の重みで補正
    counts = np.bincount(labels[train_idx], minlength=NUM_CLASSES).astype(np.float32)
    weight = torch.tensor(counts.sum() / (NUM_CLASSES * np.maximum(counts, 1)), device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    best = (-1.0, 0, None)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, m, y in batches(seqs, labels, train_idx, args.batch, True):
            opt.zero_grad()
            loss = loss_fn(model(x.to(device), m.to(device)), y.to(device))
            loss.backward()
            opt.step()
        if len(val_idx):
            acc = accuracy_score(labels[val_idx], predict(model, seqs, val_idx, args.batch, device).argmax(1))
        else:
            acc = epoch  # 検証データが無いときは最後のエポック
        if acc > best[0]:
            best = (acc, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    return model, scaler, seqs, best[1]


def split_train_val(train_idx, labels, groups, seed, val_ratio=0.15):
    """学習データから検証用を取り分ける（同じグループは片方にだけ入れる）"""
    n = max(2, int(round(1 / val_ratio)))
    y = labels[train_idx]
    if np.bincount(y).min() < n:
        n = max(2, int(np.bincount(y)[np.bincount(y) > 0].min()))
    if n < 2:
        return train_idx, np.array([], dtype=int)
    n_groups = len(np.unique(groups[train_idx]))
    if n_groups >= 2:
        splitter = StratifiedGroupKFold(n_splits=min(n, n_groups), shuffle=True, random_state=seed)
        tr, va = next(splitter.split(train_idx, y, groups[train_idx]))
    else:  # 学習側にグループが1つしかないときは動画単位で取り分ける（エポック選びにだけ使う）
        tr, va = next(StratifiedKFold(n_splits=n, shuffle=True, random_state=seed).split(train_idx, y))
    return train_idx[tr], train_idx[va]


def plot_confusion(cm, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5, 4))
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="1D-CNN と Transformer を特徴量ごとに比較する")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--features", nargs="+", default=FEATURE_SETS, choices=FEATURE_SETS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--group-regex", default="", help="ファイル名からグループ（人など）を取り出す正規表現。省略時は動画ごと")
    parser.add_argument("--final", action="store_true", help="比較のあと、全データで学習したモデルを保存する")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
                        help="計算に使う装置。Mac の GPU（mps）でエラーが出るときは cpu")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"デバイス: {device}")
    rows, feats, labels = load_dataset(args.features)
    groups = make_groups(rows, args.group_regex)
    print(f"動画 {len(rows)} 本, クラスごとの本数 {np.bincount(labels, minlength=NUM_CLASSES).tolist()}, "
          f"グループ数 {len(set(groups))}")

    n_folds = min(args.folds, int(np.bincount(labels)[np.bincount(labels) > 0].min()), len(set(groups)))
    if n_folds < 2:
        raise SystemExit("クラスごとの本数が少なすぎて交差検証ができません")
    if n_folds < args.folds:
        print(f"[注意] 本数が少ないので {n_folds} 分割にします")
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=args.seed) \
        if args.group_regex else StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
    folds = list(splitter.split(np.zeros(len(labels)), labels, groups if args.group_regex else None))

    RESULT_DIR.mkdir(exist_ok=True)
    fold_rows, summary = [], []
    combos = [(m, fs) for fs in args.features for m in args.models]
    for model_name, fs in combos:
        name = f"{model_name}_{fs}"
        accs, f1s, epochs, all_true, all_pred = [], [], [], [], []
        t0 = time.time()
        for k, (train_idx, test_idx) in enumerate(folds, 1):
            tr, va = split_train_val(train_idx, labels, groups, args.seed + k)
            model, scaler, seqs, best_epoch = train_one(model_name, feats[fs], labels, tr, va, args, device)
            pred = predict(model, seqs, test_idx, args.batch, device).argmax(1)
            acc = accuracy_score(labels[test_idx], pred)
            f1 = f1_score(labels[test_idx], pred, average="macro", labels=range(NUM_CLASSES), zero_division=0)
            accs.append(acc), f1s.append(f1), epochs.append(best_epoch)
            all_true.extend(labels[test_idx]), all_pred.extend(pred)
            fold_rows.append({"model": model_name, "features": fs, "fold": k, "accuracy": acc,
                              "macro_f1": f1, "best_epoch": best_epoch})
            print(f"{name:24s} fold {k}/{n_folds}: 正解率 {acc:.3f}, マクロF1 {f1:.3f}（最良エポック {best_epoch}）")
        params = count_params(build_model(model_name, feats[fs][0].shape[1], NUM_CLASSES))
        summary.append({"model": model_name, "features": fs, "dim": feats[fs][0].shape[1],
                        "accuracy_mean": np.mean(accs), "accuracy_std": np.std(accs),
                        "macro_f1_mean": np.mean(f1s), "macro_f1_std": np.std(f1s),
                        "params": params, "train_sec": time.time() - t0,
                        "median_best_epoch": int(np.median(epochs))})
        cm = confusion_matrix(all_true, all_pred, labels=range(NUM_CLASSES))
        plot_confusion(cm, f"{model_name} / {fs}  acc {np.mean(accs):.3f}", RESULT_DIR / f"confusion_{name}.png")

    for path, data in [(RESULT_DIR / "summary_folds.csv", fold_rows), (RESULT_DIR / "summary.csv", summary)]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)

    print("\n=== 結果（平均 ± 標準偏差）===")
    print(f"{'モデル':12s} {'特徴量':8s} {'次元':>4s} {'正解率':>14s} {'マクロF1':>14s} {'パラメータ':>10s} {'時間[s]':>8s}")
    for s in summary:
        print(f"{s['model']:12s} {s['features']:8s} {s['dim']:4d} "
              f"{s['accuracy_mean']:.3f} ± {s['accuracy_std']:.3f}  {s['macro_f1_mean']:.3f} ± {s['macro_f1_std']:.3f} "
              f"{s['params']:10,d} {s['train_sec']:8.1f}")
    print(f"\n保存先: {RESULT_DIR}")

    if args.final:
        (RESULT_DIR / "models").mkdir(exist_ok=True)
        all_idx = np.arange(len(labels))
        for s in summary:
            model_name, fs = s["model"], s["features"]
            tr, va = split_train_val(all_idx, labels, groups, args.seed)
            model, scaler, _, best_epoch = train_one(model_name, feats[fs], labels, tr, va, args, device)
            path = RESULT_DIR / "models" / f"{model_name}_{fs}.pt"
            torch.save({"model": model_name, "features": fs, "in_dim": int(feats[fs][0].shape[1]),
                        "num_classes": NUM_CLASSES, "state_dict": model.state_dict(),
                        "mean": scaler.mean, "std": scaler.std, "best_epoch": best_epoch,
                        "cv_accuracy": s["accuracy_mean"]}, path)
            print(f"全データで学習したモデルを保存: {path}")
        with open(RESULT_DIR / "models" / "train_args.json", "w") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
